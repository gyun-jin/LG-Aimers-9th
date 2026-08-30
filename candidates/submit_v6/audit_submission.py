"""제출물(submit 폴더) 대회 규칙 자동 검사.

사용: python scripts/audit_submission.py <submit_vN 폴더 경로>
검사 항목:
  1. 행 독립성  - 단독행 vs 전체 / 순서뒤집기 / 부분집합 예측 동일 (규칙 3 검증법)
  2. 원격 API   - script.py에 네트워크/외부 API import·호출 없음
  3. 외부 데이터 - script.py가 여는 파일이 test/sample_submission/model 뿐, zip에 외부파일 없음
  4. 사전학습    - requirements에 사전학습 가중치 패키지 없음
  5. 재현성     - requirements에 numpy/scipy/scikit-learn/joblib 버전 핀 존재
  6. 실행       - 클린 추출 후 script.py 정상 실행, 출력 형식/NaN/범위 검사
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# script.py를 같은 프로세스에서 불러오는 행 독립성 검사도 실제 제출 환경과
# 같은 Windows DLL 초기화 순서를 사용한다.
import lightgbm as _lightgbm_import_guard  # noqa: F401

import joblib
import numpy as np
import pandas as pd

DATA = str(Path(__file__).resolve().parents[2] / 'data')
FORBIDDEN_IMPORTS = r'requests|urllib|http\.client|socket|openai|google\.generativeai|anthropic|huggingface|transformers|torch\.hub|from_pretrained|boto3|gdown'
PRETRAINED_PKGS = r'^(transformers|huggingface|timm|torchvision|sentence-transformers|open_clip)'
REQUIRED_PINS = ['numpy', 'scipy', 'scikit-learn', 'joblib', 'pandas']

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}')


def main(sub_dir):
    sub_dir = os.path.abspath(sub_dir)
    os.chdir(sub_dir)
    print(f'=== 검사 대상: {sub_dir} ===\n')

    script = open('script.py', encoding='utf-8').read()

    # ---- 2. 원격 API ----
    print('[2] 원격 API / 네트워크')
    hits = re.findall(FORBIDDEN_IMPORTS, script)
    check('네트워크·외부API 흔적 없음', not hits, f'발견={sorted(set(hits))}' if hits else '')

    # ---- 3. 외부 데이터 ----
    print('[3] 외부 데이터')
    opened = re.findall(r'(?:read_csv|joblib\.load|open)\(\s*([^)]+)\)', script)
    print(f'     script.py가 여는 대상: {opened}')
    bad = [o for o in opened if re.search(r'trackman|mapping|kbo|statiz|http', o, re.I)]
    check('외부 데이터 파일 참조 없음', not bad, f'의심={bad}' if bad else '')
    z = zipfile.ZipFile('submit.zip')
    names = z.namelist()
    extra = [n for n in names if not (n == 'script.py' or n == 'requirements.txt' or n.startswith('model/'))]
    check('zip 구성 = script.py + requirements.txt + model/ 만', not extra, f'추가파일={extra}' if extra else '')
    check('zip 경로 forward-slash', all('\\' not in n for n in names))

    # ---- 4/5. requirements ----
    print('[4][5] requirements')
    req = open('requirements.txt', encoding='utf-8').read().splitlines()
    req = [r.strip() for r in req if r.strip()]
    pre = [r for r in req if re.match(PRETRAINED_PKGS, r, re.I)]
    check('사전학습 가중치 패키지 없음', not pre, f'{pre}' if pre else '')
    pinned = {r.split('==')[0].lower(): ('==' in r) for r in req}
    missing = [p for p in REQUIRED_PINS if not pinned.get(p, False)]
    check('numpy/scipy/scikit-learn/joblib/pandas 버전 핀', not missing, f'미핀={missing}' if missing else '')

    # ---- 6. 클린 추출 실행 ----
    print('[6] 클린 추출 실행')
    tmp = tempfile.mkdtemp(prefix='audit_')
    z.extractall(tmp)
    os.makedirs(os.path.join(tmp, 'open'), exist_ok=True)
    shutil.copy(os.path.join(DATA, 'test.csv'), os.path.join(tmp, 'open', 'test.csv'))
    shutil.copy(os.path.join(DATA, 'sample_submission.csv'), os.path.join(tmp, 'open', 'sample_submission.csv'))
    r = subprocess.run([sys.executable, 'script.py'], cwd=tmp, capture_output=True, text=True)
    check('script.py 정상 종료', r.returncode == 0, (r.stderr or '')[-300:] if r.returncode else '')
    out = os.path.join(tmp, 'output', 'submission.csv')
    if os.path.exists(out):
        s = pd.read_csv(out)
        sample = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))
        check('출력 컬럼 = [row_id, control_success]', list(s.columns) == ['row_id', 'control_success'])
        check('출력 행수 = sample_submission', len(s) == len(sample))
        check('NaN 없음', s.control_success.notna().all())
        check('값 범위 [0,1]', ((s.control_success >= 0) & (s.control_success <= 1)).all())

    # ---- 1. 행 독립성 ----
    print('[1] 행 독립성 (단독 vs 전체 / 순서 / 부분집합)')
    spec = importlib.util.spec_from_file_location('scr', os.path.join(tmp, 'script.py'))
    scr = importlib.util.module_from_spec(spec)
    cwd = os.getcwd(); os.chdir(tmp)
    try:
        spec.loader.exec_module(scr)
        bundle = joblib.load('model/final_model.joblib')
        calib = joblib.load('model/calibration_model.joblib')
        df = pd.read_csv(os.path.join(DATA, 'train.csv'), encoding='utf-8-sig', nrows=200000)
        big = df.drop(columns=['control_success']).tail(2000).reset_index(drop=True)
        pick = sorted(np.random.default_rng(0).choice(len(big), 8, replace=False))

        def infer(frame):
            p = scr.apply_calibrator(calib, scr.predict_bundle(bundle, frame))
            for extra_fn in ('apply_logit_shift',):          # 후처리 함수가 있으면 동일 적용
                if hasattr(scr, extra_fn):
                    p = getattr(scr, extra_fn)(p)
            return p
        p_all = infer(big)
        p_alone = np.array([infer(big.iloc[[i]])[0] for i in pick])
        p_rev = infer(big.iloc[::-1].reset_index(drop=True))[::-1]
        p_half = infer(big.iloc[:1000])
        d1 = float(np.abs(p_all[pick] - p_alone).max())
        d2 = float(np.abs(p_all - p_rev).max())
        d3 = float(np.abs(p_all[:1000] - p_half).max())
        check('단독행 == 전체 배치', d1 < 1e-9, f'max diff={d1:.1e}')
        check('순서 뒤집기 불변', d2 < 1e-9, f'max diff={d2:.1e}')
        check('부분집합 불변', d3 < 1e-9, f'max diff={d3:.1e}')
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    print('\n=== 결과 ===')
    fails = [n for n, ok, _ in results if not ok]
    if fails:
        print(f'FAIL {len(fails)}건: {fails}')
        sys.exit(1)
    print(f'ALL PASS ({len(results)}개 항목) - 제출 가능')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1])
