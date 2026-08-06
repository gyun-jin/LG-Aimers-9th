# LG Aimers 9th

LG Aimers 9th 데이터톤을 위한 팀 협업 저장소입니다. Tabular 데이터를 기반으로
피처를 만들고 여러 모델을 실험한 뒤, 선택된 모델을 제출용 ZIP으로 관리합니다.

## 폴더 구조

```text
.
├── candidates/              # 모델별 실험 후보
│   └── baseline/            # 새 후보를 만들 때 복사하는 템플릿
├── features/                # 피처 엔지니어링 아이디어와 공용 코드
├── submission/              # 최종 제출 ZIP에 들어갈 파일
│   ├── model/               # 선택된 model.pkl
│   ├── script.py            # 추론 실행 파일
│   └── requirements.txt     # 제출 환경 라이브러리
├── CONTRIBUTING.md          # 협업 규칙
└── README.md
```

`candidates/<모델명>/`의 각 후보는 다음 파일을 독립적으로 가집니다.

```text
candidates/<모델명>/
├── model/model.pkl
├── train.py
├── script.py
├── requirements.txt
└── README.md
```

각 후보의 `train.py`와 `script.py`에는 같은 피처 생성 로직을 반영해야 합니다.
그래야 학습과 추론에서 컬럼 누락이나 순서 불일치가 발생하지 않습니다.

## 빠른 시작

`develop`에서 최신 코드를 받고, `baseline`을 복사해 새 모델 후보를 시작합니다.

```powershell
git switch develop
git pull origin develop
git switch -c model/lgbm_v1
Copy-Item -Recurse candidates\baseline candidates\lgbm_v1
```

모델 후보 폴더의 라이브러리를 설치합니다.

```powershell
pip install -r candidates\lgbm_v1\requirements.txt
```

학습 후에는 다음 항목을 갱신합니다.

1. `candidates/<모델명>/model/model.pkl`
2. `candidates/<모델명>/script.py`의 `FEATURE_COLUMNS`와 `make_features()`
3. 필요한 경우 `requirements.txt`의 라이브러리와 버전

## 제출 준비

선택된 후보의 `model/`, `script.py`, `requirements.txt`를 `submission/`에
반영합니다. 제출 전 `submission/`에서 실제 입력 파일로 추론을 실행합니다.

```powershell
python submission\script.py --input <입력_csv_경로> --output <출력_csv_경로>
```

제출 ZIP에는 `submission/` 폴더 자체가 아니라 그 안의 세 항목이 최상위에
들어가야 합니다.

```powershell
Compress-Archive -Path submission\model, submission\script.py, submission\requirements.txt -DestinationPath submission.zip
```

최종 제출본은 `main` 브랜치에만 병합합니다. 자세한 협업 규칙은
[CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
