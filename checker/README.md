# 대회 제출 ZIP Docker 검증 환경

이 디렉터리는 LG Aimers 평가 서버와 최대한 비슷한 Docker 환경에서 제출 ZIP을 실제로 실행하고, `checker.py`로 결과를 검증하기 위한 도구입니다.

## 무엇을 검증하나요?

이 검증은 모델을 다시 학습하지 않습니다. 제출 ZIP의 `model/*.pkl` 등 이미 학습된 모델 파일을 불러와 공개 테스트 데이터에 바로 추론합니다.

실제 실행 순서는 다음과 같습니다.

```text
제출 ZIP 구조 검사
  → ZIP의 requirements.txt 패키지 설치
  → ZIP 압축 해제
  → model/*.pkl 등 저장된 모델 로드
  → data/test.csv 전처리 및 추론
  → output/submission.csv 생성
  → 컬럼, 행 수, row_id 순서, 확률값 검증
```

따라서 다음과 같은 실제 제출 실패를 미리 찾을 수 있습니다.

- 평가 환경에서 `.pkl` 모델이 정상적으로 로드되지 않는 문제
- 학습 환경과 패키지 버전이 달라 발생하는 호환성 문제
- 필요한 패키지가 `requirements.txt`에 빠진 문제
- 모델이 기대하는 특성과 추론 시 생성한 특성이 다른 문제
- 로컬 절대 경로나 인터넷 다운로드에 의존하는 문제
- `submission.csv`의 경로, 컬럼, 행 순서 또는 확률값이 잘못된 문제

## 재현하는 평가 환경

- Ubuntu 22.04
- Python 3.11.15
- CUDA 12.8
- PyTorch 2.7.1+cu128
- 대회에서 안내한 Python 및 시스템 패키지
- CPU 6개: quota와 affinity `0-5` 모두 제한
- RAM 28GiB
- 공유 메모리 4GiB
- 추론 실행 중 인터넷 차단: `--network none`
- 제출 ZIP과 테스트 데이터는 읽기 전용 마운트

이미지를 만들 때 서버 기본 패키지를 먼저 설치하고, 제출 ZIP 루트의 `requirements.txt`를 추가로 설치합니다. 제출 요구사항에 다른 버전이 명시되어 있으면 해당 버전이 서버 기본 버전을 대체합니다.

## 사전 준비

1. Docker Desktop을 실행합니다.
2. Docker Desktop에 `Engine running`이 표시될 때까지 기다립니다.
3. 프로젝트 루트에 제출 ZIP과 검증용 데이터가 있는지 확인합니다.

기본 파일 배치는 다음과 같습니다.

```text
aimers_projects/
├─ baseline_submit.zip
├─ checker.py
├─ checker/
│  ├─ Dockerfile
│  └─ run-checker.ps1
└─ data/
   ├─ test.csv
   └─ sample_submission.csv
```

제출 ZIP 루트에는 다음 항목이 있어야 합니다.

```text
submission.zip
├─ script.py
├─ requirements.txt
└─ model/
   └─ 모델 파일
```

ZIP 안에 `submission/` 같은 상위 폴더가 한 번 더 들어가면 안 됩니다.

## 기본 검증 방법

PowerShell에서 프로젝트 루트로 이동한 후 실행합니다.

```powershell
./checker/run-checker.ps1 ./baseline_submit.zip
```

ZIP 이름을 첫 번째 인자로 지정하면 됩니다. 프로젝트 루트에 있는 파일은 다음 세 표현을 모두 지원합니다.

```powershell
./checker/run-checker.ps1 submit_v3.zip -DataDir ./data
./checker/run-checker.ps1 ./submit_v3.zip -DataDir ./data
./checker/run-checker.ps1 /submit_v3.zip -DataDir ./data
```

PowerShell에서 일반적으로 권장하는 상대경로 표기는 `./submit.zip` 또는 `.\submit.zip`입니다. 존재하지 않는 파일을 입력하면 프로젝트에서 발견한 ZIP 목록을 오류 메시지에 함께 보여줍니다.

ZIP이 하위 폴더에 있더라도 같은 파일명이 하나뿐이면 파일명만 입력해 자동으로 찾을 수 있습니다. 경로가 분명한 경우에는 다음과 같이 전체 상대경로를 지정하는 것이 가장 안전합니다.

```powershell
./checker/run-checker.ps1 ./test/submit_v3.zip -DataDir ./data
```

이 명령은 다음 두 작업을 모두 수행합니다.

1. 대회 환경 Docker 이미지를 만들고 제출 ZIP의 패키지를 설치합니다.
2. 인터넷이 차단된 컨테이너에서 `checker.py`와 실제 추론을 실행합니다.

첫 빌드에서는 CUDA 이미지 다운로드, Python 3.11.15 컴파일, CUDA PyTorch 설치가 필요하므로 오래 걸리고 디스크를 많이 사용할 수 있습니다. 같은 환경을 다시 실행할 때는 Docker 캐시가 재사용됩니다.

## 같은 ZIP 빠르게 다시 검사하기

이미지를 만든 뒤 코드나 ZIP이 바뀌지 않았다면 빌드를 생략할 수 있습니다.

```powershell
./checker/run-checker.ps1 ./baseline_submit.zip -NoBuild
```

주의: ZIP의 `requirements.txt` 또는 모델 파일을 변경했다면 `-NoBuild`를 사용하지 말고 이미지를 다시 빌드해야 합니다.

## 다른 ZIP 또는 데이터 경로 검사하기

```powershell
./checker/run-checker.ps1 ./submission/my_model.zip -DataDir ./data
```

제출 ZIP은 Docker 빌드 컨텍스트에 포함될 수 있도록 프로젝트 루트 아래에 있어야 합니다. `DataDir`에는 반드시 다음 파일이 필요합니다.

- `test.csv`
- `sample_submission.csv`

## GPU 검증

로컬 NVIDIA GPU를 컨테이너에 연결하려면 다음과 같이 실행합니다.

```powershell
./checker/run-checker.ps1 ./baseline_submit.zip -Gpu
```

GPU 검증에는 다음 조건이 필요합니다.

- NVIDIA GPU와 호환 드라이버
- 최신 WSL2 커널
- Docker Desktop의 WSL2 백엔드
- Docker에서 NVIDIA GPU가 정상 인식되는 상태

`WSL environment detected but no adapters were found` 오류가 나오면 제출 코드 문제가 아니라 Windows/WSL에서 NVIDIA GPU가 컨테이너에 연결되지 않은 상태입니다.

CPU 모델(RandomForest, LightGBM CPU 설정 등)은 `-Gpu` 없이도 실제 추론 경로를 전부 검증할 수 있습니다. GPU 모델은 `-Gpu`로 CUDA 실행 여부를 확인해야 합니다.

로컬 GPU가 정상 연결되더라도 평가 서버의 NVIDIA L4와 GPU 종류가 다르면 실행시간과 최대 VRAM 사용량은 완전히 같지 않을 수 있습니다.

## 검사 성공 기준

마지막에 다음과 같이 출력되면 제출 ZIP의 구조, 모델 로드, 추론 및 출력 검증을 통과한 것입니다.

```text
[7/7] final report
Passes   : 32
Warnings : 1
Errors   : 0

READY TO SUBMIT
```

판정 의미는 다음과 같습니다.

- `PASS`: 해당 검사를 통과했습니다.
- `WARN`: 실행은 가능하지만 제출 전에 내용을 확인해야 합니다.
- `ERROR`: 제출 실패 가능성이 있으므로 수정해야 합니다.
- `READY TO SUBMIT`: 모든 필수 검사를 통과했습니다.
- `NOT READY TO SUBMIT`: 하나 이상의 필수 검사에서 실패했습니다.

패키지가 서버 기본 버전과 다르다는 경고가 나와도, 해당 버전이 제출 ZIP의 `requirements.txt`에 의도적으로 명시되어 있고 컨테이너 추론까지 성공했다면 실제 설치 및 실행 호환성은 확인된 것입니다.

Windows의 `Compress-Archive` 등으로 만든 ZIP은 `ZIP member has no Unix permission metadata` 경고가 나올 수 있습니다. CRC 검사, 안전한 압축 해제와 필수 파일 확인을 통과했다면 이 메타데이터 경고만으로 제출을 막지는 않습니다. `script.py`는 실행 파일 권한으로 직접 실행하는 것이 아니라 `python script.py`로 실행됩니다.

## 이 검증으로 알 수 없는 것

로컬의 `data/test.csv`는 공개 검증용 데이터이므로 비공개 전체 테스트 데이터 자체를 재현할 수는 없습니다. 따라서 다음 항목에는 차이가 남을 수 있습니다.

- 비공개 전체 데이터에서의 총 실행시간
- 전체 데이터 처리 시 최대 CPU RAM 및 GPU VRAM
- 로컬 GPU와 NVIDIA L4의 성능 차이
- 평가 서버 내부 구현의 공개되지 않은 세부 설정

하지만 저장된 모델 로드, 전처리, 추론, 파일 생성이라는 실제 제출 실행 경로는 동일하게 검사합니다.

## 주요 파일

- `Dockerfile`: Ubuntu, Python, CUDA 및 패키지 환경을 구성합니다.
- `base-requirements.txt`: 평가 서버 기본 Python 패키지 버전입니다.
- `container-entrypoint.sh`: 컨테이너 환경을 출력하고 `checker.py`를 실행합니다.
- `run-checker.ps1`: 이미지 빌드, 자원 제한, 마운트 및 실행을 자동화합니다.
- `../checker.py`: ZIP 구조, 실행 결과와 `submission.csv`를 검증합니다.
