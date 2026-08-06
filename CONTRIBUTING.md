# Contribution Guide

이 문서는 LG Aimers 9th 데이터톤 저장소의 협업 규칙입니다. 목표는 많은 실험을
빠르게 진행하면서도, 최종 제출 파일은 언제든 다시 실행할 수 있게 유지하는 것입니다.

## 기본 원칙

- `main`은 실제 제출 가능한 상태만 유지합니다. 직접 push하지 않습니다.
- 모든 개발은 `develop`에서 시작하고, PR을 통해 다시 `develop`으로 병합합니다.
- 한 브랜치와 한 PR에는 가능한 한 한 가지 목적만 담습니다.
- 모델의 학습 코드와 추론 코드는 같은 피처 생성 과정을 사용해야 합니다.

## 브랜치

| 브랜치 | 용도 |
| --- | --- |
| `main` | 검증을 마친 최종 제출본 |
| `develop` | 팀 공용 개발 및 통합 브랜치 |
| `model/<모델명>` | 특정 모델의 학습과 추론 구현 |
| `feature/<피처명>` | Tabular 피처 엔지니어링 실험 |
| `fix/<내용>` | 재현 또는 추론 오류 수정 |
| `docs/<내용>` | README 등 문서 수정 |

브랜치는 한 명이 맡는 것을 원칙으로 합니다. 새 작업은 항상 최신 `develop`에서
시작합니다.

```powershell
git switch develop
git pull origin develop
git switch -c model/lgbm_v1
```

한 브랜치를 여러 명이 함께 써야 한다면, 작업 전 `git pull --rebase`를 실행하고
누가 어떤 파일을 수정하는지 먼저 공유합니다.

## 모델 후보 작업

새 모델은 `candidates/baseline/`을 복사해 시작합니다.

```powershell
Copy-Item -Recurse candidates\baseline candidates\lgbm_v1
```

각 후보 폴더에는 아래 파일을 유지합니다.

```text
candidates/<모델명>/
├── model/model.pkl
├── train.py
├── script.py
├── requirements.txt
└── README.md
```

- `train.py`: 모델 학습 및 `model/model.pkl` 생성
- `script.py`: 제출 환경에서 모델을 불러와 추론
- `requirements.txt`: 해당 `.pkl`을 로드하는 데 필요한 정확한 라이브러리 버전
- `README.md`: 담당자, 모델명, 사용 피처, seed, 검증 점수

`.pkl`을 변경할 때는 반드시 같은 후보의 `script.py`와 `requirements.txt`도
확인합니다. 대용량 모델은 GitHub의 파일 크기 제한에 걸릴 수 있으므로, 100 MB가
넘는 파일은 커밋하기 전에 팀에 공유합니다.

## 피처 엔지니어링

피처 아이디어는 `features/`에 기능별 파일로 작성합니다. 예를 들어 날짜 피처는
`features/date_features.py`처럼 만듭니다.

피처가 특정 모델에 채택되면 해당 모델의 `train.py`와 `script.py`에 같은 로직을
반영합니다. 학습 때 사용한 컬럼 이름, 결측값 처리, 범주형 인코딩, 컬럼 순서는
추론 때도 반드시 같아야 합니다.

## 커밋

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 형식을
권장합니다. 커밋 메시지는 짧고 목적이 분명하게 작성합니다.

```text
feat: add lgbm baseline training
fix: align inference feature columns
docs: update submission procedure
chore: update model requirements
```

주요 타입은 `feat`, `fix`, `docs`, `refactor`, `test`, `chore`입니다. 여러 목적의
변경을 하나의 큰 커밋으로 묶지 않습니다.

## Pull Request

- 일반 PR의 대상 브랜치는 `develop`입니다.
- `main`으로의 PR은 제출 후보 검증이 끝난 경우에만 만듭니다.
- 가능하면 작성자 외 한 명 이상이 검토하고 승인합니다.
- PR 제목은 커밋 메시지 형식을 따릅니다.
- PR 본문에는 변경 내용, 모델 또는 피처 이름, seed, 검증 점수, 실행 확인 결과를 적습니다.

`main`에 병합하기 전에는 `submission/script.py`를 실제 입력 파일로 실행하고,
생성한 ZIP 안에 `model/`, `script.py`, `requirements.txt`가 최상위에 있는지 확인합니다.

## Python 및 문서 스타일

- Python 파일과 폴더 이름은 lowercase 또는 `snake_case`를 사용합니다.
- 클래스는 `PascalCase`, 함수와 변수는 `snake_case`, 상수는 `UPPER_CASE`를 사용합니다.
- import는 표준 라이브러리, 외부 라이브러리, 로컬 코드 순서로 작성합니다.
- 복잡한 전처리 함수에는 입력과 출력이 무엇인지 짧은 주석 또는 독스트링을 남깁니다.
- Markdown 문서는 제목과 목록을 사용해 읽기 쉽게 작성하고, 코드 변경 시 관련 문서도 갱신합니다.
