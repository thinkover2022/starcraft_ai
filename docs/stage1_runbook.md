# 1단계 실행 안내 — 윈도우 한 대로 객체 인식 학습 데이터 생성부터 학습·평가까지

`docs/development_plan.md` 1단계를 실제로 돌리는 순서입니다.
**RTX 3090이 달린 윈도우 10/11 컴퓨터 한 대**에서 게임 캡처, 마스크 생성, 학습, 평가, 엔진 변환을 모두 합니다.
게임 캡처 → 마스크 생성 → 학습은 그래픽 카드를 함께 쓰므로 **동시에 돌리지 말고 순서대로** 실행합니다.

## 코드 위치

| 문서 1.4절 | 코드 |
|---|---|
| 리플레이 걸러내기 (1.5절) | `datagen/replay_filter/filter_replays.py` (screp 필요) |
| [가] 라벨 수집 모듈 | `tools/label_collector/` (C++, 32비트 DLL, 스타크래프트 1.16.1 + BWAPI 4.4.0) |
| [나] 화면 캡처 프로그램 | `tools/capture/capture_server.py` |
| [다] 마스크 생성 | `datagen/sam_masks/run.py` |
| 화면-상태 어긋남 실험 | `datagen/sam_masks/lag_check.py` |
| 사람 검수 이미지 | `datagen/review/overlay.py` |
| [라] 데이터셋 구성 | `datagen/build_dataset/build.py` |
| 소유자 판정 (1.8절) | `perception/detect/owner.py` |
| 학습 / 평가 / 엔진 변환 (1.7절) | `perception/detect/train.py`, `evaluate.py`, `export.py` |
| 공통 (클래스 41개, 프레임 기록 형식, 설정, 좌표, 이미지 입출력) | `datagen/common/` |
| 단계 실행 스크립트 | `tools/stage1.ps1` (PowerShell. 명령 프롬프트에서도 호출 가능) |
| 설정 | `configs/datagen.yaml`, `perception/detect/configs/` |

## PowerShell 과 명령 프롬프트(cmd)

모든 명령을 **PowerShell** 과 **명령 프롬프트(cmd.exe)** 두 가지로 적었습니다. 하나만 골라 쓰면 됩니다.
두 셸의 차이 중 이 문서에 나오는 것은 아래뿐입니다.

| 할 일 | PowerShell | 명령 프롬프트 |
|---|---|---|
| 가상환경 켜기 | `.\.venv\Scripts\Activate.ps1` | `.venv\Scripts\activate.bat` |
| 환경 변수 설정 (그 창에서만 유효) | `$env:PYTHONUTF8 = "1"` | `set PYTHONUTF8=1` (`=` 양옆에 공백 없이) |
| 긴 명령 줄 바꿈 | 줄 끝에 `` ` `` (백틱) | 줄 끝에 `^` |
| 파일 복사 | `Copy-Item 원본 대상` | `copy 원본 대상` |
| 파일 내용 보기 | `Get-Content 파일` | `type 파일` |
| 단계 실행 스크립트 | `.\tools\stage1.ps1 -Step masks ...` | `powershell -ExecutionPolicy Bypass -File tools\stage1.ps1 -Step masks ...` 또는 아래의 `python -m ...` 직접 실행 |

- 명령 프롬프트를 쓸 때는 창을 열 때마다 `set PYTHONUTF8=1` 을 먼저 실행합니다(한글 출력·파일 이름을 UTF-8로 처리).
  PowerShell 용 `tools\stage1.ps1` 은 이 설정을 스스로 합니다.
- 명령 프롬프트에서 한글이 깨져 보이면 `chcp 65001` 을 실행합니다.

## 0. 설치 (한 번)

### 준비물
- 엔비디아 그래픽 드라이버 최신판
- Python 3.11 (64비트) — 설치할 때 "Add python.exe to PATH" 선택
- Git
- Visual Studio 2022 Build Tools ("C++를 사용한 데스크톱 개발") + CMake — 라벨 수집 모듈 빌드용
- 스타크래프트 1.16.1, BWAPI 4.4.0, Chaoslauncher, 창 모드 실행 도구(창 모드 플러그인 또는 DirectDraw 대체 래퍼)
- screp 윈도우 실행 파일 (https://github.com/icza/screp 릴리스) — PATH에 추가

### Python 환경 (저장소 폴더에서)

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 1) PyTorch CUDA 판 — 반드시 먼저. 명령은 https://pytorch.org 의 설치 선택기(Windows / Pip / CUDA)에서 복사
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"     # True 여야 함
# 2) 이 저장소 + 학습·캡처 도구
pip install -e ".[dev,train,capture,sam]"
# 3) 범용 분할 모델 2 — 윈도우에서는 선택형 CUDA 확장을 빌드하지 않고 설치
$env:SAM2_BUILD_CUDA = "0"
pip install "git+https://github.com/facebookresearch/sam2.git"
# 4) 확인
python -m pytest
```

명령 프롬프트:
```bat
python -m venv .venv
.venv\Scripts\activate.bat
:: 1) PyTorch CUDA 판 — 반드시 먼저. 명령은 https://pytorch.org 의 설치 선택기(Windows / Pip / CUDA)에서 복사
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
:: 2) 이 저장소 + 학습·캡처 도구
pip install -e ".[dev,train,capture,sam]"
:: 3) 범용 분할 모델 2 — 윈도우에서는 선택형 CUDA 확장을 빌드하지 않고 설치
set SAM2_BUILD_CUDA=0
pip install "git+https://github.com/facebookresearch/sam2.git"
:: 4) 확인
python -m pytest
```
- `SAM2_BUILD_CUDA=0`: SAM 2의 CUDA 확장은 마스크의 작은 구멍을 메우는 후처리에만 쓰여, 없어도 사각형 힌트 분할은 같게 동작합니다.
  설치 중 경고가 나와도 무시해도 됩니다.
- SAM 2 설치가 끝내 실패하면 같은 PC에 WSL2(윈도우 안의 우분투)를 설치해 마스크 생성만 그 안에서 돌립니다.
  `D:\sc_data` 는 WSL2 안에서 `/mnt/d/sc_data` 로 보입니다.
- PyTorch가 `torch.cuda.is_available() == False` 이면 CPU 판이 깔린 것입니다. `pip uninstall torch torchvision` 후 1)을 다시 합니다.

### 폴더 배치 (예시)
```
D:\sc_data\
  raw\batch01\ ...      캡처 프로그램 출력 (PNG + JSON)
  ann\                  마스크 생성 출력 (주석 JSON)
  sc_terran_v1\         데이터셋 (학습 입력)
```
한글 폴더 이름도 동작하지만(이미지 입출력을 한글 경로에 안전하게 구현함), 외부 도구와의 호환을 위해 영문 경로를 권장합니다.

## 1. 리플레이 걸러내기

PowerShell, 명령 프롬프트 모두 같은 명령입니다.
```bat
python -m datagen.replay_filter.filter_replays --replays D:\replays --out D:\sc_data\tvt_list.txt
```
통과한 리플레이를 `C:\StarCraft\maps\replays\tvt\` 로 복사합니다. 출력된 맵 목록에서 평가 세트 ③용 맵 3종을 골라
`configs\datagen.yaml` 의 `split.held_out_maps` 에 적습니다.

## 2. 라벨 수집 모듈 빌드

PowerShell:
```powershell
cmake -S tools\label_collector -B build\label_collector -A Win32 -DBWAPI_DIR=C:\BWAPI
cmake --build build\label_collector --config Release
Copy-Item build\label_collector\Release\LabelCollector.dll C:\StarCraft\bwapi-data\AI\
```

명령 프롬프트 (시작 메뉴의 "x86 Native Tools Command Prompt for VS 2022" 를 쓰면 컴파일러 경로가 잡혀 있어 편합니다):
```bat
cmake -S tools\label_collector -B build\label_collector -A Win32 -DBWAPI_DIR=C:\BWAPI
cmake --build build\label_collector --config Release
copy build\label_collector\Release\LabelCollector.dll C:\StarCraft\bwapi-data\AI\
```
`C:\StarCraft\bwapi-data\bwapi.ini` 에서 `ai = bwapi-data\AI\LabelCollector.dll` 로 지정하고, 자동 메뉴 설정으로
리플레이 폴더를 차례로 재생하게 합니다(`auto_menu`, `map = maps\replays\tvt\*.rep` 등 — 설치한 BWAPI 버전의 bwapi.ini 주석 참고).
게임은 **창 모드 640x480, 확대 없음**으로 실행합니다.

## 3. 데이터 수집

PowerShell 또는 명령 프롬프트 창 하나를 열어 캡처 프로그램을 실행해 둡니다(두 셸 모두 같은 명령):
```bat
python tools\capture\capture_server.py --out D:\sc_data\raw\pilot --interval 12 --settle 2
```
그다음 Chaoslauncher 로 BWAPI 를 주입해 게임을 실행합니다.
- 캡처 프로그램을 **먼저** 실행해야 합니다. 모듈은 시작할 때 동기화 블록을 찾고, 없으면 아무것도 하지 않습니다
  (`C:\StarCraft\bwapi-data\logs\label_collector.log` 확인).
- 결과: `{replay_id}_{frame}.png` + `.json` 쌍, `_replay_index.jsonl`(리플레이 정보).
- 화면이 검게 찍히면 `--backend gdi` 로 바꿔 봅니다.
- 수집 중에는 게임 창을 가리지 말고, 마우스 커서를 게임 창 밖에 둡니다.
- **수집이 끝나면 게임을 종료한 뒤** 다음 단계(그래픽 카드 사용)를 시작합니다.

## 4. 시범 데이터로 측정할 것 (리플레이 20개)

PowerShell:
```powershell
.\tools\stage1.ps1 -Step masks  -Raw D:\sc_data\raw\pilot -Ann D:\sc_data\ann\pilot
.\tools\stage1.ps1 -Step lag    -Raw D:\sc_data\raw\pilot -Ann D:\sc_data\ann\pilot
.\tools\stage1.ps1 -Step review -Raw D:\sc_data\raw\pilot -Ann D:\sc_data\ann\pilot
```
(PowerShell 이 스크립트 실행을 막으면 한 번만 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`)

명령 프롬프트:
```bat
set PYTHONUTF8=1
python -m datagen.sam_masks.run --raw D:\sc_data\raw\pilot --out D:\sc_data\ann\pilot --config configs\datagen.yaml --skip-existing
python -m datagen.sam_masks.lag_check --raw D:\sc_data\raw\pilot --ann D:\sc_data\ann\pilot
python -m datagen.review.overlay --raw D:\sc_data\raw\pilot --ann D:\sc_data\ann\pilot --out D:\sc_data\ann\pilot\_review --fraction 0.05
```

1. **화면-상태 어긋남**: `lag` 결과의 `best_lag_frames` 를 이후 수집에서 `capture_server.py --lag` 로 넘깁니다(기록용).
   어긋남이 1 이상이면 라벨 수집 모듈이 `prev` 위치를 쓰도록 바꾸는 것을 검토합니다.
2. **조작판 영역**: 캡처 화면 한 장을 그림판 등으로 열어 조작판 부분을 검정(0), 게임 화면을 흰색(255)으로 칠한
   PNG 를 만들어 `mask.console_mask_png` 에 지정합니다.
3. **그림자 오프셋**: 레이스·수송선 화면에서 그림자 중심 − 유닛 중심을 재서 `mask.shadow_offset` 에 넣습니다.
4. **여유 폭**: `D:\sc_data\ann\pilot\_review` 의 검수 이미지에서 건물 윗부분·포신이 잘리면 `margins` / `type_margins` 를 늘립니다.
5. 불량률 5% 미만이 될 때까지 `masks`(기존 결과 폴더를 지우고) → `review` 를 반복합니다.

## 5. 대량 생산과 데이터셋 구성

PowerShell:
```powershell
.\tools\stage1.ps1 -Step masks -Raw D:\sc_data\raw -Ann D:\sc_data\ann
.\tools\stage1.ps1 -Step build -Raw D:\sc_data\raw -Ann D:\sc_data\ann -Dataset D:\sc_data\sc_terran_v1
Get-Content D:\sc_data\sc_terran_v1\report.json   # 세트별 화면 수, 클래스별 객체 수, 목표 미달 클래스
```

명령 프롬프트:
```bat
set PYTHONUTF8=1
python -m datagen.sam_masks.run --raw D:\sc_data\raw --out D:\sc_data\ann --config configs\datagen.yaml --skip-existing
python -m datagen.build_dataset.build --raw D:\sc_data\raw --ann D:\sc_data\ann ^
    --out D:\sc_data\sc_terran_v1 --config configs\datagen.yaml
type D:\sc_data\sc_terran_v1\report.json
```
정밀 검수 세트가 있으면 build 명령 끝에 `--gold D:\sc_data\gold_ann` 을 붙입니다.
- 데이터셋 이미지는 원본과 같은 드라이브면 하드 링크로 만들어 디스크를 아낍니다(다른 드라이브면 복사).
- 실제 대전 캡처(평가 세트 ④)는 프레임 기록의 `source` 가 `"live"` 이면 자동으로 `live` 세트로 갑니다.
  (라벨 수집 모듈의 실제 대전 모드는 아직 구현 전 — 리플레이 모드만 지원)
- 정밀 검수 세트 ⑤는 사람이 고친 주석 JSON 폴더를 `-Gold` 로 넘깁니다.

## 6. 소유자 색상표 보정과 평가

PowerShell:
```powershell
.\tools\stage1.ps1 -Step owner -Raw D:\sc_data\raw -Ann D:\sc_data\ann
```

명령 프롬프트:
```bat
set PYTHONUTF8=1
python -m perception.detect.owner calibrate --raw D:\sc_data\raw --ann D:\sc_data\ann --out configs\team_palette.json
python -m perception.detect.owner evaluate --raw D:\sc_data\raw --ann D:\sc_data\ann --palette configs\team_palette.json
```
결과 색상표는 `configs\team_palette.json` 에 저장됩니다.

## 7. 학습·평가·변환

PowerShell:
```powershell
.\tools\stage1.ps1 -Step train  -Dataset D:\sc_data\sc_terran_v1 -Experiment s640
.\tools\stage1.ps1 -Step eval   -Dataset D:\sc_data\sc_terran_v1 -Experiment s640
.\tools\stage1.ps1 -Step export -Experiment s640
```

명령 프롬프트:
```bat
set PYTHONUTF8=1
python -m perception.detect.train --experiment s640 --set data=D:\sc_data\sc_terran_v1\sc_terran.yaml
python -m perception.detect.evaluate --weights runs\terran\s640\weights\best.pt ^
    --dataset D:\sc_data\sc_terran_v1 --imgsz 640 --out runs\terran\s640\eval.json
python -m perception.detect.export --weights runs\terran\s640\weights\best.pt --imgsz 640 --format engine --bench
```
입력 크기 1280 실험(`n1280, s1280, m1280`)은 evaluate·export 의 `--imgsz` 를 1280 으로 바꿉니다
(PowerShell 스크립트는 실험 이름을 보고 자동으로 정함).
- 실험 이름: `n640, s640, m640, n1280, s1280, m1280` (`perception\detect\configs\experiments.yaml`).
- 학습 중 그래픽 메모리 부족이 나면 `--set batch=16` 처럼 배치를 줄입니다 (`python -m perception.detect.train --experiment s640 --set batch=16 data=...`).
- 윈도우에서 데이터 로더 오류(작업자 프로세스 관련)가 나면 `--set workers=0` 으로 확인합니다.
- 학습 곡선은 `runs\terran\{실험}\` 의 그림 파일과 `results.csv` 로 봅니다.
- TensorRT 엔진 변환에는 `pip install tensorrt` 가 필요할 수 있습니다(ultralytics 가 안내 메시지를 냄).
  안 되면 `--format onnx` 로 대신 변환합니다.

## 아직 확인되지 않은 것 (실제 환경에서 처음 돌릴 때 점검)

- 라벨 수집 모듈은 이 저장소에서 컴파일·실행해 보지 못했습니다. 첫 빌드에서 BWAPI 4.4.0 헤더와의
  함수 이름 차이가 있으면 고쳐야 합니다.
- 게임이 `onFrame` 안에서 멈춘 동안 화면에 보이는 것이 몇 프레임 전 상태인지 → 4-1 실험으로 확인.
- screp JSON 필드 이름 → `filter_replays.py` 상단 주석대로 한 번 대조.
- SAM 2를 CUDA 확장 없이 윈도우에 설치하는 방법은 SAM 2 저장소의 설치 안내가 바뀌었을 수 있으니 오류가 나면 그쪽을 먼저 확인.
- 안개 모드(`--vision-random`)에서는 안개 속 건물의 "마지막 모습"이 라벨 없이 남을 수 있어 해당 화면을 자동으로 뺍니다.
  처음에는 안개 없이(기본값) 수집합니다.
