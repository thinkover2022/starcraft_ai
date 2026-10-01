# 1단계 실행 안내 — 객체 인식 학습 데이터 생성부터 학습·평가까지

`docs/development_plan.md` 1단계를 실제로 돌리는 순서입니다.
윈도우 컴퓨터(게임·캡처)와 우분투 컴퓨터(마스크 생성·학습)를 오갑니다.

## 코드 위치

| 문서 1.4절 | 코드 | 실행 환경 |
|---|---|---|
| 리플레이 걸러내기 (1.5절) | `datagen/replay_filter/filter_replays.py` | 어디서나 (screp 필요) |
| [가] 라벨 수집 모듈 | `tools/label_collector/` (C++, 32비트 DLL) | 윈도우 + 스타크래프트 1.16.1 + BWAPI 4.4.0 |
| [나] 화면 캡처 프로그램 | `tools/capture/capture_server.py` | 윈도우 |
| [다] 마스크 생성 | `datagen/sam_masks/run.py` | 우분투 + RTX 3090 |
| 화면-상태 어긋남 실험 | `datagen/sam_masks/lag_check.py` | 어디서나 |
| 사람 검수 이미지 | `datagen/review/overlay.py` | 어디서나 |
| [라] 데이터셋 구성 | `datagen/build_dataset/build.py` | 어디서나 |
| 소유자 판정 (1.8절) | `perception/detect/owner.py` | 어디서나 |
| 학습 / 평가 / 엔진 변환 (1.7절) | `perception/detect/train.py`, `evaluate.py`, `export.py` | 우분투 / 우분투 / 윈도우 |
| 공통 (클래스 41개, 프레임 기록 형식, 설정, 좌표) | `datagen/common/` | |
| 설정 | `configs/datagen.yaml`, `perception/detect/configs/` | |

## 0. 설치

```bash
pip install -e ".[dev]"              # 공통
pip install -e ".[sam]"              # 우분투: 범용 분할 모델 (torch + sam2)
pip install -e ".[train]"            # 우분투: 객체 인식 모델 학습 (ultralytics)
pip install -e ".[capture]"          # 윈도우: dxcam
python -m pytest                     # 그래픽 카드 없이 도는 단위 시험
```

## 1. 리플레이 걸러내기

```bash
python -m datagen.replay_filter.filter_replays --replays D:/replays --out data/tvt_list.txt
```
통과한 리플레이를 `StarCraft/maps/replays/tvt/` 로 복사합니다. 출력된 맵 목록에서 평가 세트 ③용 맵 3종을 골라
`configs/datagen.yaml` 의 `split.held_out_maps` 에 적습니다.

## 2. 라벨 수집 모듈 빌드 (윈도우)

```bat
cmake -S tools/label_collector -B build/label_collector -A Win32 -DBWAPI_DIR=C:/BWAPI
cmake --build build/label_collector --config Release
copy build\label_collector\Release\LabelCollector.dll C:\StarCraft\bwapi-data\AI\
```
`bwapi-data/bwapi.ini` 에서 `ai = bwapi-data/AI/LabelCollector.dll` 로 지정하고, 자동 메뉴 설정으로
리플레이 폴더를 차례로 재생하게 합니다(`auto_menu`, `map = maps\replays\tvt\*.rep` 등 — 설치한 BWAPI 버전의 bwapi.ini 주석 참고).
게임은 **창 모드 640x480, 확대 없음**으로 실행합니다.

## 3. 데이터 수집 (윈도우)

```bat
python tools/capture/capture_server.py --out D:\sc_raw\pilot --interval 12 --settle 2
:: 다른 창에서 Chaoslauncher 로 BWAPI 주입 + 게임 실행
```
- 캡처 프로그램을 **먼저** 실행해야 합니다. 모듈은 시작할 때 동기화 블록을 찾고, 없으면 아무것도 하지 않습니다
  (`bwapi-data/logs/label_collector.log` 확인).
- 결과: `{replay_id}_{frame}.png` + `.json` 쌍, `_replay_index.jsonl`(리플레이 정보).
- 화면이 검게 찍히면 `--backend gdi` 로 바꿔 봅니다.

## 4. 시범 데이터로 측정할 것 (리플레이 20개)

1. 마스크 생성: `python -m datagen.sam_masks.run --raw D:/sc_raw/pilot --out data/pilot_ann --config configs/datagen.yaml`
2. **화면-상태 어긋남**: `python -m datagen.sam_masks.lag_check --raw D:/sc_raw/pilot --ann data/pilot_ann`
   → `best_lag_frames` 를 이후 수집에서 `capture_server.py --lag` 로 넘깁니다(기록용).
   어긋남이 1 이상이면 `--settle` 은 그대로 두고, 라벨 수집 모듈이 `prev` 위치를 쓰도록 바꾸는 것을 검토합니다.
3. **조작판 영역**: 캡처 화면 한 장에서 조작판 부분을 0, 게임 화면을 255로 칠한 PNG를 만들어
   `mask.console_mask_png` 에 지정합니다.
4. **그림자 오프셋**: 레이스·수송선 화면에서 그림자 중심 − 유닛 중심을 재서 `mask.shadow_offset` 에 넣습니다.
5. **여유 폭**: 검수 이미지(아래)에서 건물 윗부분·포신이 잘리면 `margins` / `type_margins` 를 늘립니다.
6. 검수 이미지: `python -m datagen.review.overlay --raw D:/sc_raw/pilot --ann data/pilot_ann --out data/review --fraction 0.05`
   — 불량률 5% 미만이 될 때까지 3–5를 반복합니다.

## 5. 대량 생산과 데이터셋 구성

```bash
python -m datagen.sam_masks.run --raw /data/sc_raw --out /data/sc_ann --config configs/datagen.yaml --skip-existing
python -m datagen.build_dataset.build --raw /data/sc_raw --ann /data/sc_ann --out /data/sc_terran_v1 \
    --config configs/datagen.yaml [--gold /data/sc_gold_ann]
cat /data/sc_terran_v1/report.json     # 세트별 화면 수, 클래스별 객체 수, 목표 미달 클래스
```
- 실제 대전 캡처(평가 세트 ④)는 프레임 기록의 `source` 가 `"live"` 이면 자동으로 `live` 세트로 갑니다.
  (라벨 수집 모듈의 실제 대전 모드는 아직 구현 전 — 리플레이 모드만 지원)
- 정밀 검수 세트 ⑤는 사람이 고친 주석 JSON 폴더를 `--gold` 로 넘깁니다.

## 6. 소유자 색상표 보정과 평가

```bash
python -m perception.detect.owner calibrate --raw /data/sc_raw --ann /data/sc_ann --out configs/team_palette.json
python -m perception.detect.owner evaluate  --raw /data/sc_raw --ann /data/sc_ann --palette configs/team_palette.json
```

## 7. 학습·평가·변환

```bash
python -m perception.detect.train --experiment s640 --set data=/data/sc_terran_v1/sc_terran.yaml
python -m perception.detect.evaluate --weights runs/terran/s640/weights/best.pt \
    --dataset /data/sc_terran_v1 --out runs/terran/s640/eval.json
# 윈도우 실행 컴퓨터에서
python -m perception.detect.export --weights best.pt --format engine --bench
```

## 아직 확인되지 않은 것 (실제 환경에서 처음 돌릴 때 점검)

- 라벨 수집 모듈은 이 저장소에서 컴파일·실행해 보지 못했습니다(리눅스 환경). 첫 빌드에서 BWAPI 4.4.0 헤더와의
  함수 이름 차이가 있으면 고쳐야 합니다.
- 게임이 `onFrame` 안에서 멈춘 동안 화면에 보이는 것이 몇 프레임 전 상태인지 → 4-2 실험으로 확인.
- screp JSON 필드 이름 → `filter_replays.py` 상단 주석대로 한 번 대조.
- 안개 모드(`--vision-random`)에서는 안개 속 건물의 "마지막 모습"이 라벨 없이 남을 수 있어 해당 화면을 자동으로 뺍니다.
  처음에는 안개 없이(기본값) 수집합니다.
