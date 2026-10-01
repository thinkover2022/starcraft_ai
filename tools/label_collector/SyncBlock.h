// 라벨 수집 모듈 ↔ 화면 캡처 프로그램 동기화 블록.
// tools/capture/sync_block.py 의 struct 형식과 반드시 같이 바꿀 것.
#pragma once
#include <cstdint>

namespace sclabel {

constexpr uint32_t kMagic = 0x424C4353;  // "SCLB"
constexpr uint32_t kVersion = 1;

constexpr const wchar_t* kMappingName = L"Local\\SCLabelSync";
constexpr const wchar_t* kRequestEvent = L"Local\\SCCaptureRequest";  // 모듈 → 캡처: 지금 찍어라
constexpr const wchar_t* kDoneEvent = L"Local\\SCCaptureDone";        // 캡처 → 모듈: 다 찍었다

enum CaptureStatus : int32_t { kPending = 0, kOk = 1, kFailed = 2 };

#pragma pack(push, 1)
struct SyncBlock {
  uint32_t magic;
  uint32_t version;
  // ---- 캡처 프로그램이 채우는 설정 (모듈은 읽기만) ----
  int32_t interval;            // 몇 프레임마다 화면 1장 (기본 12)
  int32_t settle_frames;       // 카메라를 옮긴 뒤 기다릴 프레임 (기본 2)
  int32_t capture_lag_frames;  // 실험으로 정한 화면-상태 어긋남 (기록용)
  int32_t vision_random;       // 1이면 리플레이 시야를 무작위로 바꿈
  uint32_t seed;
  int32_t timeout_ms;          // 캡처 완료를 기다리는 최대 시간
  // ---- 요청마다 바뀌는 값 ----
  int32_t request_frame;
  int32_t status;              // CaptureStatus, 캡처 프로그램이 씀
  char stem[128];              // 파일 이름 (확장자 제외), 모듈이 씀
  char out_dir[512];           // 저장 폴더 UTF-8, 캡처 프로그램이 씀
};
#pragma pack(pop)

static_assert(sizeof(SyncBlock) == 4 * 10 + 128 + 512, "SyncBlock 크기가 Python 쪽과 달라짐");

}  // namespace sclabel
