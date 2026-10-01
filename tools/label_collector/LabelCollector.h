#pragma once
#include <BWAPI.h>
#include <windows.h>

#include <array>
#include <random>
#include <unordered_map>
#include <string>
#include <vector>

#include "SyncBlock.h"

// 라벨 수집 모듈 (문서 1.4절 [가]).
// 리플레이를 재생하며 interval 프레임마다 카메라를 옮기고, settle 프레임 뒤에
// 화면 안 유닛 목록을 기록한 다음 캡처 프로그램이 화면을 찍을 때까지 게임을 멈춘다.
class LabelCollector : public BWAPI::AIModule {
 public:
  void onStart() override;
  void onEnd(bool isWinner) override;
  void onFrame() override;

 private:
  bool openSync();
  BWAPI::Position pickCameraTarget();
  void randomizeVision();
  std::string buildFrameJson(int frame) const;
  bool requestCapture(int frame, const std::string& stem);
  void writeReplayIndex() const;
  void log(const std::string& msg) const;
  void updateHistory();

  HANDLE mapping_ = nullptr;
  HANDLE requestEvent_ = nullptr;
  HANDLE doneEvent_ = nullptr;
  sclabel::SyncBlock* sync_ = nullptr;

  std::string replayId_;
  std::string outDir_;
  int pendingFrame_ = -1;
  int captured_ = 0;
  int failed_ = 0;
  std::vector<BWAPI::Player> visionPlayers_;
  std::mt19937 rng_;
  // 유닛별 최근 3프레임 위치 [지금, 1프레임 전, 2프레임 전] — 화면-상태 어긋남 실험용
  std::unordered_map<int, std::array<BWAPI::Position, 3>> history_;
};
