#include "LabelCollector.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace BWAPI;

namespace {

constexpr int kScreenW = 640;
constexpr int kScreenH = 480;
constexpr int kViewCenterX = 320;  // 카메라 목표를 화면 가운데(조작판 위)에 두기 위한 값
constexpr int kViewCenterY = 200;
constexpr int kOnScreenSlack = 64;  // 화면 가장자리에 걸친 큰 건물도 기록

std::string jsonEscape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o += static_cast<char>(c);
        }
    }
  }
  return o;
}

uint32_t fnv1a(const std::string& s) {
  uint32_t h = 2166136261u;
  for (unsigned char c : s) { h ^= c; h *= 16777619u; }
  return h;
}

bool onScreen(Unit u, Position cam) {
  return u->getRight() - cam.x >= -kOnScreenSlack && u->getLeft() - cam.x < kScreenW + kOnScreenSlack &&
         u->getBottom() - cam.y >= -kOnScreenSlack && u->getTop() - cam.y < kScreenH + kOnScreenSlack;
}

std::vector<Player> gamePlayers() {
  std::vector<Player> out;
  for (Player p : Broodwar->getPlayers())
    if (!p->isNeutral() && !p->isObserver()) out.push_back(p);
  return out;
}

}  // namespace

void LabelCollector::log(const std::string& msg) const {
  std::ofstream f("bwapi-data/logs/label_collector.log", std::ios::app);
  f << "[" << Broodwar->getFrameCount() << "] " << msg << "\n";
}

bool LabelCollector::openSync() {
  mapping_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, sclabel::kMappingName);
  if (!mapping_) return false;
  sync_ = static_cast<sclabel::SyncBlock*>(
      MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sclabel::SyncBlock)));
  requestEvent_ = OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, sclabel::kRequestEvent);
  doneEvent_ = OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, sclabel::kDoneEvent);
  if (!sync_ || !requestEvent_ || !doneEvent_) return false;
  if (sync_->magic != sclabel::kMagic || sync_->version != sclabel::kVersion) {
    log("동기화 블록 버전 불일치");
    return false;
  }
  outDir_ = std::string(sync_->out_dir, strnlen(sync_->out_dir, sizeof sync_->out_dir));
  return true;
}

void LabelCollector::onStart() {
  if (!Broodwar->isReplay()) {
    // 실제 대전 캡처(평가 세트 ④)는 별도 모드로 확장 예정. 지금은 리플레이만 지원.
    Broodwar->sendText("label collector: replay only");
    return;
  }
  if (!openSync()) {
    log("캡처 프로그램을 찾을 수 없음 (tools/capture/capture_server.py 를 먼저 실행)");
    sync_ = nullptr;
    return;
  }
  // 이 함수 안에서 기다리는 동안 게임이 멈추므로 속도는 최대로 둔다.
  Broodwar->setLocalSpeed(0);
  rng_.seed(sync_->seed);

  // 리플레이 식별자: 맵 이름 + 플레이어 + 총 프레임 수의 해시 (같은 리플레이는 항상 같은 값)
  std::ostringstream key;
  key << Broodwar->mapName() << "|" << Broodwar->getReplayFrameCount();
  for (Player p : gamePlayers()) key << "|" << p->getName() << ":" << p->getRace().getName();
  char buf[16];
  std::snprintf(buf, sizeof buf, "r%08x", fnv1a(key.str()));
  replayId_ = buf;

  visionPlayers_ = gamePlayers();
  if (sync_->vision_random) randomizeVision();
  writeReplayIndex();
  log("시작 " + replayId_ + " " + Broodwar->mapName());
}

void LabelCollector::onEnd(bool) {
  if (sync_) {
    std::ostringstream m;
    m << "종료 " << replayId_ << " 캡처 " << captured_ << " 실패 " << failed_;
    log(m.str());
    UnmapViewOfFile(sync_);
  }
  if (mapping_) CloseHandle(mapping_);
  if (requestEvent_) CloseHandle(requestEvent_);
  if (doneEvent_) CloseHandle(doneEvent_);
  sync_ = nullptr;
}

void LabelCollector::randomizeVision() {
  // 50%: 모든 플레이어 시야(안개 없음), 25%씩: 한 플레이어 시야만(안개 있음)
  std::vector<Player> players = gamePlayers();
  std::uniform_real_distribution<double> u01(0, 1);
  double r = u01(rng_);
  visionPlayers_.clear();
  if (r < 0.5 || players.size() < 2) {
    visionPlayers_ = players;
  } else {
    visionPlayers_.push_back(players[r < 0.75 ? 0 : 1]);
  }
  for (Player p : players) {
    bool on = std::find(visionPlayers_.begin(), visionPlayers_.end(), p) != visionPlayers_.end();
    Broodwar->setVision(p, on);
  }
}

Position LabelCollector::pickCameraTarget() {
  std::uniform_real_distribution<double> u01(0, 1);
  double r = u01(rng_);
  std::vector<Unit> units;
  for (Unit u : Broodwar->getAllUnits())
    if (!u->getPlayer()->isNeutral()) units.push_back(u);

  if (!units.empty() && r < 0.4) {
    // 교전 지점: 무작위 유닛 중 300픽셀 안에 다른 플레이어 유닛이 있는 것
    std::uniform_int_distribution<size_t> pick(0, units.size() - 1);
    for (int tries = 0; tries < 50; ++tries) {
      Unit a = units[pick(rng_)];
      for (Unit b : units)
        if (b->getPlayer() != a->getPlayer() && a->getDistance(b) < 300) return a->getPosition();
    }
  }
  if (!units.empty() && r < 0.7) {
    // 기지: 무작위 건물
    std::vector<Unit> buildings;
    for (Unit u : units)
      if (u->getType().isBuilding()) buildings.push_back(u);
    if (!buildings.empty()) {
      std::uniform_int_distribution<size_t> pick(0, buildings.size() - 1);
      return buildings[pick(rng_)]->getPosition();
    }
  }
  std::uniform_int_distribution<int> rx(0, Broodwar->mapWidth() * 32 - 1);
  std::uniform_int_distribution<int> ry(0, Broodwar->mapHeight() * 32 - 1);
  return Position(rx(rng_), ry(rng_));
}

void LabelCollector::updateHistory() {
  std::unordered_map<int, std::array<Position, 3>> next;
  for (Unit u : Broodwar->getAllUnits()) {
    auto it = history_.find(u->getID());
    Position now = u->getPosition();
    next[u->getID()] = it == history_.end() ? std::array<Position, 3>{now, now, now}
                                            : std::array<Position, 3>{now, it->second[0], it->second[1]};
  }
  history_.swap(next);
}

void LabelCollector::onFrame() {
  if (!sync_) return;
  const int f = Broodwar->getFrameCount();
  updateHistory();

  if (sync_->vision_random && f > 0 && f % 2000 == 0) randomizeVision();

  if (pendingFrame_ < 0 && f % sync_->interval == 0) {
    Position t = pickCameraTarget();
    Broodwar->setScreenPosition(t.x - kViewCenterX, t.y - kViewCenterY);
    pendingFrame_ = f + sync_->settle_frames;  // 화면이 새 위치로 다시 그려질 때까지 기다림
    return;
  }
  if (f < pendingFrame_) return;
  pendingFrame_ = -1;

  char stem[64];
  std::snprintf(stem, sizeof stem, "%s_%06d", replayId_.c_str(), f);
  std::string json = buildFrameJson(f);
  if (!requestCapture(f, stem)) {
    ++failed_;
    return;
  }
  std::ofstream(outDir_ + "\\" + stem + ".json", std::ios::binary) << json;
  ++captured_;
}

bool LabelCollector::requestCapture(int frame, const std::string& stem) {
  sync_->request_frame = frame;
  sync_->status = sclabel::kPending;
  strncpy_s(sync_->stem, stem.c_str(), _TRUNCATE);
  ResetEvent(doneEvent_);
  SetEvent(requestEvent_);
  // 여기서 기다리는 동안 게임은 다음 프레임으로 넘어가지 않는다 → 정지된 화면이 캡처됨
  DWORD w = WaitForSingleObject(doneEvent_, sync_->timeout_ms);
  if (w != WAIT_OBJECT_0 || sync_->status != sclabel::kOk) {
    log("캡처 실패 " + stem);
    return false;
  }
  return true;
}

std::string LabelCollector::buildFrameJson(int frame) const {
  Position cam = Broodwar->getScreenPosition();
  std::ostringstream o;
  o << "{\"version\":1,\"replay_id\":\"" << replayId_ << "\",\"frame\":" << frame
    << ",\"map\":\"" << jsonEscape(Broodwar->mapName()) << "\",\"source\":\"replay\""
    << ",\"screen\":[" << kScreenW << "," << kScreenH << "]"
    << ",\"capture_lag_frames\":" << sync_->capture_lag_frames
    << ",\"camera\":[" << cam.x << "," << cam.y << "]";

  o << ",\"vision_players\":[";
  for (size_t i = 0; i < visionPlayers_.size(); ++i) o << (i ? "," : "") << visionPlayers_[i]->getID();
  o << "]";

  o << ",\"resources\":[";
  bool first = true;
  for (Player p : gamePlayers()) {
    o << (first ? "" : ",") << "{\"player\":" << p->getID() << ",\"minerals\":" << p->minerals()
      << ",\"gas\":" << p->gas() << ",\"supply_used\":" << p->supplyUsed()
      << ",\"supply_total\":" << p->supplyTotal() << "}";
    first = false;
  }
  o << "]";

  int fogged = 0;
  o << ",\"units\":[";
  first = true;
  for (Unit u : Broodwar->getAllUnits()) {
    if (!onScreen(u, cam)) continue;
    bool visible = false;
    for (Player p : visionPlayers_) visible = visible || u->isVisible(p);
    if (!visible) {
      // 안개 속 건물은 '마지막으로 본 모습'이 화면에 남을 수 있다 → 그런 화면은 학습에서 뺀다
      if (u->getType().isBuilding() && !u->getPlayer()->isNeutral()) ++fogged;
      continue;
    }
    Position pos = u->getPosition();
    o << (first ? "" : ",") << "{\"id\":" << u->getID() << ",\"type\":\"" << u->getType().getName()
      << "\",\"owner\":" << u->getPlayer()->getID() << ",\"color\":" << u->getPlayer()->getColor().getID()
      << ",\"x\":" << pos.x << ",\"y\":" << pos.y << ",\"box\":[" << u->getLeft() << "," << u->getTop()
      << "," << u->getRight() << "," << u->getBottom() << "]";
    auto h = history_.find(u->getID());
    if (h != history_.end())
      o << ",\"prev\":[[" << h->second[1].x << "," << h->second[1].y << "],[" << h->second[2].x << ","
        << h->second[2].y << "]]";
    o << ",\"hp\":" << u->getHitPoints() << ",\"selected\":" << (u->isSelected() ? "true" : "false")
      << ",\"flying\":" << (u->isFlying() ? "true" : "false")
      << ",\"lifted\":" << (u->isLifted() ? "true" : "false")
      << ",\"constructing\":" << (u->isBeingConstructed() ? "true" : "false")
      << ",\"cloaked\":" << (u->isCloaked() ? "true" : "false")
      << ",\"burrowed\":" << (u->isBurrowed() ? "true" : "false") << "}";
    first = false;
  }
  o << "],\"fogged_buildings\":" << fogged << "}";
  return o.str();
}

void LabelCollector::writeReplayIndex() const {
  std::ofstream f(outDir_ + "\\_replay_index.jsonl", std::ios::app | std::ios::binary);
  f << "{\"replay_id\":\"" << replayId_ << "\",\"map\":\"" << jsonEscape(Broodwar->mapName())
    << "\",\"map_file\":\"" << jsonEscape(Broodwar->mapFileName())
    << "\",\"frames\":" << Broodwar->getReplayFrameCount() << ",\"players\":[";
  bool first = true;
  for (Player p : gamePlayers()) {
    f << (first ? "" : ",") << "{\"id\":" << p->getID() << ",\"name\":\"" << jsonEscape(p->getName())
      << "\",\"race\":\"" << p->getRace().getName() << "\",\"color\":" << p->getColor().getID() << "}";
    first = false;
  }
  f << "]}\n";
}
