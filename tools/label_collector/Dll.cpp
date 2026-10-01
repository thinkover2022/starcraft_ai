// 게임 정보 인터페이스 라이브러리(BWAPI) 4.4.0 AI 모듈 진입점 (ExampleAIModule 과 같은 형식)
#include <BWAPI.h>
#include <windows.h>

#include "LabelCollector.h"

extern "C" __declspec(dllexport) void gameInit(BWAPI::Game* game) { BWAPI::BroodwarPtr = game; }

BOOL APIENTRY DllMain(HANDLE, DWORD, LPVOID) { return TRUE; }

extern "C" __declspec(dllexport) BWAPI::AIModule* newAIModule() { return new LabelCollector(); }
