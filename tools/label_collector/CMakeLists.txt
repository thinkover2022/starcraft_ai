# 라벨 수집 모듈 빌드 (윈도우, 32비트 필수)
#   cmake -S tools/label_collector -B build/label_collector -A Win32 -DBWAPI_DIR=C:/BWAPI
#   cmake --build build/label_collector --config Release
# 결과: LabelCollector.dll → StarCraft/bwapi-data/AI/ 에 복사하고 bwapi.ini 의 ai 항목에 지정
cmake_minimum_required(VERSION 3.20)
project(LabelCollector CXX)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
  message(FATAL_ERROR "게임 정보 인터페이스 라이브러리 모듈은 32비트로 빌드해야 함: -A Win32")
endif()

set(BWAPI_DIR "" CACHE PATH "BWAPI 4.4.0 설치 폴더 (include/, lib/ 포함)")
set(CMAKE_CXX_STANDARD 17)

add_library(LabelCollector SHARED Dll.cpp LabelCollector.cpp)
target_include_directories(LabelCollector PRIVATE ${BWAPI_DIR}/include)
target_link_libraries(LabelCollector PRIVATE
  $<$<CONFIG:Debug>:${BWAPI_DIR}/lib/BWAPId.lib>
  $<$<NOT:$<CONFIG:Debug>>:${BWAPI_DIR}/lib/BWAPI.lib>)
target_compile_definitions(LabelCollector PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
target_compile_options(LabelCollector PRIVATE /utf-8 /W3)
