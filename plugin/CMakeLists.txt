cmake_minimum_required(VERSION 3.21)

project(
	Radiance
	VERSION 0.5.0
	LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# CommonLibVR (alandtse) - one DLL for Skyrim SE, AE and VR
set(COMMONLIB_DIR "" CACHE PATH "CommonLibVR checkout")
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${COMMONLIB_DIR}" CommonLibSSE EXCLUDE_FROM_ALL)
include("${COMMONLIB_DIR}/cmake/CommonLibSSE.cmake")

add_commonlibsse_plugin(
	Radiance
	NAME "Radiance"
	AUTHOR "Midnight"
	USE_ADDRESS_LIBRARY
	SOURCES src/main.cpp
)

target_precompile_headers(Radiance PRIVATE src/PCH.h)
find_package(ZLIB REQUIRED)
target_link_libraries(Radiance PRIVATE d3d11.lib ZLIB::ZLIB)
target_compile_definitions(Radiance PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(Radiance PRIVATE /W3 /permissive- /utf-8)
