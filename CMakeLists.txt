cmake_minimum_required(VERSION 2.8)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -msse2")
endif()

ADD_DEFINITIONS(-DWITH_SSE)

# OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED xfeatures2d)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB srcs ./*.cpp ./*.h*)
add_executable(cpm ${srcs})
target_link_libraries(cpm ${OpenCV_LIBS})
