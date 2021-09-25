cmake_minimum_required(VERSION 3.20)
project(InferDemo)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR "D:/opencv/build/x64/vc15/lib" )
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("D:/onnxruntime-win-x64-1.8.1/include")
link_directories("D:/onnxruntime-win-x64-1.8.1/lib")

add_executable(InferDemo main.cpp)
target_link_libraries(InferDemo ${OpenCV_LIBS})
target_link_libraries(InferDemo onnxruntime)


#set(Torch_DIR "D:/libtorch-win-shared-with-deps-1.9.0+cpu/libtorch/share/cmake/Torch")
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#include_directories("D:/libtorch-win-shared-with-deps-1.9.0+cpu/libtorch/include" "D:/libtorch-win-shared-with-deps-1.9.0+cpu/libtorch/include/torch/csrc/api/include")
#target_link_libraries(InferDemo "${TORCH_LIBRARIES}")


