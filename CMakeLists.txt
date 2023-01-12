cmake_minimum_required(VERSION 3.15.0)

project(example)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}
                example/main.cpp
)

add_subdirectory(FANN)

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)

target_include_directories(${PROJECT_NAME} PUBLIC fann ${Python3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} fann ${Python3_LIBRARIES})