cmake_minimum_required(VERSION 3.22.1)

project(example)


add_executable(${PROJECT_NAME} 
                example/main.cpp
)

add_subdirectory(ANN)

target_include_directories(${PROJECT_NAME} PUBLIC fann)
target_link_directories(${PROJECT_NAME} PRIVATE fann)
target_link_libraries(${PROJECT_NAME} ann python3.10)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(PythonLibs 3.10 REQUIRED)
include_directories(/usr/include/python3.10)