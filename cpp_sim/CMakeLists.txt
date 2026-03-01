cmake_minimum_required(VERSION 3.10)
project(MyUnderwaterSimulator)

# 1. 启用 C++17 (Stonefish 需要)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 查找 Stonefish 库 (这是最关键的一步)
# find_package 会去系统的 /usr/local/lib/cmake 等路径下找 StonefishConfig.cmake
# 找到后，它会提供 ${Stonefish_INCLUDE_DIRS} 和 Stonefish::Stonefish 目标
find_package(Stonefish REQUIRED)

# 3. 查找 OpenGL 等依赖 (Stonefish 依赖它们，显式查找更保险)
find_package(OpenGL REQUIRED)
# 通常 Stonefish 会自动处理它的依赖，但如果报错，可能需要手动 find_package(Eigen3) 等

# 3.1 显式查找 Freetype (解决 ft2build.h 找不到的问题)
find_package(Freetype REQUIRED)

# 4. 包含头文件路径
# 让编译器知道 <stonefish/core/Robot.h> 在哪里
include_directories(${Stonefish_INCLUDE_DIRS})
# 让编译器知道你自己的 .h 文件在哪里 (即 src 目录)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# 4.1 将 Freetype 的头文件路径告诉编译器
# 这会将类似 /usr/include/freetype2 的路径加进去
include_directories(${FREETYPE_INCLUDE_DIRS})

# 5. 定义资源路径宏 (解决 DATA_DIR_PATH 问题)， DATA_DIR_PATH 是一个预编译宏。
# 我们在这里把它指向你项目里的 resources 文件夹。
add_definitions(-DDATA_DIR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/")

# 6. 定义可执行文件
add_executable(MySim
    src/main.cpp
    src/MyAUVApp.cpp
    src/MyAUVApp.h
    src/MyAUVManager.cpp
    src/MyAUVManager.h
)

# 7. 链接库
# 将你的程序链接到找到的 Stonefish 库
target_link_libraries(MySim PUBLIC Stonefish::Stonefish)