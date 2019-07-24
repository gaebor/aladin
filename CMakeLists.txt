cmake_minimum_required (VERSION 3.1)
project (aladin)

include_directories("$ENV{OPENBLAS_INC}" "${PROJECT_SOURCE_DIR}/inc")
link_directories("$ENV{OPENBLAS_LIB}")

set (CMAKE_CXX_STANDARD 11)

if(UNIX)
    add_compile_options("-Wall")
elseif(WIN32)
    add_compile_options("/W4")
endif()

file(GLOB ALADIN_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB ALADIN_INC ${PROJECT_SOURCE_DIR}/inc/*.h)

add_executable(aladin ${ALADIN_SRC} ${ALADIN_INC})

target_link_libraries(aladin openblas)

if (UNIX)
    target_link_libraries(aladin pthread)
endif()

include(CTest)

add_test(testhelp aladin -h)
