if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(tokenizer.axera)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(src)

add_library(tokenizer src/BaseTokenizer.cpp src/tokenizer/tokenizer_optimized.cpp)
set_target_properties(tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

function(add_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} tokenizer)
endfunction()

# option to build tests
option(TOKENIZER_BUILD_TESTS "Build tests" ON)
MESSAGE(STATUS "TOKENIZER_BUILD_TESTS = ${TOKENIZER_BUILD_TESTS}")

# Optional jinja chat templating (minja). Pulls in third_party/minja (header-only,
# uses <regex> on the small template string only). Off-by-default builds stay lean
# for edge targets that keep the hand-written templates.
option(TOKENIZER_WITH_JINJA "Enable minja-based jinja chat templates" ON)
MESSAGE(STATUS "TOKENIZER_WITH_JINJA = ${TOKENIZER_WITH_JINJA}")
if(TOKENIZER_WITH_JINJA)
    include_directories(third_party)
endif()

# Optional general HF-directory loader: encode/decode straight from a HuggingFace
# tokenizer.json via wangzhaode/tokenizer.cpp (+ oniguruma/utf8proc), paired with
# minja for chat templates. No per-model code for new models. Heavier deps, so it
# is off by default -- edge builds keep the lean .txt + hand-written path.
option(TOKENIZER_WITH_HF_LOADER "Enable direct HuggingFace tokenizer-dir loading" OFF)
MESSAGE(STATUS "TOKENIZER_WITH_HF_LOADER = ${TOKENIZER_WITH_HF_LOADER}")
if(TOKENIZER_WITH_HF_LOADER)
    if(NOT TOKENIZER_WITH_JINJA)
        message(FATAL_ERROR "TOKENIZER_WITH_HF_LOADER requires TOKENIZER_WITH_JINJA=ON")
    endif()
    add_subdirectory(third_party/tokenizer.cpp EXCLUDE_FROM_ALL)
endif()

if(TOKENIZER_BUILD_TESTS)
    add_test(test_tokenizer)
    add_test(test_num2words_en)
    add_test(test_string_create)
    add_test(test_hf_parity)
    if(TOKENIZER_WITH_JINJA)
        add_test(test_jinja_parity)
    endif()
    if(TOKENIZER_WITH_HF_LOADER)
        add_executable(test_hf_loader tests/test_hf_loader.cpp)
        target_link_libraries(test_hf_loader tokenizer tokenizer_lib)
    endif()
endif()


install(TARGETS tokenizer DESTINATION lib)
install(FILES include/BaseTokenizer.hpp DESTINATION include)