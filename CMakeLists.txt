if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(tokenizer.axera)

include_directories(include)

add_library(tokenizer src/base_tokenizer.cpp src/tokenizer/tokenizer.cpp)

function(add_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} tokenizer)
endfunction()

add_test(test_tokenizer)


install(TARGETS tokenizer DESTINATION lib)
install(FILES include/base_tokenizer.hpp DESTINATION include)