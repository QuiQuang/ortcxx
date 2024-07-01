cmake_minimum_required(VERSION 3.13)
project(cinnamon_runtime VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# Enable debug symbols
# set(CMAKE_BUILD_TYPE Debug)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Include all cmake files in /cmake directory.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntimeConfig.cmake)
# Find OnnxRuntime lib.
find_package(onnxruntime)
# Make executable
# add_executable(hello hello.cpp)
add_executable(modelTest modelTest.cpp
    ../cxx/core/model.cpp
    ../cxx/core/modelManager.cpp
    ../cxx/core/serviceManager.cpp)
add_executable(managerTest managerTest.cpp
    ../cxx/core/model.cpp
    ../cxx/core/modelManager.cpp
    ../cxx/core/serviceManager.cpp)
add_executable(managerServiceTest managerServiceTest.cpp
    ../cxx/core/model.cpp
    ../cxx/core/modelManager.cpp
    ../cxx/core/serviceManager.cpp)
# Link libraries.
# add include directories.

target_include_directories(modelTest PUBLIC ${onnxruntime_INCLUDE_DIRS} ../cxx/include)
target_include_directories(managerTest PUBLIC ${onnxruntime_INCLUDE_DIRS} ../cxx/include)
target_include_directories(managerServiceTest PUBLIC ${onnxruntime_INCLUDE_DIRS} ../cxx/include)
target_link_libraries(modelTest PRIVATE ${onnxruntime_LIBRARIES})
target_link_libraries(managerTest PRIVATE ${onnxruntime_LIBRARIES})
target_link_libraries(managerServiceTest PRIVATE ${onnxruntime_LIBRARIES} pthread)
# target_link_libraries(testModel ${onnxruntime_LIBRARY})
set_target_properties(modelTest PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(managerTest PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(managerServiceTest PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)