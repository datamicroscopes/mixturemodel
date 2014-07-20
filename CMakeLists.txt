cmake_minimum_required(VERSION 2.6)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(microscopes_mixturemodel)

set(CMAKE_CXX_FLAGS_BASE "${CMAKE_CXX_FLAGS} -fPIC -g -MD -Wall -std=c++0x")
# taken from distributions
set(CMAKE_CXX_FLAGS_MATHOPT "-mfpmath=sse -msse4.1 -ffast-math -funsafe-math-optimizations")

if(APPLE)
  set(CMAKE_CXX_FLAGS_BASE "${CMAKE_CXX_FLAGS_BASE} -stdlib=libc++")
  # for anaconda builds
  set(CMAKE_CXX_FLAGS_BASE "${CMAKE_CXX_FLAGS_BASE} -mmacosx-version-min=10.7")
  # clang complains about register
  set(CMAKE_CXX_FLAGS_BASE "${CMAKE_CXX_FLAGS_BASE} -Wno-deprecated-register")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BASE} -O3 -DNDEBUG ${CMAKE_CXX_FLAGS_MATHOPT}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_BASE} -O3 -DNDEBUG ${CMAKE_CXX_FLAGS_MATHOPT}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_BASE} -DDEBUG_MODE")

# give our include dirs the most precedent
include_directories(include)

# followed by the EXTRA_* ones
if(DEFINED EXTRA_INCLUDE_PATH)
  include_directories(${EXTRA_INCLUDE_PATH})
endif()
if(DEFINED EXTRA_LIBRARY_PATH)
  link_directories(${EXTRA_LIBRARY_PATH})
endif()

find_package(Protobuf REQUIRED)
message(STATUS "found protobuf INC=${PROTOBUF_INCLUDE_DIRS}, LIB=${PROTOBUF_LIBRARIES}")
include_directories(${PROTOBUF_INCLUDE_DIRS})

find_package(Distributions)
if(DISTRIBUTIONS_FOUND)
  message(STATUS "found distributions INC=${DISTRIBUTIONS_INCLUDE_DIRS}, LIB=${DISTRIBUTIONS_LIBRARY_DIRS}")
  include_directories(${DISTRIBUTIONS_INCLUDE_DIRS})
  link_directories(${DISTRIBUTIONS_LIBRARY_DIRS})
else()
  message(FATAL_ERROR "Could not find distributions")
endif()

find_package(MicroscopesCommon)
if(MICROSCOPES_COMMON_FOUND)
  message(STATUS "found microscopes_common INC=${MICROSCOPES_COMMON_INCLUDE_DIRS}, LIB=${MICROSCOPES_COMMON_LIBRARY_DIRS}")
  include_directories(${MICROSCOPES_COMMON_INCLUDE_DIRS})
  link_directories(${MICROSCOPES_COMMON_LIBRARY_DIRS})
else()
  message(FATAL_ERROR "Could not find microscopes_common")
endif()

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY microscopes DESTINATION cython FILES_MATCHING PATTERN "*.pxd" PATTERN "__init__.py")

set(MICROSCOPES_MIXTUREMODEL_SOURCE_FILES src/mixture/model.cpp)
add_library(microscopes_mixturemodel SHARED ${MICROSCOPES_MIXTUREMODEL_SOURCE_FILES})
target_link_libraries(microscopes_mixturemodel ${PROTOBUF_LIBRARIES} distributions_shared microscopes_common)
install(TARGETS microscopes_mixturemodel LIBRARY DESTINATION lib)

# test executables
enable_testing()
add_executable(test_state test/cxx/test_state.cpp)
add_test(test_state test_state)
target_link_libraries(test_state ${PROTOBUF_LIBRARIES} distributions_shared microscopes_common microscopes_mixturemodel)
