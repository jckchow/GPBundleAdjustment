###
# Author: Jacky Chow
# -DCMAKE_BUILD_TYPE=Release
###

cmake_minimum_required(VERSION 2.8)
project(bundleAdjustment)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# targetProject name of project, should be same as name of folder where this cmakefile exists. 
function(createTarget targetProject)	
	file(GLOB _CPP "src/*.cpp")
	file(GLOB _H "include/*.h")

	source_group("Header Files" FILES ${_H})
	source_group("Source Files" FILES ${_CPP})

    add_executable(${targetProject} ${_H} ${_CPP})
    
endfunction()

# STEAM
createTarget(bundleAdjustment)
target_link_libraries(bundleAdjustment ${CERES_LIBRARIES} ${PYTHON_LIBRARIES})
