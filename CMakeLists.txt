CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(lib) #project name

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

######################################################
##################ADJUST##############################
######################################################
#what shall be build?
SET(MEXF TRUE) #generate mex file for matlab usage
SET(EXEF FALSE) #generate executable to call code from commandline
SET(CUDA TRUE) #use cuda for speedup if applicable

SET(OPENCV_INSTALLED TRUE)
SET(USE_EIGEN FALSE) #remember to clone the eigen repository into third_party dir
##################CPP & CUDA FILES####################
######################################################
#file(s) for binary version only
SET(CPP_MAIN_FILES src/main.cpp)

#file(s) for mex ( e.g. mex entrance function) version only
file(GLOB MEX_MAIN_FILES src/mex/*.cpp src/mex/*.h)

#files needed for both, cpp and cuda built
file(GLOB COMMON_SRC_FILES src/lib/*.cpp src/lib/*.h )

#cuda files (e.g. kernels)
file(GLOB CU_SRC_FILES src/lib/cuda/*.cu src/lib/cuda/*.cuh )

#cpp files needed for built with non-cuda support only
SET(CPP_SRC_FILES src/lib/cpuadd.cpp src/lib/cpuadd.h )

##################LIBRARIES###########################
######################################################

#used for cuda build only (nothing happens if ${CUDA}==FALSE)
SET(CUDA_LIBS CUDA_CUBLAS_LIBRARIES CUDA_CUFFT_LIBRARIES)

######################################################
##################SET EVERYTHING UP###################
######################################################
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) #output for executable i.e., EXEF=TRUE
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)    #output for mex-file i.e., MEXF=TRUE

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) #add modules e.g., FindMatlab.cmake

MESSAGE(STATUS "MEXF=" ${MEXF})
MESSAGE(STATUS "EXEF=" ${EXEF})
MESSAGE(STATUS "CUDA=" ${CUDA})


##################FIND OTHER LIBRARIES################
######################################################
if (${OPENCV_INSTALLED})
  ADD_DEFINITIONS(-DUSE_OPENCV)
  FIND_PACKAGE( OpenCV REQUIRED)
endif(${OPENCV_INSTALLED})
if (${USE_EIGEN})
  ADD_DEFINITIONS(-DUSE_EIGEN)
  SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/)
  FIND_PACKAGE(Eigen REQUIRED)
  INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})
endif(${USE_EIGEN})

##################FIND MATLAB#########################
######################################################
if(${MEXF})
  #not sure if the following two definitions are necessary
#  ADD_DEFINITIONS(/DMATLAB_MEX_FILE) #define matlab macros
#  ADD_DEFINITIONS(/DMX_COMPAT_32)

  FIND_PACKAGE(Matlab REQUIRED)

  #add definition USE_MEX for C++ code, to be able to use preprocessor if matlab is not used
  ADD_DEFINITIONS(-DUSE_MEX)

  # set up matlab libraries
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

  #use correct suffix depending on OS
  if(WIN32) # 32-bit or 64-bit mex
      if (CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw64 )
      else(CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw32 )
      endif(CMAKE_CL_64)
  else(WIN32)
      if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexa64 )
      else(CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexglx )
      endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
  endif(WIN32)
endif(${MEXF})

##################FIND CUDA###########################
######################################################
if(${CUDA})

  FIND_PACKAGE(CUDA REQUIRED)

  #add definition USE_MEX for C++ code, to be able to use preprocessor if CUDA is not used
  ADD_DEFINITIONS(-DUSE_CUDA)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_INCLUDE_DIRS}")

  # set SRC_FILES using the common cpp files and cu source files
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CU_SRC_FILES})

  #to correctly include the above cuda libraries, a splitting must take place
  foreach(libname ${CUDA_LIBS})
    LIST(APPEND CUDA_LIBS_SEP ${${libname}})
  endforeach(${CUDA_LIBS_SEP})

  #compiler flags for cuda
  SET(CUDA_NVCC_FLAGS
	-Xcompiler #Explicitly specify the language for the input files, rather than letting the compiler choose a default based on the file name suffix. Allowed values for this option: c, c++, cu. 
	-fPIC
	-use_fast_math
	#-gencode=arch=compute_50,code="sm_50"
	-arch=sm_35
	--ptxas-options=-v
	#--device-debug #possibility to debug device code; uncomment for speedup
	#--debug #possibility to debug host code; uncomment for speedup
	-O3 #specify optimization level for host code
	-Wno-deprecated-gpu-targets #to suppress warning for deprecated architectures
	#-D_FORCE_INLINES #uncommet if cuda<8.0 and some linker error of memcpy in string.h is thrown during build
	)

else(${CUDA})#if cuda is not used 
  #set SRC_FILES using the common cpp files and cpp source files
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CPP_SRC_FILES} )

endif(${CUDA})

######################################################
##################BUILD FILES#########################
######################################################

######################MEX#############################
######################################################
if(${MEXF})

  #if mex lib is build, add suffix MEX to lib file
  SET(MEX_LIB_NAME "${PROJECT_NAME}MEX")
  
  if(${CUDA})
    CUDA_COMPILE(${MEX_LIB_NAME} ${MEX_MAIN_FILES} ${SRC_FILES} SHARED)   
  endif(${CUDA})

  ADD_LIBRARY(${MEX_LIB_NAME} SHARED ${${MEX_LIB_NAME}} ${MEX_MAIN_FILES} ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)

  TARGET_LINK_LIBRARIES(${MEX_LIB_NAME}
                        ${MATLAB_LIBRARIES}
                        )

  if(${CUDA})
    MESSAGE(STATUS "LINKING CUDA LIBRARIES: " ${CUDA_LIBS_SEP})
    TARGET_LINK_LIBRARIES(${MEX_LIB_NAME}
                          ${CUDA_LIBRARIES} #Cudart library.
                          ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
                          )
  endif(${CUDA})

  SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES PREFIX "" SUFFIX ${MEX_SUFFIX})
  if(${CUDA})
    SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif(${CUDA})

endif(${MEXF})


##################EXECUTABLE##########################
######################################################
if(${EXEF}) #now take care of the executable

  if(${CUDA}) #if cuda is used
    CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES} )
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
                          ${CUDA_LIBRARIES} #Cudart library.
                          ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
                          )

  else(${CUDA})#is cuda is not used
    ADD_EXECUTABLE(${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES})
  endif(${CUDA})

  if (${OPENCV_INSTALLED})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS}) #load opencv libs (available only in ${PROJECT_NAME} target)
  endif(${OPENCV_INSTALLED})

endif(${EXEF})


CUDA_ADD_EXECUTABLE( run src/Run.cu src/helper.cu ${CU_SRC_FILES} )
TARGET_LINK_LIBRARIES(run
	${CUDA_LIBRARIES} #Cudart library.
	${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
	${OpenCV_LIBS}
)

