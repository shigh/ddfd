
include ../Make.inc

CXX = g++
CXXFLAGS  = -g -std=c++0x -fopenmp -fPIC $(LAPACKINC)
OBJ = 

ifeq ($(LAPACK_TYPE), LAPACK_MKL)
BACKEND_OBJ = linalg/backend_mkl.cpp
endif

ifeq ($(LAPACK_TYPE), LAPACK_LAPACKE)
BACKEND_OBJ = linalg/backend_lapack.cpp
endif

all:

backend.o: $(BACKEND_OBJ)
	mpic++ $(CPPFLAGS) -g -fopenmp -fPIC $(LAPACKINC) -c -o $@ $<

linear_solve.o: linalg/linear_solve.cpp
	mpic++ $(CPPFLAGS) -g -fopenmp -fPIC $(LAPACKINC) -c -o $@ $<

linalg_utils.o: linalg/linalg_utils.cpp
	mpic++ $(CPPFLAGS) -g -fopenmp -fPIC $(LAPACKINC) -c -o $@ $<

clean:
	rm -vf nul *.o *.a

# For emacs flymake mode
check-syntax:
	$(CC) $(TEST_INCLUDE) $(CXXFLAGS) $(LAPACKINC) -c -o nul -Wall $(CHK_SOURCES)


