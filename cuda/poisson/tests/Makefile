
CC  = nvcc
CUSP=-I/home/highscot/cusplibrary/

#MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile)
MPI_COMPILE_FLAGS = -I/opt/software/OpenMPI/1.4.3--GCC-4.4.5/include
#MPI_LINK_FLAGS = $(shell mpicc --showme:link)
MPI_LINK_FLAGS = -L/usr/lib64 -L/opt/software/OpenMPI/1.4.3--GCC-4.4.5/lib \
                 -lmpi_cxx -lmpi -lopen-rte -lopen-pal -lrdmacm -libverbs \
                 -ltorque -lnsl -lutil -lm -lcr -lnuma -ldl \
                 -lnsl -lutil -lm -ldl

CPPFLAGS+=$(CUSP)
CPPFLAGS+=-I../
CPPFLAGS+=$(MPI_COMPILE_FLAGS)
LDLIBS  += -lboost_unit_test_framework
LDLIBS  +=$(MPI_LINK_FLAGS)

OBJS = tests.o test_cusp_3d.o test_solver_cusp.o test_boundary_set.o \
       test_two_domain.o test_grid.o

MPI_OBJS = mpi_mwe.o

.PHONY: all test mwe clean

all: tests

test: tests
	./tests --log_level=test_suite

mpi_mwe: mpi_mwe.o

tests: $(OBJS)
	$(CC) $(LDLIBS) -o $@ $^

$(OBJS) $(MPI_OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -vf nul *.o tests mwe

# For emacs flymake mode
check-syntax:
	$(CC) $(TEST_INCLUDE) $(CXXFLAGS) -c -o nul -Wall $(CHK_SOURCES) -lboost_unit_test_framework

