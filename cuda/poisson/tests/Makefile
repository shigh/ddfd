
CC  = nvcc
CUSP=-I/home/highscot/cusplibrary/

CPPFLAGS+=$(CUSP)
CPPFLAGS+=-I../
LDLIBS  += -lboost_unit_test_framework

OBJS = tests.o test_cusp_3d.o test_solver_cusp.o test_boundary_set.o \
       test_two_domain.o

.PHONY: all test clean

all: tests

test: tests
	./tests --log_level=test_suite

tests: $(OBJS)
	$(CC) $(LDLIBS) -o $@ $^

$(OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -vf nul *.o tests

# For emacs flymake mode
check-syntax:
	$(CC) $(TEST_INCLUDE) $(CXXFLAGS) -c -o nul -Wall $(CHK_SOURCES) -lboost_unit_test_framework

