
CC  = nvcc
CUSP=-I/home/highscot/cusplibrary/

CPPFLAGS+=$(CUSP)
CPPFLAGS+=-I../
LDLIBS  += -lboost_unit_test_framework

OBJS = tests.o test_cusp_3d.o

all: tests

tests: $(OBJS)
	$(CC) $(LDLIBS) -o $@ $^

$(OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<


clean:
	rm -vf nul *.o tests

# For emacs flymake mode
check-syntax:
	$(CC) $(TEST_INCLUDE) $(CXXFLAGS) -c -o nul -Wall $(CHK_SOURCES) -lboost_unit_test_framework

