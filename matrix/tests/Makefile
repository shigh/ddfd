
CXX = g++
CC  = $(CXX)
CXXFLAGS += -I../ -g -std=c++0x -fopenmp
LDLIBS  += -lboost_unit_test_framework
LDFLAGS += -fopenmp -std=c++0x 

OBJS = tests.o test_offsets.o

all: tests

test: tests 
	./tests

tests: $(OBJS)

clean:
	rm -vf nul *.o tests 

# For emacs flymake mode
check-syntax:
	$(CC) $(TEST_INCLUDE) $(CXXFLAGS) -c -o nul -Wall $(CHK_SOURCES) -lboost_unit_test_framework

