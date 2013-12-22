
CC=nvcc
OBJS = jacobi_2d.o utils.o
TEST_OBJS = test_jacb.o

all : jacobi_2d test_jacb

test: test_jacb
	./test_jacb

$(OBJS) $(TEST_OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<

jacobi_2d: $(OBJS)
	$(CC) -o $@ $^

test_jacb: test_jacb.o utils.o
	$(CC) -o $@ $^ -lboost_unit_test_framework

clean:
	rm -vf *.o *.out
	rm -vf jacobi_2d test_jacb

check-syntax:
	$(CC) -o nul -Wall $(CHK_SOURCES)

