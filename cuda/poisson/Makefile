
CC=nvcc
OBJS = jacobi.o utils.o run_jacobi.o
TEST_OBJS = 

all : run_jacobi

# test: test_jacb
# 	./test_jacb

run_jacobi: $(OBJS)

$(OBJS) $(TEST_OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<

# test_jacb: test_jacb.o utils.o
# 	$(CC) -o $@ $^ -lboost_unit_test_framework

clean:
	rm -vf *.o *.out
	rm -vf jacobi test_jacb run_jacobi

check-syntax:
	$(CC) -o nul -Wall $(CHK_SOURCES)

