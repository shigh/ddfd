
CC=nvcc
CUSP=/home/highscot/cusplibrary/
OBJS = jacobi.o utils.o
RUN_OBJS = run_jacobi_2d.o run_jacobi_3d.o
TEST_OBJS = 

all : run_jacobi_2d run_jacobi_3d

# test: test_jacb
# 	./test_jacb

run_jacobi_2d : $(OBJS)
run_jacobi_3d : $(OBJS)

cusp_poisson : cusp_poisson.o
cusp_poisson.o : cusp_poisson.cu
	$(CC) $(CPPFLAGS) -I$(CUSP) -c -o $@ $<

$(OBJS) $(RUN_OBJS) $(TEST_OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<

# test_jacb: test_jacb.o utils.o
# 	$(CC) -o $@ $^ -lboost_unit_test_framework

clean:
	rm -vf *.o *.out
	rm -vf jacobi test_jacb run_jacobi_2d run_jacobi_3d

check-syntax:
	$(CC) -o nul -Wall $(CHK_SOURCES)

