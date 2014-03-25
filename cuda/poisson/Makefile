
CC=nvcc
CUSP=-I/home/highscot/cusplibrary/

OBJS = jacobi.o utils.o
RUN_OBJS = run_jacobi_2d.o run_jacobi_3d.o run_cusp_3d.o tinker.o run_cusp_solver.o
TEST_OBJS = 

CPPFLAGS+=$(CUSP)

all : run_jacobi_2d run_jacobi_3d run_cusp_3d run_cusp_solver

# test: test_jacb
# 	./test_jacb

run_jacobi_2d : $(OBJS)
run_jacobi_3d : $(OBJS)
run_cusp_3d   : $(OBJS)
run_cusp_solver : $(OBJS)
tinker : $(OBJS)

$(OBJS) $(RUN_OBJS) $(TEST_OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<

# test_jacb: test_jacb.o utils.o
# 	$(CC) -o $@ $^ -lboost_unit_test_framework

clean:
	rm -vf *.o *.out
	rm -vf jacobi test_jacb run_jacobi_2d tinker
	rm -vf run_jacobi_3d run_cusp_3d run_cusp_solver

check-syntax:
	$(CC) -o nul -Wall $(CHK_SOURCES)

