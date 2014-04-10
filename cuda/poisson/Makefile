
CC=nvcc
CUSP=-I/home/highscot/cusplibrary/

OBJS = jacobi.o utils.o
TEST_OBJS = 

CPPFLAGS+=$(CUSP)

all : $(OBJS)

tinker : $(OBJS)

docs :
	doxygen Doxyfile

$(OBJS) $(RUN_OBJS) $(TEST_OBJS) : %.o : %.cu
	$(CC) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -vf *.o *.out
	rm -rf html

# For emacs flymake mode
check-syntax:
	$(CC) -o nul -Wall $(CHK_SOURCES)

