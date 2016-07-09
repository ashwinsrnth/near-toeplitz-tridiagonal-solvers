bench-cusparse: main.o
	nvcc -o bench-cusparse -arch=sm_35 main.o -lcublas -lcusparse

main.o:	main.cpp
	nvcc -c -O2 main.cpp -arch=sm_35

clean:
	rm -f *.o bench-cusparse
