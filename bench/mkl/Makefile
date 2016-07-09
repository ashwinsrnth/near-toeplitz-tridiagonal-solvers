bench-mkl: main.o
	icc -o bench-mkl main.o -mkl=parallel -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -O2

main.o:	main.cpp
	icc -c main.cpp -mkl=parallel -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -O2

clean:
	rm -f *.o bench-mkl
