CC := g++
CFLAGS := -march=native
CPPFLAGS := -march=native -O3
OUTPUTS := val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 gs2D-omp jacobi2D-omp

all: $(OUTPUTS)

val_test01_solved: val_test01_solved.cpp
	$(CC) -o $@ $^

val_test02_solved: val_test02_solved.cpp
	$(CC) -o $@ $^

omp_solved2: omp_solved2.c
	$(CC) -fopenmp -o $@ $^

omp_solved3: omp_solved3.c
	$(CC) -fopenmp -o $@ $^

omp_solved4: omp_solved4.c
	$(CC) -fopenmp -o $@ $^

omp_solved5: omp_solved5.c
	$(CC) -fopenmp -o $@ $^

omp_solved6: omp_solved6.c
	$(CC) -fopenmp -o $@ $^

MMult1: MMult1.cpp
	$(CC) $(CPPFLAGS) -std=c++11 -fopenmp -o $@ $^

gs2D-omp: gs2D-omp.cpp
	$(CC) $(CPPFLAGS) -std=c++11 -fopenmp -o $@ $^

jacobi2D-omp: jacobi2D-omp.cpp
	$(CC) $(CPPFLAGS) -std=c++11 -fopenmp -o $@ $^



clean:
	rm -f $(OUTPUTS)
