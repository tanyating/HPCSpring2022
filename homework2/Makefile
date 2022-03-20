CC := g++
CXXFLAGS := -march=native -O3
OUTPUTS := $(basename $(%.c)) $(basename $(%.cpp))

all: $(OUTPUTS)

%val_test%: %val_test%.cpp
	$(CC) $(CXXFLAGS) $< -o $@

%omp_solved%: %omp_solved%.c
	$(CC) $(CXXFLAGS) $< -fopenmp -o $@

MMult1: MMult1.cpp
	$(CC) $(CXXFLAGS) -std=c++11 -fopenmp -o $@ $^

gs2D-omp: gs2D-omp.cpp
	$(CC) $(CXXFLAGS) -std=c++11 -fopenmp -o $@ $^

jacobi2D-omp: jacobi2D-omp.cpp
	$(CC) $(CXXFLAGS) -std=c++11 -fopenmp -o $@ $^



clean:
	rm -f $(OUTPUTS)
