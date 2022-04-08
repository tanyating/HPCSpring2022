CXX = nvcc
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean