CXX = nvcc
CXXFLAGS = -std=c++11 -O3 #-fopenmp

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
