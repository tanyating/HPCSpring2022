CXX = nvcc
CXXFLAGS = -Xcompiler -fopenmp 

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
