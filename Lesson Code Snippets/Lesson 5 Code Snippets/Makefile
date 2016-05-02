NVCC = nvcc
NVCC_OPTS = -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

TARGETS = transpose deviceQuery_simplified

all: $(TARGETS)

transpose: transpose.cu
	$(NVCC) $(NVCC_OPTS) -o $@ $<

deviceQuery_simplified: deviceQuery_simplified.cpp
	$(NVCC) $(NVCC_OPTS) -o $@ $<
clean:
	rm -f *.o $(TARGETS)
