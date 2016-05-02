NVCC = nvcc
NVCC_OPTS = -O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

TARGETS = atomics associative

all: $(TARGETS)

atomics: atomics.cu
	$(NVCC) $(NVCC_OPTS) atomics.cu -o $@

associative: associative.cu
	$(NVCC) $(NVCC_OPTS) associative.cu -o $@
	
memory: memory.cu
	$(NVCC) $(NVCC_OPTS) memory.cu -o $@

clean:
	rm -f *.o $(TARGETS)
