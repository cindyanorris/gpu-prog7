NVCC = /usr/local/cuda-8.0/bin/nvcc
CC = g++
GENCODE_FLAGS = -arch=sm_30

#Optimization flags. Don't use this for debugging.
NVCCFLAGS = -c -m64 -O2 --compiler-options -Wall -Xptxas -O2,-v

#No optimizations. Debugging flags. Use this for debugging.
#NVCCFLAGS = -c -g -G -m64 --compiler-options -Wall

OBJS = wrappers.o convolute.o h_convolute.o d_convolute.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

convolute: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o convolute

convolute.o: convolute.cu h_convolute.h d_convolute.h config.h

h_convolute.o: h_convolute.cu h_convolute.h CHECK.h

d_convolute.o: d_convolute.cu d_convolute.h CHECK.h config.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm convolute *.o
