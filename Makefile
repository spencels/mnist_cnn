CXX=g++
NVCC=nvcc
CXXFLAGS=-std=c++11 -fPIC -O2 -fmax-errors=1 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -lcudart
CUDAFLAGS=-std=c++11 -c -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC=`python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
INCLUDES=-I $(TF_INC) -I "$(CUDA_HOME)include"

BINDIR=bin

all: $(BINDIR)/image_vector_distort.so

test: image_vector_distort_test

clean:
	rm -rf $(BINDIR)

train: $(BINDIR)/image_vector_distort.so
	python mnist_basic.py --train

quicktrain: $(BINDIR)/image_vector_distort.so
	python mnist_basic.py --train --columns=1 --epochs=30

$(BINDIR)/:
	mkdir $(BINDIR)

$(BINDIR)/image_vector_distort.cu.o: image_vector_distort.cu.cc image_vector_distort.h
	mkdir -p $(BINDIR)
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) -o $(BINDIR)/image_vector_distort.cu.o \
	image_vector_distort.cu.cc

$(BINDIR)/image_vector_distort.so: image_vector_distort.cc image_vector_distort.h $(BINDIR)/image_vector_distort.cu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) image_vector_distort.cc \
		$(BINDIR)/image_vector_distort.cu.o -o $(BINDIR)/image_vector_distort.so \
		-L $(CUDA_HOME)lib64

# cuda
# so

image_vector_distort_test: $(BINDIR)/image_vector_distort.so
	python image_vector_distort_test.py