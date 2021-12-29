

PY_INC=`python3-config --includes`
NPY_INC=`python3 -c "import numpy as np; print(np.get_include())"`
CUDA_INC=/usr/local/cuda-11.2/include
CUDA_LIB=/usr/local/cuda-11.2/lib64/libcudart_static.a

all:
	swig -python -c++ buffer.i
	g++ buffer.cpp -c ${PY_INC} -I ${NPY_INC} -I ${CUDA_INC} -g3 -O0 -fPIC -std=c++11
	g++ buffer_wrap.cxx -c ${PY_INC} -I ${NPY_INC} -I ${CUDA_INC} -g3 -O0 -fPIC -std=c++11
	g++ -shared -lstdc++ buffer_wrap.o buffer.o ${CUDA_LIB} -lrt -o _buffer.so

clean:
	rm -f *.so *.o
	rm -f buffer.py buffer_wrap.cxx
	rm -rf __pycache__

