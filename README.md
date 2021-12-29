This is a code illustrating a memory management issue with numpy array interface c-struct approach. See mumpy issue #XXXX.

The code contains a simple class that shares an array with Numpy via the array interface c-struct method. The class can share an array allocated on the GPU or the CPU. 

Compiling:

1. edit the Make file. you'll need paths to Python, Numpy, and CUDA headers.
2. make


Testing:

valgrind python3 test_buffer.py



