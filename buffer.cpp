#include "buffer.h"

#include <iostream>
#include <stddef.h>

#include <cuda_runtime.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// --------------------------------------------------------------------------
buffer::buffer(size_t n, double val, int cuda) : m_size(n), m_data(nullptr)
{
    std::cerr << "buffer::buffer " << this << std::endl
        << "allocated a " << (cuda ? "CUDA" : "CPU")
        << " buffer of size  " << n << " initialized to " << val
        << std::endl;

    size_t n_bytes = n*sizeof(double);

    // intialize an array with val
    double *tmp = (double*)malloc(n*sizeof(double));
    for (size_t i = 0; i < n; ++i)
        tmp[i] = val;

    if (cuda)
    {
        // send to the GPU
        cudaMalloc(&m_data, n_bytes);
        cudaMemcpy(m_data, tmp, n_bytes, cudaMemcpyHostToDevice);
        free(tmp);
        m_cuda = 1;
    }
    else
    {
        // keep it on the cpu
        m_data = tmp;
        m_cuda = 0;
    }
}

// --------------------------------------------------------------------------
buffer::~buffer()
{
    std::cerr << "buffer::~buffer " << this << std::endl;

    if (m_cuda)
    {
        cudaFree(m_data);
    }
    else
    {
        free(m_data);
    }
}


void delete_array_interface(PyObject *cap)
{
    double *ptr = (double*)PyCapsule_GetContext(cap);
    free(ptr);

    PyArrayInterface *nai = (PyArrayInterface*)
        PyCapsule_GetPointer(cap, nullptr);

    free(nai->shape);
    free(nai);

    std::cerr << "delete_array_interface <---------------- here!" << std::endl
        << "cap = " << cap << " nai = "  << nai << " ptr = " << (size_t)ptr
        << std::endl;
}

// --------------------------------------------------------------------------
PyObject *buffer::new_array_struct()
{
    size_t n_bytes = m_size*sizeof(double);

    // numpy always needs the data on the CPU.  if the data is on the GPU
    // allocate a temporary, move the data from the gpu, and point to the
    // tmeporary. the PyCapsule destructor will free it
    double *ptr = nullptr;
    if (m_cuda)
    {
        ptr = (double*)malloc(n_bytes);
        cudaMemcpy(ptr, m_data, n_bytes, cudaMemcpyDeviceToHost);
    }

    // calculate the shape and stride
    int nd = 1;

    npy_intp *ss = (npy_intp*)malloc(2*nd*sizeof(npy_intp));
    npy_intp *shape = ss;
    npy_intp *stride = ss + nd;

    shape[0] = m_size;
    stride[0] = sizeof(double);

    // construct the array interface
    PyArrayInterface *nai = (PyArrayInterface*)
        malloc(sizeof(PyArrayInterface));

    memset(nai, 0, sizeof(PyArrayInterface));

    nai->two = 2;
    nai->nd = nd;
    nai->typekind = 'f';
    nai->itemsize = sizeof(double);
    nai->shape = shape;
    nai->strides = stride;
    nai->data = m_cuda ? ptr : m_data;
    nai->flags = (m_cuda ? 0x0 : NPY_ARRAY_WRITEABLE) |
        NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;

    // package into a capsule
    PyObject *cap = PyCapsule_New(nai, nullptr, delete_array_interface);

    // save the pointer to the data
    PyCapsule_SetContext(cap, ptr);

    std::cerr << "buffer::new_array_struct" << std::endl
        << "cap = " << cap << " nai = "  << nai << " ptr = " << (size_t)ptr
        << std::endl;

    return cap;
}
