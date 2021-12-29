#ifndef double_buffer_h
#define double_buffer_h

#include "Python.h"

/* a class for tetsing sharing GPU or CPU data with numpy via the array
 * interface protocol.
 */
class buffer
{
public:
    buffer() = delete;
    buffer(const buffer &) = delete;
    void operator=(buffer &) = delete;

    // allocate a buffer of size n and initialize it to val.
    buffer(size_t n, double val, int cuda);

    // free the memory associated with the buffer
    ~buffer();

    /* get a PyArrayInterface struct packaged in a PyCapsule to share data with
     * numpy.
     */
    PyObject *new_array_struct();


private:
    double *m_data;
    size_t m_size;
    int m_cuda;
};

#endif
