%module buffer
%{
#define SWIG_FILE_WITH_INIT
#include "buffer.h"
#include "Python.h"
%}

%include "buffer.h"

%extend buffer
{
    %pythoncode
    {
    __array_struct__ = property(new_array_struct, None,
                                None, 'Numpy PyArrayInterface')
    }
}
