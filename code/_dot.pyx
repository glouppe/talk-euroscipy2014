# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython cimport view
import numpy as np
cimport numpy as np
np.import_array()


def dot_python(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


cpdef dot_mv(double[::1] a, double[::1] b):
    cdef double s = 0
    cdef int i
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return s


cpdef dot_ptr(np.ndarray[double, ndim=1] a, np.ndarray[double, ndim=1] b):
    cdef double* a_ptr = <double*> a.data
    cdef double* b_ptr = <double*> b.data
    cdef double s = 0
    cdef int i
    for i in range(a.shape[0]):
        s += a_ptr[i] * b_ptr[i]
    return s
