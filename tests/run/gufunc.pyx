# mode: run
# tag: numpy

cimport cython
cimport numpy as cnp

import numpy as np

@cython.gufunc("(),()->(3)")
cdef void multidim_out(double a1, double a2, double* out):
    out[0] = a1 + a2
    out[1] = a1 * a2
    out[2] = a1 + a2 + 1

def test_multidim_out():
    """
    >>> a1 = np.array([1., 2., 3., 4.])
    >>> a2 = np.array([4., 5., 6., 7.])
    >>> out = multidim_out(a1, a2)
    >>> out.shape
    (4, 3)
    >>> out
    array([[ 5.,  4.,  6.],
           [ 7., 10.,  8.],
           [ 9., 18., 10.],
           [11., 28., 12.]])
    """

@cython.gufunc("(3),()->()")
cdef void reduction(double* a1, double a2, double* out):
    out[0] = a1[0] +a1[1] + a1[2] + a2

def test_reduction():
    """
    >>> a1 = np.array([1., 2., 3.])
    >>> a2 = np.array([1., 2., 3., 4.])
    >>> out = reduction(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([ 7.,  8.,  9., 10.])
    """

@cython.gufunc("(3),()->(3)")
cdef void mimo(double* a1, double a2, double* out):
    out[0] = (a1[0] + a1[1]) * a2
    out[1] = (a1[1] + a1[2]) * a2
    out[2] = (a1[0] + a1[2]) * a2

def test_mimo():
    """
    >>> a1 = np.array([1., 2., 3.])
    >>> a2 = np.array([1., 2.])
    >>> out = mimo(a1, a2)
    >>> out.shape
    (2, 3)
    >>> out
    array([[ 3.,  5.,  4.],
           [ 6., 10.,  8.]])
    """

@cython.gufunc("(3),(),()->()")
cdef void long_signature(double* a1, double a2, double a3, double* out):
    out[0] = a1[0] +a1[1] + a1[2] + a2 + a3

def test_long_signature():
    """
    >>> a1 = np.array([1., 2., 3.])
    >>> a2 = np.array([1., 2., 3., 4.])
    >>> a3 = np.array([1., 2., 3., 4.])
    >>> out = long_signature(a1, a2, a3)
    >>> out.shape
    (4,)
    >>> out
    array([ 8., 10., 12., 14.])
    """

@cython.gufunc("(3),(),()->()")
cdef void args_float(double* a1, double a2, float a3, double* out):
    out[0] = a1[0] +a1[1] + a1[2] + a2 + a3

def test_args_float():
    """
    >>> a1 = np.array([1., 2., 3.])
    >>> a2 = np.array([1., 2., 3., 4.])
    >>> a3 = np.array([1., 2., 3., 4.], dtype=np.float32)
    >>> out = args_float(a1, a2, a3)
    >>> out.shape
    (4,)
    >>> out
    array([ 8., 10., 12., 14.])
    """

@cython.gufunc("(),()->()")
cdef void args_int(int a1, int a2, int* out):
    out[0] = a1 + a2

def test_args_int():
    """
    >>> a1 = np.array([1, 2, 3, 4], dtype=np.int32)
    >>> a2 = np.array([1, 2, 3, 4], dtype=np.int32)
    >>> out = args_int(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([2, 4, 6, 8], dtype=int32)
    """

ctypedef fused integral:
    short
    int
    long
    long long

@cython.gufunc("(),()->()")
cdef void args_int_fused(integral a1, integral a2, integral* out):
    out[0] = a1 + a2

def test_args_int_fused():
    """
    Test with int32
    >>> a1 = np.array([1, 2, 3, 4], dtype=np.int32)
    >>> a2 = np.array([1, 2, 3, 4], dtype=np.int32)
    >>> out = args_int_fused(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([2, 4, 6, 8], dtype=int32)
    
    Test with int64
    >>> a1 = np.array([1, 2, 3, 4], dtype=np.int64)
    >>> a2 = np.array([1, 2, 3, 4], dtype=np.int64)
    >>> out = args_int_fused(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([2, 4, 6, 8])
    
    Test with int16
    >>> a1 = np.array([1, 2, 3, 4], dtype=np.int16)
    >>> a2 = np.array([1, 2, 3, 4], dtype=np.int16)
    >>> out = args_int_fused(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([2, 4, 6, 8], dtype=int16)
    """

from libc.math cimport atan2, hypot
@cython.gufunc("(3)->(3)")
cdef void car2cyl(double *input, double *output):
    """Convert cartesian to cylindrical coordinate for gufunc"""
    cdef double phi = atan2(input[1], input[0])
    output[0] = hypot(input[0], input[1])
    output[1] = phi
    output[2] = input[2]

def test_car2cyl():
    """
    >>> car = np.array([[3, -4, 1], [0,0,0]])
    >>> cyl = car2cyl(car)
    >>> cyl.shape
    (2, 3)
    >>> cyl
    array([[ 5.        , -0.92729522,  1.        ],
           [ 0.        ,  0.        ,  0.        ]])
    """

@cython.gufunc("(),()->()")
cdef void args_complex(int a1, double complex a2, double complex* out):
    out[0] = a1 + a2

def test_args_number_t():
    """
    >>> a1 = np.array([1, 2, 3, 4], dtype=np.int32)
    >>> a2 = np.array([1+1j, 2+1j, 3+1j, 4+2j])
    >>> out = args_complex(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([2.+1.j, 4.+1.j, 6.+1.j, 8.+2.j])
    """