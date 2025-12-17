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
cdef void args_complex(long a1, double complex a2, double complex* out):
    out[0] = a1 + a2

def test_args_number_t():
    """
    >>> a1 = np.array([1, 2, 3, 4])
    >>> a2 = np.array([1+1j, 2+1j, 3+1j, 4+2j])
    >>> out = args_complex(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([2.+1.j, 4.+1.j, 6.+1.j, 8.+2.j])
    """

ctypedef fused number_t:
    double
    double complex

@cython.gufunc("(),()->()")
cdef void args_number_t(number_t a1, number_t a2, number_t* out):
    out[0] = a1 + a2*a1

def test_args_number_t():
    """
    Test with double
    >>> a1 = np.array([1., 2., 3., 4.])
    >>> a2 = np.array([1., 2., 3., 4.])
    >>> out = args_number_t(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([ 2.,  6., 12., 20.])

    Test with double complex
    >>> a1 = np.array([1.+1j, 2.+2j, 3.+3j, 4.+4j])
    >>> a2 = np.array([1.+0j, 0.+1j, 1.+1j, 0.+0j])
    >>> out = args_number_t(a1, a2)
    >>> out.shape
    (4,)
    >>> out
    array([2.+2.j, 0.+4.j, 3.+9.j, 4.+4.j])
    """

# Test multiple outputs
@cython.gufunc("(),()->(),()")
cdef void two_outputs(double a, double b, double* out1, double* out2):
    out1[0] = a + b
    out2[0] = a * b

def test_two_outputs():
    """
    >>> a = np.array([1., 2., 3., 4.])
    >>> b = np.array([4., 5., 6., 7.])
    >>> out1, out2 = two_outputs(a, b)
    >>> out1
    array([ 5.,  7.,  9., 11.])
    >>> out2
    array([ 4., 10., 18., 28.])
    """

@cython.gufunc("(2)->(),()")
cdef void array_to_two_outputs(double* arr, double* sum_out, double* prod_out):
    sum_out[0] = arr[0] + arr[1]
    prod_out[0] = arr[0] * arr[1]

def test_array_to_two_outputs():
    """
    >>> arr = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> sum_out, prod_out = array_to_two_outputs(arr)
    >>> sum_out
    array([ 3.,  7., 11.])
    >>> prod_out
    array([ 2., 12., 30.])
    """

@cython.gufunc("()->(2)")
cdef void scalar_to_array_pair(double x, double* out):
    out[0] = x
    out[1] = x * x

def test_scalar_to_array_pair():
    """
    >>> x = np.array([1., 2., 3., 4.])
    >>> out = scalar_to_array_pair(x)
    >>> out
    array([[ 1.,  1.],
           [ 2.,  4.],
           [ 3.,  9.],
           [ 4., 16.]])
    """

# Test multidimensional batching
@cython.gufunc("(),()->()")
cdef void simple_add(double a, double b, double* out):
    out[0] = a + b

def test_multidim_batching():
    """
    Test 1D batching
    >>> a = np.array([1., 2., 3., 4.])
    >>> b = np.array([10., 20., 30., 40.])
    >>> result = simple_add(a, b)
    >>> result
    array([11., 22., 33., 44.])
    
    Test 2D batching
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> b = np.array([[10., 20.], [30., 40.]])
    >>> result = simple_add(a, b)
    >>> result
    array([[11., 22.],
           [33., 44.]])
    
    Test 3D batching
    >>> a = np.ones((2, 3, 4))
    >>> b = np.ones((2, 3, 4)) * 2
    >>> result = simple_add(a, b)
    >>> result.shape
    (2, 3, 4)
    >>> np.float64(result[0, 0, 0])
    np.float64(3.0)
    """

@cython.gufunc("(3)->()")
cdef void sum_3(double* arr, double* out):
    out[0] = arr[0] + arr[1] + arr[2]

def test_multidim_batching_with_core():
    """
    Test 2D array with core dimension
    >>> arr = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> result = sum_3(arr)
    >>> result
    array([ 6., 15.])
    
    Test 3D array with core dimension (batching over first dimension)
    >>> arr = np.array([[[1., 2., 3.], [4., 5., 6.]], 
    ...                 [[7., 8., 9.], [10., 11., 12.]]])
    >>> result = sum_3(arr)
    >>> result
    array([[ 6., 15.],
           [24., 33.]])
    """

@cython.gufunc("(2),(2)->(2)")
cdef void add_2d_vectors(double* a, double* b, double* out):
    out[0] = a[0] + b[0]
    out[1] = a[1] + b[1]

def test_batching_with_multiple_inputs():
    """
    Test broadcasting: (3, 2) and (2,) -> (3, 2)
    >>> a = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> b = np.array([10., 20.])
    >>> result = add_2d_vectors(a, b)
    >>> result
    array([[11., 22.],
           [13., 24.],
           [15., 26.]])
    """

