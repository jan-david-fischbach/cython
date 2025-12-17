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

# Tests for named dimensions (variable size dimensions)

@cython.gufunc("(i),(i)->()")
cdef void inner1d(double* a, double* b, double* out, cnp.npy_intp i):
    """Inner product of two 1D arrays with variable size"""
    cdef cnp.npy_intp j
    out[0] = 0
    for j in range(i):
        out[0] += a[j] * b[j]

def test_inner1d():
    """
    Test inner product with 1D arrays
    >>> a = np.array([1., 2., 3.])
    >>> b = np.array([4., 5., 6.])
    >>> result = inner1d(a, b)
    >>> float(result)
    32.0
    
    Test with 2D batch
    >>> a = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> b = np.array([[7., 8., 9.], [10., 11., 12.]])
    >>> result = inner1d(a, b)
    >>> result
    array([ 50., 167.])
    
    Test with different size
    >>> a = np.array([1., 2., 3., 4., 5.])
    >>> b = np.array([5., 4., 3., 2., 1.])
    >>> result = inner1d(a, b)
    >>> float(result)
    35.0
    """

@cython.gufunc("(m,n),(n,p)->(m,p)")
cdef void matmat(double* a, double* b, double* out, cnp.npy_intp m, cnp.npy_intp n, cnp.npy_intp p):
    """Matrix-matrix multiplication"""
    cdef cnp.npy_intp ii, jj, kk
    # Initialize output
    for ii in range(m):
        for jj in range(p):
            out[ii * p + jj] = 0
    # Compute matrix multiplication
    for ii in range(m):
        for jj in range(p):
            for kk in range(n):
                out[ii * p + jj] += a[ii * n + kk] * b[kk * p + jj]

def test_matmat():
    """
    Test 2x3 @ 3x2
    >>> a = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> b = np.array([[7., 8.], [9., 10.], [11., 12.]])
    >>> result = matmat(a, b)
    >>> result
    array([[ 58.,  64.],
           [139., 154.]])
    
    Test with batching: (2,2,3) @ (2,3,2)
    >>> a = np.array([[[1., 2., 3.], [4., 5., 6.]], [[1., 1., 1.], [2., 2., 2.]]])
    >>> b = np.array([[[7., 8.], [9., 10.], [11., 12.]], [[1., 0.], [0., 1.], [1., 0.]]])
    >>> result = matmat(a, b)
    >>> result.shape
    (2, 2, 2)
    >>> result[0]
    array([[ 58.,  64.],
           [139., 154.]])
    >>> result[1]
    array([[2., 1.],
           [4., 2.]])
    """

@cython.gufunc("(n),(n,p)->(p)")
cdef void vecmat(double* v, double* m, double* out, cnp.npy_intp n, cnp.npy_intp p):
    """Vector-matrix multiplication: v @ m"""
    cdef cnp.npy_intp i, j
    for i in range(p):
        out[i] = 0
        for j in range(n):
            out[i] += v[j] * m[j * p + i]

def test_vecmat():
    """
    Test vector @ matrix
    >>> v = np.array([1., 2., 3.])
    >>> m = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> result = vecmat(v, m)
    >>> result
    array([22., 28.])
    """

@cython.gufunc("(m,n),(n)->(m)")
cdef void matvec(double* mat, double* v, double* out, cnp.npy_intp m, cnp.npy_intp n):
    """Matrix-vector multiplication: mat @ v"""
    cdef cnp.npy_intp i, j
    for i in range(m):
        out[i] = 0
        for j in range(n):
            out[i] += mat[i * n + j] * v[j]

def test_matvec():
    """
    Test matrix @ vector
    >>> m = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> v = np.array([7., 8., 9.])
    >>> result = matvec(m, v)
    >>> result
    array([ 50., 122.])
    """

@cython.gufunc("(i,t),(j,t)->(i,j)")
cdef void outer_inner(double* a, double* b, double* out, cnp.npy_intp i, cnp.npy_intp t, cnp.npy_intp j):
    """Outer product of inner products
    
    Note: Dimension parameters must be in order of first appearance in signature.
    For "(i,t),(j,t)->(i,j)", scanning left-to-right gives: i, t, j (j appears after t).
    This matches NumPy's dimension ordering for gufuncs.
    """
    cdef cnp.npy_intp ii, jj, k
    for ii in range(i):
        for jj in range(j):
            out[ii * j + jj] = 0
            for k in range(t):
                out[ii * j + jj] += a[ii * t + k] * b[jj * t + k]

def test_outer_inner():
    """
    Test outer-inner product
    >>> a = np.array([[1., 2.], [3., 4.]])  # 2x2: i=2, t=2
    >>> b = np.array([[5., 6.], [7., 8.], [9., 10.]])  # 3x2: j=3, t=2
    >>> result = outer_inner(a, b)
    >>> result.shape
    (2, 3)
    >>> np.round(result, 2)
    array([[17., 23., 29.],
           [39., 53., 67.]])
    """

# Test to verify dimension passing with debugging
@cython.gufunc("(i,t),(j,t)->(i,j)")
cdef void outer_inner_debug(double* a, double* b, double* out, cnp.npy_intp i, cnp.npy_intp t, cnp.npy_intp j):
    """Debug version - stores dimension values in output"""
    # Store dimensions in first row for debugging
    out[0] = <double>i
    out[1] = <double>t
    out[2] = <double>j

def test_outer_inner_debug():
    """
    >>> a = np.array([[1., 2.], [3., 4.]])  # shape (2, 2)
    >>> b = np.array([[5., 6.], [7., 8.], [9., 10.]])  # shape (3, 2)
    >>> result = outer_inner_debug(a, b)
    >>> result.shape
    (2, 3)
    >>> result[0, :]  # Should show i, t, j = 2, 2, 3
    array([2., 2., 3.])
    """
@cython.gufunc("(n),(n)->()")
cdef void sum_of_products(double* a, double* b, double* out, cnp.npy_intp n):
    """Sum of element-wise products"""
    cdef cnp.npy_intp i
    out[0] = 0
    for i in range(n):
        out[0] += a[i] * b[i]

def test_sum_of_products():
    """
    Test that dimensions are passed correctly
    >>> a = np.array([1., 2., 3.])
    >>> b = np.array([4., 5., 6.])
    >>> result = sum_of_products(a, b)
    >>> float(result)
    32.0
    
    Test with different size
    >>> a = np.array([1., 2.])
    >>> b = np.array([3., 4.])
    >>> result = sum_of_products(a, b)
    >>> float(result)
    11.0
    """

# Test matrix trace with named dimensions
@cython.gufunc("(n,n)->()")
cdef void matrix_trace(double* m, double* out, cnp.npy_intp n):
    """Compute trace of square matrix"""
    cdef cnp.npy_intp i
    out[0] = 0
    for i in range(n):
        out[0] += m[i * n + i]

def test_matrix_trace():
    """
    Test computing matrix trace
    >>> m = np.array([[1., 2.], [3., 4.]])
    >>> result = matrix_trace(m)
    >>> float(result)
    5.0
    
    Test with 3x3 matrix
    >>> m = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    >>> result = matrix_trace(m)
    >>> float(result)
    6.0
    """

# Test multiple outputs with named dimensions
@cython.gufunc("(n)->(),()")
cdef void mean_and_sum(double* arr, double* mean_out, double* sum_out, cnp.npy_intp n):
    """Compute both mean and sum"""
    cdef cnp.npy_intp i
    sum_out[0] = 0
    for i in range(n):
        sum_out[0] += arr[i]
    mean_out[0] = sum_out[0] / <double>n

def test_mean_and_sum():
    """
    Test multiple outputs
    >>> arr = np.array([1., 2., 3., 4.])
    >>> mean, sum_val = mean_and_sum(arr)
    >>> float(mean)
    2.5
    >>> float(sum_val)
    10.0
    """

