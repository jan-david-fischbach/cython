# mode: run
# tag: numpy

cimport cython

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
    array([11., 12., 13., 14.])
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
    array([[3., 5., 4.],
           [6., 10., 8.]])
    """