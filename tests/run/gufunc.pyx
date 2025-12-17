# mode: run
# tag: numpy

cimport cython

@cython.gufunc("(),()->(3)")
cdef void with_signature(double a1, double a2, double* out):
    out[0] = a1 + a2
    out[1] = a1 * a2
    out[2] = a1 + a2 + 1

def test_with_signature():
    """
    >>> a1 = np.array([1., 2., 3., 4.])
    >>> a2 = np.array([4., 5., 6., 7.])
    >>> out = with_signature(a1, a2)
    >>> out.shape
    (4, 3)
    >>> out
    array([[ 5.,  4.,  6.],
           [ 7., 10.,  8.],
           [ 9., 18., 10.],
           [11., 28., 12.]])
    """