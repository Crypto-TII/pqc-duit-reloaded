
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler
from sage.all import (
    matrix,
    IntegralLattice,
    GF,
    ZZ,
    sqrt,
    QQ,
    RR,
    log,
    e,
    pi,
    round
)

def get_distribution_parameter(Q):
    n = Q.nrows()
    B, _ = Q.cholesky().gram_schmidt()
    s = round(B.norm() * sqrt(log(2 * n + 4, e) / pi), ndigits=5)
    return s


def inner_product(u, q):
    """
    Inner product concerning the matrix Q
    :param u: unimodular matrix U over ZZ
    :param q: Gram matrix, which gives a quadratic form
    :return: Uᵀ×Q×U
    """
    return u.transpose() * q * u

def SampleLattice(n: int, factor: int, sampling: str):
    """
    Sampling a random positive definite matrix Q with integer coefficients in [-5n, 5n]
    :param n: matrix dimension
    :param sampling: Type of sampling: uniform or gaussian
    :return: Positive definite matrix Q
    """
    if sampling == 'uniform':
        b = matrix(ZZ, [[ZZ.random_element(-factor, factor) for _ in range(0, n, 1)] for _ in range(0, n, 1)])
        while (b.det() == 0):
            b = matrix(ZZ, [[ZZ.random_element(-factor, factor) for _ in range(0, n, 1)] for _ in range(0, n, 1)])
    else:
        assert sampling == 'gaussian'
        zzn = ZZ ** n
        sampling = DiscreteGaussianDistributionLatticeSampler(zzn, factor)
        b = matrix(ZZ, [sampling() for _ in range(0, n, 1)])
        while (b.det() == 0):
            b = matrix(ZZ, [sampling() for _ in range(0, n, 1)])

    q = b.transpose() * b
    L = IntegralLattice(q)
    return L, q

def sampling_from_dsq(q: matrix, n: int, s: float):
    """
    Sampling from Dₛ([Q])
    :param q: a quadratic form Q
    :param n: matrix dimension
    :param s: parameter required on the Discrete Gaussian Distribution
    :return: unimodular matrix U, and the quadratic form Uᵀ×Q×U
    """
    zzn = ZZ ** n
    # To check from Lemma 2.9 from eprint 2021/1332
    d = [DiscreteGaussianDistributionLatticeSampler(zzn, s, c=q[i]) for i in range(0, n, 1)]
    while True:
        y = []
        i = 0
        while i < n:
            vector_x = d[i]()
            if vector_x not in zzn.span(y):
                i += 1
                y.append(vector_x)

        y = matrix(ZZ, y)
        if y.det() != 0:
            break

    assert (y.det() != 0)
    t = y.echelon_form()
    u = y * t.inverse()
    r = inner_product(u, q)
    return u, r, y


def VecExtract(M, row):

    R1 = matrix(RR, int(sqrt(M.ncols()/2)), int(sqrt(M.ncols()/2)), M[row,0:M.ncols()/2].list() )
    R2 = matrix(RR, int(sqrt(M.ncols()/2)), int(sqrt(M.ncols()/2)), M[row,M.ncols()/2:M.ncols()].list() )
    return R1, R2

def vecToMatrix(vect, n):
    mat1 = matrix(RR, n, n, vect[:,0:n*n].list())
    mat2 = matrix(RR, n, n, vect[:,n*n:2*n*n].list())

    return mat1, mat2

