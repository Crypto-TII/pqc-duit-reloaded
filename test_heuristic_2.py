from http.client import error

from sage.all import *
from utils import *
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def n_th_root(input, n):
    fact_n = factor(n)
    if len(fact_n) == 1 and fact_n[0][0] == 2:
        k = fact_n[0][1]
        for _ in range(k):
            input = ZZ(sqrt(input))
    else:
        print("ERROR: Unsupported value of n.")
        exit(0)
    return input

def process_task(task):

    (i, n, s, lat_sampling_factor, solve) = task

    P = PolynomialRing(QQ, name="x")
    x = P.gen()

    # sample class representative
    _, Q = SampleLattice(n, lat_sampling_factor, "uniform")
    if (s == 0):
        s = get_distribution_parameter(Q)

    # sample 6 quadratic forms
    U, Q0, _ = sampling_from_dsq(Q, n, s)
    _, Q1, _ = sampling_from_dsq(Q, n, s)
    _, Q2, _ = sampling_from_dsq(Q, n, s)
    Q0_ = U.transpose() * Q0 * U
    Q1_ = U.transpose() * Q1 * U
    Q2_ = U.transpose() * Q2 * U

    # construct matrix A
    I = identity_matrix(ZZ, n)
    A1 = Q0_.tensor_product(I).augment(I.tensor_product(-Q0))
    A2 = Q1_.tensor_product(I).augment(I.tensor_product(-Q1))
    A3 = Q2_.tensor_product(I).augment(I.tensor_product(-Q2))
    A = A1.stack(A2).stack(A3)

    if solve: # solve the system / test theorem
        A = A.change_ring(QQ)  # faster kernel computation
        solution = A.right_kernel_matrix()
        Ut = matrix(QQ, n, n, solution[:, n ** 2:2 * n ** 2].list())
        V = matrix(QQ, n, n, solution[:, 0 : n ** 2].list())
        Vdet = 1/V.determinant()
        if Vdet < 0:
            Vdet = -Vdet
        y = n_th_root(Vdet, n)
        Ut = Ut * y # experimentally, the determinant is always +-1
        for i in range(n):
            for j in range(n):
                try:
                    Ut[i,j] = ZZ(Ut[i,j]) # round to remove noise from floating point precision
                except:
                    print(Ut)
                    return i, 0
        Ut = Ut.change_ring(ZZ) # change to integer to check for equality
        if Ut == U.transpose() or Ut == -U.transpose():
            return i, 1
        else:
            return i, 0
    else: # test heuristic only
        A = A.change_ring(QQ)  # faster kernel computation
        # check the rank
        if (rank(A) == 2*(n**2) -1):
            return i, 1
        else:
            return i, 0

def test_n_random_instances(n, s, lat_sampling_factor, trials, solve=False, verbose=False):
    # Flatten the loops into a single list of tasks
    tasks = [
        (i, n, s, lat_sampling_factor, solve)
        for i in range(trials)
    ]

    # Parallelize the tasks
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_task, tasks),total=len(tasks)))

    # Collect results back into the all_responses structure
    sum = 0
    for _, result in results:
        sum += result

    if solve:
        print(f"Result: recovered/total = {sum}/{trials} = {round(sum / trials, 4)}")
    else:
        print(f"Result: max_rank/total = {sum}/{trials} = {round(sum/trials,4)}")

    return 0




if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot how much the rank of the system decreases when applying Rouché-Capelli row-column guessing.')
    parser.add_argument('--n', type=int, default=10, help='Dimension of the lattice')
    parser.add_argument('--s', type=float, default=0.5, help='Sampling parameter. Set to zero to use the lower bound from DvW22')
    parser.add_argument('--lat_sampling_factor', type=int, default=2, help='Bound when sampling lattice.')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials.')
    parser.add_argument('--solve',  action='store_true', help='Test whether the recovered secret corresponds to the original one')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')

    # Parse the command-line arguments
    args = parser.parse_args()

    test_n_random_instances(args.n, args.s, args.lat_sampling_factor, args.trials, args.solve, args.verbose)



