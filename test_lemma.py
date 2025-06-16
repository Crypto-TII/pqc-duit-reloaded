
from sage.all import *
from utils import *
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor



def process_task(task):

    (i, n, s, lat_sampling_factor) = task

    # sample class representative
    _, Q = SampleLattice(n, lat_sampling_factor, "uniform")
    if (s == 0):
        s = get_distribution_parameter(Q)

    # sample 4 quadratic forms
    _, Q0, _ = sampling_from_dsq(Q, n, s)
    _, Q1, _ = sampling_from_dsq(Q, n, s)
    U, Q0_, _ = sampling_from_dsq(Q0, n, s)
    Q1_ = U.transpose()*Q1*U

    # construct matrix A
    I = identity_matrix(ZZ, n)
    A1 = Q0_.tensor_product(I).augment(I.tensor_product(-Q0))
    A2 = Q1_.tensor_product(I).augment(I.tensor_product(-Q1))
    A = A1.stack(A2)
    A = A.change_ring(QQ)

    # check the rank
    if (rank(A) <= 2*(n**2) -n):
        return i, 1
    else:
        return i, 0


def test_n_random_instances(n, s, lat_sampling_factor, trials, verbose=False):
    # Flatten the loops into a single list of tasks
    tasks = [
        (i, n, s, lat_sampling_factor)
        for i in range(trials)
    ]

    # Parallelize the tasks
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_task, tasks),total=len(tasks)))

    # Collect results back into the all_responses structure
    sum = 0
    for _, result in results:
        sum += result

    print(f"Result: max_rank/total = {sum}/{trials} = {round(sum/trials,4)}")

    return 0




if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Plot how much the rank of the system decreases when applying Rouché-Capelli row-column guessing.')
    parser.add_argument('--n', type=int, default=10, help='Dimension of the lattice')
    parser.add_argument('--s', type=float, default=0.5, help='Sampling parameter. Set to zero to use the lower bound from DvW22')
    parser.add_argument('--lat_sampling_factor', type=int, default=2, help='Bound when sampling lattice.')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity.')

    # Parse the command-line arguments
    args = parser.parse_args()

    test_n_random_instances(args.n, args.s, args.lat_sampling_factor, args.trials, args.verbose)



