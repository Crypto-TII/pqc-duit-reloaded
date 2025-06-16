# Don't Use It Twice: Reloaded!

Accompanying repository to the manuscript titled "Don't Use It Twice: Reloaded! On the Lattice Isomorphism Group Action".

## Requirements

- The sofware `Sagemath`.
- The package `tqdm`, which can be installed as below:

```bash
sage
pip install tqdm
```

## Usage

Testing Lemma 1

```bash
sage -python test_lemma.py --n 10 --trials 100
```

Testing Heuristic 1

```bash
sage -python test_heuristic_1.py --n 10 --trials 100
```

Testing Heuristic 2

```bash
sage -python test_heuristic_2.py --n 10 --trials 100
```

Add the option `--solve` to test Theorem 2. **NOTE**: this last option supports only values of `n` that are a power of 2.

## License

Apache License Version 2.0, January 2004
