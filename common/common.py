import itertools

from multiprocessing import Pool, cpu_count
import numpy as np


def char_to_int(ch: str) -> int:
    return 10 if ch == "X" or ch == "0" else int(ch)

    
def read_numbers(path: str) -> list[int]:
    with open(path) as f:
        data = f.read()
    return list(map(char_to_int, data))


def numbers_to_averages(ns: list[int]):
    res = np.empty(len(ns))
    s = 0
    for i, n in enumerate(ns):
        s += n
        res[i] = s / (i + 1)
    return res


def average_sequence(balls) -> int:
    rng = np.random.default_rng()
    s = 0
    for i in itertools.count(1):
        s += balls[rng.integers(0, len(balls))]
        yield s / i
        

def _create_histogram(days: int, iterations: int, balls: list[int], precision):
    res = np.zeros(
        (int(12 / precision + 1), days),
        dtype=int
    )
    
    for _ in range(iterations):
        average = average_sequence(balls)
        for d in range(days):
            a = int(next(average) / precision)
            res[a][d] += 1
            
    return res


def create_histogram(days: int, iterations: int, balls: list[int], precision=0.01):
    cpus = cpu_count()
    with Pool(processes=cpus) as pool:
        work = [(days, iterations // cpus, balls, precision)] * cpus
        return sum(pool.starmap(_create_histogram, work))
    

def normalize_histogram(hist):
    total = sum(hist.T[0])
    return hist / total