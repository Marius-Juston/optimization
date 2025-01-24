from collections.abc import Iterable
from random import uniform
import numpy as np
import time

import algorithms

eval_f = lambda f, x: [f(a) for a in x] if isinstance(x, list) else f(x)


class TestFunction:
    _dim: int
    _domain: list[tuple[float, float]]
    _xmin: tuple[float, ...]
    _fmin: float

    def f(self, x, noise=False):
        if self._dim == 1:
            if isinstance(x, np.ndarray):
                x = float(x.item())
            assert isinstance(x, np.float64) or isinstance(x, float)
        else:
            assert len(x) == self._dim

        noise = self._noise() if noise else 0.0
        return self._f(x) + noise
    
    def _f(self, x):
        raise NotImplementedError
    
    def _noise(self):
        raise NotImplementedError
    

class Quadratic(TestFunction):
    _dim = 1
    _domain = [(-10.0, 10.0)]
    _xmin = (-1.0)
    _fmin = 5.0

    def _f(self, x):
        return 3*(x+1)**2+5
    
    def _noise(self):
        return np.random.normal(0, 3/8)
    

class ExpSymmetric(TestFunction):
    _dim = 1
    _domain = [(-2, 5)]
    _xmin = (0.0)
    _fmin = 2.0

    def _f(self, x):
        return np.exp(x)+np.exp(-x)
    
    def _noise(self):
        return np.random.normal(0, 3/8)


# https://github.com/facebook/Ax/blob/main/ax/utils/measurement/synthetic_functions.py
class Hartmann6(TestFunction):
    _dim = 6
    _domain = [(0.0, 1.0) for i in range(6)]
    _xmin = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    _fmin = -3.32237

    _alpha = np.array([1.0, 1.2, 3.0, 3.2])
    _A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    _P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    def _f(self, x):
        y = 0.0
        for j, alpha_j in enumerate(self._alpha):
            t = 0
            for k in range(6):
                t += self._A[j, k] * ((x[k] - self._P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return float(y)

    def _noise(self):
        return np.random.normal(0, 3/8)


def test_optimization_method(method_name, method_func, test_funcs, method_args=[], tol=1e-5, iters=100):
    results = []
    for tfunc in test_funcs:
        f = tfunc
        xmin, xmax = list(zip(*tfunc._domain))
        
        true_min_x = tfunc._xmin
        true_min_val = tfunc._fmin

        start = time.monotonic()
        x_star = method_func(f, xmin, xmax, iters, *method_args)
        end = time.monotonic()
        final_val = f(x_star)

        x_error = abs(x_star - true_min_x)
        val_error = abs(final_val - true_min_val)
        converged = (x_error < tol) or (val_error < tol)

        result = {
            'method': method_name,
            'function': tfunc['name'],
            'final_x': x_star,
            'final_val': final_val,
            'true_min_x': true_min_x,
            'true_min_val': true_min_val,
            'x_error': x_error,
            'val_error': val_error,
            'converged': converged,
            'time': end-start,
        }
        results.append(result)
    return results

def test_optimization_method_class(method_name, method_class: algorithms.Optimizer, test_funcs, iters, method_args=[], tol=1e-2, noise=False):
    assert len(test_funcs) == len(iters)

    results = []
    for i, tfunc in enumerate(test_funcs):
        iter = iters[i]
        xmin, xmax = list(zip(*tfunc._domain))
        method = method_class(xmin, xmax, iter, *method_args)
        f = lambda x: tfunc().f(x, noise=noise)
        true_min_x = tfunc._xmin
        true_min_val = tfunc._fmin

        start = time.monotonic()
        while not method.stop():
            x = method.ask()
            y = eval_f(f, x)
            method.tell(x, y)
        end = time.monotonic()

        x_star = method.best()
        final_val = f(x_star)

        x_error = float(np.linalg.norm(np.array(x_star) - np.array(true_min_x)))
        val_error = abs(final_val - true_min_val)
        converged = (x_error < tol)

        result = {
            'method': method_name,
            'noisy': noise,
            'function': tfunc.__name__,
            'final_x': x_star,
            'final_val': final_val,
            'true_min_x': true_min_x,
            'true_min_val': true_min_val,
            'x_error': x_error,
            'val_error': val_error,
            'converged': converged,
            'time': end-start,
        }
        results.append(result)
    return results


def print_results(test_result):
    print(f"Method: {result[0]['method']}")
    for r in test_result:
        print(f"\tFunction: {r['function']}, Noisy: {r['noisy']}")
        print(f"\t\tfinal_x = {r['final_x']}, final_val = {r['final_val']:.6f}")
        print(f"\t\ttrue_min_x = {r['true_min_x']}, true_min_val = {r['true_min_val']}")
        print(f"\t\tx_error = {r['x_error']:.6e}, val_error = {r['val_error']:.6e}")
        print(f"\t\tConverged: {r['converged']}")
        print(f"\t\ttime: {r['time']:.6f}")


if __name__ == "__main__":
    test_functions = [Hartmann6, Quadratic, ExpSymmetric]

    # results = [
    #     test_optimization_method("bisection", algorithms.bisection, test_functions),
    #     test_optimization_method("stochastic_bisection",
    #                              algorithms.bisection_with_noise, 
    #                              test_functions,
    #                              method_args=[0.05],
    #                              iters=2000, tol=1e-1),
    # ]
    
    # for result in results:
    #     print_results(result)

    results2 = [
        test_optimization_method_class("RandomSearch", algorithms.RandomSearch, [Hartmann6, Quadratic, ExpSymmetric], iters=[60, 10, 10]),
        test_optimization_method_class("Bayesian", algorithms.Bayesian, [Hartmann6, Quadratic, ExpSymmetric], iters=[60, 10, 10]),
        # test_optimization_method_class("bisection_class", algorithms.Bisection, [Quadratic, ExpSymmetric])
    ]

    for result in results2:
        print_results(result)
        