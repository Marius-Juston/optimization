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
    _xmin = 0.0
    _fmin = 2.0

    def _f(self, x):
        return np.exp(x)+np.exp(-x)
    
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

def test_optimization_method_class(method_name, method_class: algorithms.Optimizer, test_funcs, method_args=[], tol=1e-5, iters=100, noise=False):
    results = []
    for tfunc in test_funcs:
        xmin, xmax = list(zip(*tfunc._domain))
        method = method_class(xmin, xmax, iters, *method_args)
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
        converged = (x_error < tol) or (val_error < tol)

        result = {
            'method': method_name,
            'noisy': noise,
            'function': tfunc().__str__(),
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
    print("-"*50)
    print(f"Method: {result[0]['method']}")
    print("-"*50)
    for r in test_result:
        print(f"Method: {r['method']}, Function: {r['function']}")
        print(f"  final_x = {r['final_x']:.6f}, final_val = {r['final_val']:.6f}")
        print(f"  true_min_x = {r['true_min_x']}, true_min_val = {r['true_min_val']}")
        print(f"  x_error = {r['x_error']:.6e}, val_error = {r['val_error']:.6e}")
        print(f"  Converged: {r['converged']}")
        print(f"  time: {r['time']:.6f}")
        print("-"*50)


if __name__ == "__main__":
    test_functions = [Quadratic, ExpSymmetric]

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
        # test_optimization_method_class("randomsearch_class", algorithms.RandomSearch, test_functions),
        test_optimization_method_class("bisection_class", algorithms.Bisection, test_functions)
    ]

    for result in results2:
        print_results(result)
        