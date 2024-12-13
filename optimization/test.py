from random import uniform
import numpy as np
import algorithms


# Test functions
test_functions = [
    {
        'name': 'Quadratic_1',
        'f': lambda x: (x - 2)**2,
        'true_min_x': 2.0,
        'true_min_val': 0.0,
        'test_interval': (-10, 10)
    },
    # {
    #     'name': 'Quadratic_2',
    #     'f': lambda x: 3*(x+1)**2+5,
    #     'true_min_x': -1.0,
    #     'true_min_val': 5.0,
    #     'test_interval': (-10, 10)
    # },
    # {
    #     'name': 'Absolute_Value',
    #     'f': lambda x: abs(x),
    #     'true_min_x': 0.0,
    #     'true_min_val': 0.0,
    #     'test_interval': (-5, 5)
    # },
    # {
    #     'name': 'Exp_Symmetric',
    #     'f': lambda x: (np.exp(x)+np.exp(-x)),
    #     'true_min_x': 0.0,
    #     'true_min_val': 2.0,
    #     'test_interval': (-2, 2)
    # },
    # {
    #     'name': 'Quadratic_1_noise',
    #     'f': lambda x: (x - 2)**2 + np.random.normal(0, 3/8),
    #     'true_min_x': 2.0,
    #     'true_min_val': 0.0,
    #     'test_interval': (-10, 10)
    # },
    # {
    #     'name': 'Quadratic_2_noise',
    #     'f': lambda x: 3*(x+1)**2+5 + np.random.normal(0, 3/8),
    #     'true_min_x': -1.0,
    #     'true_min_val': 5.0,
    #     'test_interval': (-10, 10)
    # },
    {
        'name': 'Absolute_Value_noise',
        'f': lambda x: abs(x) + np.random.normal(0, 3/8),
        'true_min_x': 0.0,
        'true_min_val': 0.0,
        'test_interval': (-5, 5)
    },
    {
        'name': 'Exp_Symmetric_noise',
        'f': lambda x: (np.exp(x)+np.exp(-x)) + np.random.normal(0, 3/8),
        'true_min_x': 0.0,
        'true_min_val': 2.0,
        'test_interval': (-2, 2)
    },
]

def test_optimization_method(method_name, method_func, test_funcs, method_args=[], tol=1e-5, iters=100):
    results = []
    for tfunc in test_funcs:
        f = tfunc['f']
        k_min, k_max = tfunc['test_interval']
        true_min_x = tfunc['true_min_x']
        true_min_val = tfunc['true_min_val']

        x_star = method_func(f, k_min, k_max, iters, *method_args)
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
            'converged': converged
        }
        results.append(result)
    return results

# Example usage:
if __name__ == "__main__":
    tests = [
        test_optimization_method("bisection", algorithms.bisection, test_functions),
        test_optimization_method("stochastic_bisection", 
                                 algorithms.bisection_with_noise, 
                                 test_functions, 
                                 method_args=[0.01], 
                                 iters=500),
    ]
    
    for result in tests:
        print("-"*50)
        print(f"Method: {result[0]['method']}")
        print("-"*50)
        for r in result:
            print(f"Method: {r['method']}, Function: {r['function']}")
            print(f"  final_x = {r['final_x']:.6f}, final_val = {r['final_val']:.6f}")
            print(f"  true_min_x = {r['true_min_x']}, true_min_val = {r['true_min_val']}")
            print(f"  x_error = {r['x_error']:.6e}, val_error = {r['val_error']:.6e}")
            print(f"  Converged: {r['converged']}")
            print("-"*50)
        