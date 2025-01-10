from math import sqrt, log
import numpy as np


class Optimizer:
    def __init__(self, xmin, xmax, iter):
        if type(xmin) != int and len(xmin) != len(xmax):
            raise ValueError("xmin and xmax must have same dimension")
        self._xmin = xmin
        self._xmax = xmax
        self._iter = iter

    def ask(self):
        raise NotImplementedError

    def tell(self, x, y):
        raise NotImplementedError
    
    def best(self):
        raise NotImplementedError

    def stop(self):
        return self._iter <= 0


class RandomSearch(Optimizer):
    def __init__(self, xmin, xmax, iter):
        super().__init__(xmin, xmax, iter)
        self._x = None
        self._y = float('inf')

    def ask(self):
        return np.random.uniform(low=self._xmin, high=self._xmax)

    def tell(self, x, y):
        if y < self._y:
            self._x = x
            self._y = y
        self._iter -= 1

    def best(self):
        return self._x
    

class Bisection(Optimizer):
    def __init__(self, xmin, xmax, iter):
        super().__init__(xmin, xmax, iter)
        if type(xmin) != float and type(xmin) != int:
            raise ValueError("bisection only supports one dimensional search")
        self._leftbound = xmin
        self._rightbound = xmax

    def ask(self):
        x0 = (2 / 3) * self._leftbound + (1 / 3) * self._rightbound
        x1 = (1 / 3) * self._leftbound + (2 / 3) * self._rightbound
        return [x0, x1]
    
    def tell(self, x, y):
        x0, x1 = x
        y0, y1 = y
        if y1 >= y0:
            self._rightbound = x1
        else:
            self._leftbound = x0
        self._iter -= 1

    def best(self):
        return (self._leftbound + self._rightbound) / 2


def bisection(f, k_min, k_max, iters):
    for _ in range(iters):
        x0 = (2 / 3) * k_min + (1 / 3) * k_max
        x1 = (1 / 3) * k_min + (2 / 3) * k_max
        if f(x1) >= f(x0):
            k_max = x1
        else:
            k_min = x0
    return (k_min + k_max) / 2


# A learner interacts with the environment over n roounds
def bisection_with_noise(f, k_min, k_max, iters, delta):  # page 54
    def bisect(x, y, n, delta):
        Ys = []
        xs = [
            (3 / 4) * x + (1 / 4) * y,  # x0
            (1 / 2) * x + (1 / 2) * y,  # x1
            (1 / 4) * x + (3 / 4) * y,  # x2
        ]
        for t in range(1, n+1):
            ct = sqrt((6 / t) * log(n / delta))
            
            Xt = xs[t % 3]
            Ys.append(f(Xt))
            
            if t % 3 == 0:
                fhat = lambda k: (3 / t) * sum([Ys[u-1] for u in range(1, t+1) if u % 3 == k])
                if fhat(2) - fhat(1) >= ct:
                    return x, xs[2]
                if fhat(0) - fhat(1) >= ct:
                    return xs[0], y
                
        return (x, y)
    
    z = 1 + ( log(iters) / log(4/3) )
    for _ in range(1, iters+1):
        k_min, k_max = bisect(x=k_min, y=k_max, n=iters, delta=delta/z)
        
    return (k_min + k_max) / 2
            
    