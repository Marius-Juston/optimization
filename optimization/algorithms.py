from math import sqrt, log
import numpy as np

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

import cma


class Optimizer:
    def __init__(self, xmin: list[float], xmax: list[float], iter: int):
        if type(xmin) != float and len(xmin) != len(xmax):
            raise ValueError("xmin and xmax must have same dimension")
        self._xmin = xmin
        self._xmax = xmax
        self._iter = iter

    def ask(self) -> list[np.ndarray]:
        raise NotImplementedError

    def tell(self, x: list[np.ndarray], y: list[float]):
        raise NotImplementedError
    
    def best(self) -> np.ndarray:
        raise NotImplementedError

    def stop(self) -> bool:
        return self._iter <= 0
    

class CMA_ES(Optimizer):
    def __init__(self, xmin, xmax, iter, x0=None, sigma0=None, seed=None):
        super().__init__(xmin, xmax, iter)

        # TODO: could initialize x0 with some other method, since CMA_ES
        # is a local optimizer

        if x0 is None:
            x0 = (( np.array(xmin) + np.array(xmax) ) / 2).tolist()
        if sigma0 is None:
            # "x0 should be within sigma +- 3 of xstar"
            sigma0 = float(np.mean( ( np.abs(np.array(xmin)) + np.abs(np.array(xmax)) ) / 6 ))
            
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, inopts={'seed': seed})

    def ask(self) -> list[np.ndarray]:
        return self.es.ask()

    def tell(self, x: list[np.ndarray], y: list[float]):
        self.es.tell(x, y)
        self._iter -= 1
    
    def best(self) -> np.ndarray:
        return self.es.result.xfavorite


class Bayesian(Optimizer):
    def __init__(self, xmin, xmax, iter):
        super().__init__(xmin, xmax, iter)

        initialization_iter = 5
        
        gs = GenerationStrategy(
            steps=[
                # Initialization
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=initialization_iter,
                    min_trials_observed=5,
                    max_parallelism=1,
                    model_kwargs={"seed": 42},
                    model_gen_kwargs={},
                ),
                # Bayesian optimization
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=1,
                )
            ]
        )
        self.ax_client = AxClient(generation_strategy=gs)
        self.ax_client.create_experiment(
            name="my_optimization",
            parameters=[
                {
                    "name": f"param{i}",
                    "type": "range",
                    "bounds": [xmin[i], xmax[i]],
                    "value_type": "float",
                } for i in range(len(xmin))
            ],
            objectives={"cost": ObjectiveProperties(minimize=True)},
            # could include parameter constraints or outcome constraints
            # https://ax.dev/tutorials/tune_cnn_service.html
        )
        self.last_trial_index = None

    def ask(self) -> list[np.ndarray]:
        parameters, self.last_trial_index = self.ax_client.get_next_trial()
        return np.array([v for k, v in parameters.items()])

    def tell(self, x: list[np.ndarray], y: list[float]):
        assert len(y) == 1
        self.ax_client.complete_trial(trial_index=self.last_trial_index, raw_data=y[0])
        self._iter -= 1

    def best(self) -> np.ndarray:
        tup = self.ax_client.get_best_parameters()
        if tup is None:
            return None
        else:
            best = tup[0]
            return np.array([v for k, v in best.items()])        


class RandomSearch(Optimizer):
    def __init__(self, xmin, xmax, iter):
        super().__init__(xmin, xmax, iter)
        self._x = None
        self._y = float('inf')

    def ask(self) -> list[np.ndarray]:
        return [np.random.uniform(low=self._xmin, high=self._xmax)]

    def tell(self, x: list[np.ndarray], y: list[float]):
        for a, b in zip(x, y):
            if b < self._y:
                self._x = a
                self._y = b
            self._iter -= 1

    def best(self) -> np.ndarray:
        return self._x
    

class Bisection(Optimizer):
    def __init__(self, xmin, xmax, iter):
        super().__init__(xmin, xmax, iter)

        if not isinstance(xmin, float) and len(xmin) == 1:
            xmin = xmin[0]
        if not isinstance(xmax, float) and len(xmax) == 1:
            xmax = xmax[0]
        
        if type(xmin) != float and type(xmin) != int:
            raise ValueError("bisection only supports one dimensional search")
        
        self._leftbound = np.array([xmin])
        self._rightbound = np.array([xmax])

    def ask(self) -> list[np.ndarray]:
        x0 = (2 / 3) * self._leftbound + (1 / 3) * self._rightbound
        x1 = (1 / 3) * self._leftbound + (2 / 3) * self._rightbound
        return [x0, x1]
    
    def tell(self, x: list[np.ndarray], y: list[float]):
        x0, x1 = x
        y0, y1 = y
        if y1 >= y0:
            self._rightbound = x1
        else:
            self._leftbound = x0
        self._iter -= 1

    def best(self) -> np.ndarray:
        return (self._leftbound + self._rightbound) / 2
    

class MultiBisection(Optimizer):
    def __init__(self, xmin, xmax, iter):
        super().__init__(xmin, xmax, iter)
        self.optims = [Bisection(*args) for args in zip(xmin, xmax, iter)]

    def ask(self) -> list[np.ndarray]:
        x_new = [o.ask() for o in self.optims]
        x_new = list(zip(*x_new))
        x_new = [np.array(x) for x in x_new]
        return x_new
    
    def tell(self, xs: list[np.ndarray], ys):
        assert all(isinstance(x, np.ndarray) for x in xs)
        xs_ordered = list(zip(*xs))
        ys_ordered = list(zip(*ys))
        for x, y, o in zip(xs_ordered, ys_ordered, self.optims):
            o.tell(x,y) # (x0, x1) (y0, y1)
    
    def best(self) -> np.array:
        return np.array([o.best() for o in self.optims])
    
    def stop(self) -> bool:
        return any(o.stop() for o in self.optims)


# TODO: implement ask-and-tell
# A learner interacts with the environment over n roounds
def bisection_with_noise(f, k_min, k_max, iters, delta):  # page 54
    inner_t = 0
    def bisect(x, y, n, delta):
        nonlocal inner_t
        Ys = []
        xs = [
            (3 / 4) * x + (1 / 4) * y,  # x0
            (1 / 2) * x + (1 / 2) * y,  # x1
            (1 / 4) * x + (3 / 4) * y,  # x2
        ]
        for t in range(1, n+1):
            inner_t += 1
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
    while True:
        if inner_t > iters: break
        k_min, k_max = bisect(x=k_min, y=k_max, n=iters-inner_t+1, delta=delta/z)
        print(f"{k_min:.3f} {k_max:.3f}")
        
    return (k_min + k_max) / 2
            
    