from math import sqrt, log

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
            
    