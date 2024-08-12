import numpy as np
from pymoo.core.problem import ElementwiseProblem

NO_CLIENTS = 140

def EMD(Z_i,Z_j):
    magnitude = lambda vector: math.sqrt(sum(pow(element, 2) for element in vector))
    Z_ij = Z_i+Z_j
    return magnitude(Z_ij/magnitude(Z_ij) - Z_global/magnitude(Z_global)) #+ (magnitude(np.sqrt(zi*zj)) / magnitude(Z_global))


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


problem = MyProblem()
print(MyProblem)
