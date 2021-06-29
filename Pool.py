from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
import numpy as np

def fun(a, b):
    return a + b

def funk(a):
    return a +a 
x = np.arange(100).reshape(-1, 2)

core = cpu_count()
pool = Pool(core)

z1 = pool.map(fun, x[:, 0], x[:,1])
z3 = pool.map(funk, x[:, 0])

pool.close()
pool.join()
