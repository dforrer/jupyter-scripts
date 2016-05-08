
from IPython.Parallel import Client
c = Client()
view = c.load_balanced_view()


import numpy as np
import pandas as pd
import pickle

class MyLongAssComputation(object):
    def compute(self, x):
        return x**2

def compute_func(x):
    c = MyLongAssComputation()
    return c.compute(x)


x = np.arange(1, 1000)
squared = view.map_sync(compute_func, x)
pickle.dump(squared, open('output.pickle', 'wb'))
