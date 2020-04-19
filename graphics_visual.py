import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random

fig, ax = plt.subplots()
x1 = np.linspace(1,10,10)
x = np.arange(1, 5)
y1 = np.exp(x)
print(y1)
plt.plot(x, y1)
plt.show()
