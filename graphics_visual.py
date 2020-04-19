import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random

lam = 9
scale = 100
np.random.seed(25)
expX = np.random.exponential(lam, scale)

mu = 3
sigma = 0.25
normX = np.random.normal(mu, sigma, scale)

plt.hist(normX, 50)
plt.show()
plt.hist(expX, 50, color='r')
plt.show()