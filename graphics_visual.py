import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random

# [start, end)
def elementInRnage(elements, start, end):
    count = 0
    for el in elements:
        if (el >= start) and (el < end):
            count += 1
    return count


def makeIntervals(samples, start, delt, count):
    samples.sort()
    intervals = []
    for i in range(1, count+1):
        currentStart = start+(i-1)*delt
        currentEnd = start + i*delt
        print(currentStart, currentEnd)
        intervals.append([(currentStart + currentEnd)/2, elementInRnage(samples, currentStart, currentEnd)])
    intervals[-1][1] += 1
    return intervals

mas = [1,2,3,4,5,6,7,8,9,10]
print(makeIntervals(mas, 1, 9/3, 3))
lam = 9
scale = 100
np.random.seed(25)
expX = np.random.exponential(lam, scale)

mu = 3
sigma = 0.25
normX = np.random.normal(mu, sigma, scale)
normX.sort()
expX.sort()
print("norm: ", normX, "\n exp: ", expX)

d = normX[-1] - normX[0]
print("d= ", d, " (Размах)")
intervalCount = 5

delta = d/intervalCount
print("Delta = ", delta)
print(makeIntervals(normX, normX[0], delta, intervalCount))

#Graphic section
plt.hist(normX, 50)
# plt.show()
plt.hist(expX, 50, color='r')
# plt.show()
