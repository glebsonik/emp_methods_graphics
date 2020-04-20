import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
from prettytable import PrettyTable

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
        intervals.append([(currentStart + currentEnd)/2,
                          elementInRnage(samples, currentStart, currentEnd),
                          (currentStart, currentEnd)])
    intervals[-1][1] += 1
    return intervals

scale = 100
lam = 9
np.random.seed(25)
expX = np.random.exponential(lam, scale)

mu = 3
sigma = 0.25
normX = np.random.normal(mu, sigma, scale)
normX.sort()
expX.sort()
# print("norm: ", normX, "\n exp: ", expX)

d = normX[-1] - normX[0]
print("d= ", d, " (Размах)")
intervalCount = 5

delta = d/intervalCount
print("Delta = ", delta)
normIntervals = makeIntervals(normX, normX[0], delta, intervalCount)
intTuples = []
intElCount = []
intAver = []
for interval in normIntervals:
    intAver.append(interval[0])
    intElCount.append(interval[1])
    intTuples.append(interval[2])

nIntTable = PrettyTable()
nIntTable.add_row(intTuples)
nIntTable.add_row(intElCount)
nIntTable.add_row(intAver)
print(nIntTable)

localEstTableNorm = PrettyTable(["Ji", "xi*", "ni", "ui", "ni*ui", "ni*ui^2", "ni(ui + 1)^2"])
falseZero = normX[49]
for interval in normIntervals:
    ui = (interval[0] - falseZero)/delta
    niui = interval[1] * ui
    niuisqr = interval[1] * ui * ui
    controlCl = interval[1]*pow((ui + 1), 2)
    currentRow = [interval[2], interval[0], interval[1], ui, niui, niuisqr, controlCl]
    localEstTableNorm.add_row(currentRow)
print(localEstTableNorm)

#Graphic section
plt.hist(normX, 50)
# plt.show()
plt.hist(expX, 50, color='r')
# plt.show()
