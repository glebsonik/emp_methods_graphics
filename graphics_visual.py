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
lam = 0.5
np.random.seed(42)
expX = np.random.exponential(1/lam, scale)

mu = 3
sigma = 0.25
normX = np.random.normal(mu, sigma, scale)
normX.sort()
expX.sort()
# print("norm: ", normX, "\n exp: ", expX)

#Exponential scatter
print("\n=========================== Экспоненциальное распределение ===========================")
d = expX[-1] - expX[0]
print("d= ", d, " (Размах)")
intervalCount = 3

expDelta = d/intervalCount
print("expDelta = ", expDelta)
expIntervals = makeIntervals(expX, expX[0], expDelta, intervalCount)
intTuples = []
intElCount = []
intAver = []
for interval in expIntervals:
    intAver.append(interval[0])
    intElCount.append(interval[1])
    intTuples.append(interval[2])

xIntTable = PrettyTable()
xIntTable.add_row(intTuples)
xIntTable.add_row(intElCount)
xIntTable.add_row(intAver)
print(xIntTable)

localEstTableExp = PrettyTable(["Ji", "xi*", "ni", "ui", "ni*ui", "ni*ui^2", "ni(ui + 1)^2"])
falseZero = expX[49]
xHash = {"niui": [],
         "niui_2": []}
for interval in expIntervals:
    ui = (interval[0] - falseZero)/expDelta
    niui = interval[1] * ui
    niuisqr = interval[1] * (ui * ui)
    controlCl = interval[1]*pow((ui + 1), 2)
    xHash["niui"].append(niui)
    xHash["niui_2"].append(niuisqr)
    currentRow = [interval[2], interval[0], interval[1], ui, niui, niuisqr, controlCl]
    localEstTableExp.add_row(currentRow)
print(localEstTableExp)

expMoment_1 = sum(xHash["niui"])/scale
expMoment_2 = sum(xHash["niui_2"])/scale

averageChosen = expDelta * expMoment_1 + falseZero
chosenDisp = (expMoment_2 - pow(expMoment_1, 2))*pow(expDelta, 2)

estLambda = 1/averageChosen
estLambdaSqr = 1/chosenDisp
print("Среднее выборочное: ", averageChosen, "Лямбда (ескп. закон): ", estLambda, "Лямбда квадрат: ", estLambdaSqr)

# X sqr
x_alpha = 77.93
x_beta = 140.17
gamma = 1 - 0.05*2  # alpha == beta
intLamSt = x_alpha/(2*scale*averageChosen)
intLamEnd = x_beta/(2*scale*averageChosen)
print("Границы доверительного интервала для лямбда: ", intLamSt, " ", intLamEnd)

# Hypothesis check
# expHypothesisTable = PrettyTable(["Ji", "ni", "n'i", "ni-n'i", "(ni-n'i)^2", "((ni-n'i)^2)/n'i"])
# for interval in expIntervals:
#     n_i =
#     niui = interval[1] * ui
#     niuisqr = interval[1] * (ui * ui)
#     controlCl = interval[1]*pow((ui + 1), 2)
#     xHash["niui"].append(niui)
#     xHash["niui_2"].append(niuisqr)
#     currentRow = [interval[2], interval[1], , ui, niui, niuisqr, controlCl]


# Normal scatter intervals
print("\n=========================== Нормальное распределение ===========================")
d = normX[-1] - normX[0]
print("d= ", d, " (Размах)")
intervalCount = 5

normDelta = d/intervalCount
print("Delta = ", normDelta)
normIntervals = makeIntervals(normX, normX[0], normDelta, intervalCount)

intTuples = []
intElCount = []
intAver = []
for interval in normIntervals:
    intAver.append(interval[0])
    intElCount.append(interval[1])
    intTuples.append(interval[2])

normIntTable = PrettyTable()
normIntTable.add_row(intTuples)
normIntTable.add_row(intElCount)
normIntTable.add_row(intAver)
print(normIntTable)

localEstTableNorm = PrettyTable(["xi*", "ni", "ui", "ni*ui", "ni*ui^2",
                                 "ni(ui + 1)^2", "ni*ui^3", "ni*ui^4", "ni*(ui+1)^4"])
expFalseZero = normX[49]
momentumTempHash = {"momentum_1": [],
                "momentum_2": [],
                "momentum_3": [],
                "momentum_4": [],}
for interval in normIntervals:
    ui = (interval[0] - expFalseZero)/normDelta
    niui = interval[1] * ui
    niuisqr = interval[1] * ui * ui
    controlCl_1 = interval[1] * pow((ui + 1), 2)
    niuicube = interval[1] * pow(ui, 3)
    niui_4 = interval[1] * pow(ui, 4)
    controlCl_2 = interval[1]*pow((ui+1), 4)
    momentumTempHash["momentum_1"].append(niui)
    momentumTempHash["momentum_2"].append(niuisqr)
    momentumTempHash["momentum_3"].append(niuicube)
    momentumTempHash["momentum_4"].append(niui_4)
    currentRow = [interval[0], interval[1], ui, niui, niuisqr, controlCl_1, niuicube, niui_4, controlCl_2]
    localEstTableNorm.add_row(currentRow)
print(localEstTableNorm)

# Local points estimation
moment_1 = (sum(momentumTempHash["momentum_1"]))/scale
moment_2 = (sum(momentumTempHash["momentum_2"]))/scale
moment_3 = (sum(momentumTempHash["momentum_3"]))/scale
moment_4 = (sum(momentumTempHash["momentum_4"]))/scale

expectedValEstimation = normDelta*moment_1 + expFalseZero
expectedDispEstimation = (moment_2 - pow(moment_1, 2))*pow(normDelta, 2)
expectedDeviation = pow(expectedDispEstimation, 0.5)
centMom_3 = (moment_3 - 3*moment_1*moment_2 + 2*pow(moment_1, 3))*pow(normDelta, 3)
centMom_4 = (moment_4 - 4*moment_1*moment_3 + pow(moment_1, 2)*6*moment_2 - 3*pow(moment_1, 4))*pow(normDelta, 4)

asymmetry = centMom_3/pow(expectedDeviation, 3)
excess = centMom_4/pow(expectedDeviation, 4) - 3

print("Оценка мат ожидания(среднее выборочное): ", expectedValEstimation, "\nВыборочная дисперсия: ", expectedDispEstimation,
      "\nОценка отклонения: ", expectedDeviation, "\nАсимметрия: ", asymmetry, "Эксцесс", excess)
# X sqr
t_coef = 1.66
temp_S = 0
for x in normX:
    temp_S += pow((x - expectedValEstimation), 2)
fixedChosenDisp = temp_S/(scale - 1)
intMSt = expectedValEstimation - (pow(fixedChosenDisp/scale, 0.5))*t_coef
intMEnd = expectedValEstimation + (pow(fixedChosenDisp/scale, 0.5))*t_coef

intDeviationSt = pow((fixedChosenDisp*(scale-1))/x_beta, 0.5)
intDeviationEnd = pow((fixedChosenDisp*(scale-1))/x_alpha, 0.5)

print("Интервальная оценка для нормального m: ", intMSt, intMEnd,
      "\nИнтервальная оценка для sigma:", intDeviationSt, intDeviationEnd)


# Graphic section
plt.hist(normX, 50)
# plt.show()
plt.hist(expX, 50, color='r')
# plt.show()
