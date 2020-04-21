import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
from prettytable import PrettyTable
from scipy.stats import chi2
import scipy

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
print("Лямда: ", lam)
d = expX[-1] - expX[0]
print("Размах: ", d)
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

#Local points estimation
estLambda = 1/averageChosen
estLambdaSqr = 1/chosenDisp
print("Среднее выборочное: ", averageChosen, "Лямбда (ескп. закон): ", estLambda, "Лямбда квадрат: ", estLambdaSqr)

# X sqr
alpha_chi_coef = 0.05
x_alpha = chi2.ppf(alpha_chi_coef, scale*2)
x_beta = chi2.ppf(1 - alpha_chi_coef, scale*2)
gamma = 1 - alpha_chi_coef*2  # alpha == beta
intLamSt = x_alpha/(2*scale*averageChosen)
intLamEnd = x_beta/(2*scale*averageChosen)
print("Границы доверительного интервала для лямбда: ", intLamSt, " ", intLamEnd, "\nгамма: ", gamma)

# Hypothesis check
expHypothesisTable = PrettyTable(["Ji", "ni", "n'i", "ni-n'i", "(ni-n'i)^2", "((ni-n'i)^2)/n'i"])
exp_x_chosen = 0
for interval in expIntervals:
    n_i = scale*(math.exp(-1*estLambda*interval[2][0]) - math.exp(-1*estLambda*interval[2][1]))
    ni_sub = interval[1] - n_i
    print("ni", interval[1], "n_i", n_i, "ni sub", ni_sub)
    ni_sub_sqr = pow(interval[1]-n_i, 2)
    control = ni_sub_sqr/n_i
    currentRow = [interval[2], interval[1], n_i, ni_sub, ni_sub_sqr, control]
    exp_x_chosen += control
    expHypothesisTable.add_row(currentRow)
print(expHypothesisTable)
importance_level = 0.1
x_critical = chi2.ppf(importance_level, intervalCount - 2)
print("Хи^2 кр: ", x_critical, "Хи^2 выб", exp_x_chosen)
if x_critical > exp_x_chosen:
    print("Гипотеза принята на уровне значимости альфа:", importance_level)
else:
    print("Гипотеза отвержена")

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
temp_S = 0  # temporary var for sum to get X of set (среднее выборочное)
for x in normX:
    temp_S += pow((x - expectedValEstimation), 2)
fixedChosenDisp = temp_S/(scale - 1)
intMSt = expectedValEstimation - (pow(fixedChosenDisp/scale, 0.5))*t_coef
intMEnd = expectedValEstimation + (pow(fixedChosenDisp/scale, 0.5))*t_coef

alpha_chi_coef = 0.05
x_alpha = chi2.ppf(alpha_chi_coef, scale-1)
x_beta = chi2.ppf(1 - alpha_chi_coef, scale-1)
intDeviationSt = pow((fixedChosenDisp*(scale-1))/x_beta, 0.5)
intDeviationEnd = pow((fixedChosenDisp*(scale-1))/x_alpha, 0.5)

print("Интервальная оценка для нормального m: ", intMSt, intMEnd,
      "\nИнтервальная оценка для sigma:", intDeviationSt, intDeviationEnd)

normHypothesisTable = PrettyTable(["xi", "xi+1", "xi - x_ch", "xi+1 - x_ch",
                                   "Zi = (xi-x_ch)/sigma", "Zi+1 = ((xi+1)-x_ch)/sigma"])
chosenSigma = pow(expectedDispEstimation, 0.5)
zArray = []
for interval in normIntervals:
    x_ch = expectedValEstimation
    xi = interval[2][0]
    xi_1 = interval[2][1]
    xi_sub_xch = xi - x_ch
    xi_1_sub_xch = xi_1 - x_ch
    zi = xi_sub_xch/chosenSigma
    zi_1 = xi_1_sub_xch/chosenSigma
    zArray.append([(zi, zi_1), interval[1]])
    normHypothesisTable.add_row([xi, xi_1, xi_sub_xch, xi_1_sub_xch, zi, zi_1])
print(normHypothesisTable)

zNormHypTable = PrettyTable(["zi", "zi+1", "Fo(zi)", "Fo(zi+1)", "Pi = Fo(zi) - Fo(zi+1)", "nPi = ni'"])
chiNormHypTable = PrettyTable(["ni", "ni'", "(ni-ni')^2", "((ni-ni')^2)/ni'"])
xNorm = 0
for i in zArray:
    zi = i[0][0]
    zi_1 = i[0][1]
    fi = scipy.stats.norm.cdf(zi)-0.5
    fi_1 = scipy.stats.norm.cdf(zi_1)-0.5
    pi = abs(fi_1 - fi)
    n_i = scale*pi
    xNorm += (pow(i[1] - n_i, 2))/n_i
    zNormHypTable.add_row([zi, zi_1, fi, fi_1, pi, n_i])
    chiNormHypTable.add_row([i[1], n_i, pow(i[1] - n_i, 2), (pow(i[1] - n_i, 2))/n_i])
print(zNormHypTable)
print(chiNormHypTable)

importance_level = 0.1
x_critical = chi2.ppf(importance_level, intervalCount - 3)
print("Хи^2 кр: ", x_critical, "Хи^2 выб: ", xNorm)
if xNorm < x_critical:
    print("гипотеза принята на уровне значимости альфа: ", importance_level)
else:
    print("Гипотеза отвержена")

# Graphic section
plt.hist(normX, 50)
# plt.show()
plt.hist(expX, 50, color='r')
# plt.show()
