import warnings
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import sys

warnings.filterwarnings("ignore") # specify to ignore warning messages

file_name = sys.argv[1]
def parser(x):
	return datetime.datetime.strptime(x, '%d-%m-%Y')
y = pd.read_csv(file_name, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

pdqls = []
spdqls = []
aicls = []


for param in pdq:
	for param_seasonal in seasonal_pdq:
		try:
			mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
			results = mod.fit()
			print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
			pdqls.append(param)
			spdqls.append(param_seasonal)
			aicls.append(results.aic)
		except:
			continue

alllist = [[] for k in range(len(pdqls))]
for j in range(len(pdqls)):
	alllist[j].append(pdqls[j])
	alllist[j].append(spdqls[j])
	alllist[j].append(aicls[j])
alllist =  sorted(alllist, key=lambda x: x[2])

mod = sm.tsa.statespace.SARIMAX(y,order=alllist[0][0],seasonal_order=alllist[0][1],enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Get forecast 9 steps ahead in future i.e. for 9 months in future in this case
pred_uc = results.get_forecast(steps=9)

print('pred_uc')
print(pred_uc)
# Get confidence intervals of forecasts
pred_ci2 = pred_uc.conf_int()
print('pred_ci2')
print(pred_ci2)

y1 = pred_ci2.iloc[:, 0]
y2 = pred_ci2.iloc[:, 1]
mse3 = ((y1 - y2) ** 2).mean()
print('The Mean Squared Error of our dynamic forecasts is {}'.format(round(mse3, 2)))

print('y1')
print(y1)
print('y2')
print(y2)
ax = y.plot(label='observed')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci2.index,pred_ci2.iloc[:, 0],pred_ci2.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sales Count')

plt.legend()
plt.show()
