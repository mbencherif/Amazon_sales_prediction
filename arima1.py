import warnings
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime

##plt.style.use('fivethirtyeight')

'''
data = sm.datasets.co2.load_pandas()
y = data.data
#print(y)
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
print(y)
print(type(y))
y.plot(figsize=(15, 6))
plt.show()
'''


def parser(x):
	return datetime.datetime.strptime(x, '%d-%m-%Y')
y = pd.read_csv('150215.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
##y.plot()
##plt.show()
'''
fname = '152018tstunif.csv'
y = pd.read_csv(fname)
y3 = y
y3.set_index('date')
y2 = y3[(y3['date'] >= "01-01-2015") & (y3['date'] <= "24-11-2016")]
y2 = y2.iloc[:, :2]
'''

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

##print('Examples of parameter combinations for Seasonal ARIMA...')
##print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
##rint('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
##print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
##print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
	for param_seasonal in seasonal_pdq:
		try:
			mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
			results = mod.fit()
			print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
		except:
			continue

mod = sm.tsa.statespace.SARIMAX(y,order=(1, 1, 0),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
#print(results.summary().tables[1])
##results.plot_diagnostics(figsize=(15, 12))
##plt.show()

smp = pd.to_datetime("1-09-2016",format='%d-%m-%Y')
print(smp)
pred = results.get_prediction(start=pd.to_datetime("1-09-2016",format='%d-%m-%Y'), dynamic=False)
pred_ci = pred.conf_int()
ax = y["1-09-2016":].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Predictions', alpha=.7)

ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Sales Count')
plt.legend()
##plt.show()

y_forecasted = pred.predicted_mean
y_truth = y["1-09-2016":]

# Compute the mean square error
mse1 = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our osf predictions is {}'.format(round(mse1, 2)))

pred_dynamic = results.get_prediction(start=pd.to_datetime("01-09-2016",format='%d-%m-%Y'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = y["01-09-2016":].plot(label='observed')
pred_dynamic.predicted_mean.plot(label='Dynamic predictions', ax=ax)

ax.fill_between(pred_dynamic_ci.index,pred_dynamic_ci.iloc[:, 0],pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime("01-09-2016",format='%d-%m-%Y'), y.index[-1],alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('Sales Count')

plt.legend()
plt.show()

# Extract the predicted and true values of our time series for dynamic forecast
y_forecasted = pred_dynamic.predicted_mean
y_truth = y["01-09-2016":]

# Compute the mean square error
mse2 = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our dynamic predictions is {}'.format(round(mse2, 2)))

# Get forecast 6 steps ahead in future i.e. for 9 months in future in this case
pred_uc = results.get_forecast(steps=9)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
print('pred_ci {}'.format(pred_ci))


y1 = pred_ci.iloc[:, 0]
y2 = pred_ci.iloc[:, 1]
print('y1 {}'.format(y1))
print('y2 {}'.format(y2))
mse3 = ((y1 - y2) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse3, 2)))
print('pred_uc.predicted_mean {}'.format(pred_uc.predicted_mean))
ax = y.plot(label='observed')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sales Count')

plt.legend()
plt.show()
