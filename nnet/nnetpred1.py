import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import logging
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") # specify to ignore warning messages

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

'''
Whenever I used to run this used to give different result. 
To have always the same result, try to propose a fixed seed for the initial weights.
'''
seed = 7
np.random.seed(seed)

fname = "1520-18.csv"

for filename in os.listdir("."):


	class TimeSeriesNnet(object):
		def __init__(self, hidden_layers = [20, 15, 5], activation_functions = ['relu', 'relu', 'relu'], 
	              optimizer = SGD(), loss = 'mean_squared_error'):
			self.hidden_layers = hidden_layers
			self.activation_functions = activation_functions
			self.optimizer = optimizer
			self.loss = loss
			#print(loss)
			if len(self.hidden_layers) != len(self.activation_functions):
				raise Exception("hidden_layers size must match activation_functions size")

		def fit(self, timeseries, lag = 7, epochs = 10000, verbose = 0):
			self.timeseries = np.array(timeseries, dtype = "float64") # Apply log transformation por variance stationarity
			self.lag = lag
			self.n = len(timeseries)
			if self.lag >= self.n:
				raise ValueError("Lag is higher than length of the timeseries")
			self.X = np.zeros((self.n - self.lag, self.lag), dtype = "float64")
			self.y = np.log(self.timeseries[self.lag:])
			self.epochs = epochs
			self.scaler = StandardScaler()
			self.verbose = verbose

			logging.info("Building regressor matrix")
			# Building X matrix
			for i in range(0, self.n - lag):
				self.X[i, :] = self.timeseries[range(i, i + lag)]

			logging.info("Scaling data")
			self.scaler.fit(self.X)
			self.X = self.scaler.transform(self.X)

			logging.info("Checking neural-network consistency")
			# Neural net architecture
			self.nn = Sequential()
			self.nn.add(Dense(self.hidden_layers[0], input_shape = (self.X.shape[1],)))
			self.nn.add(Activation(self.activation_functions[0]))

			for layer_size, activation_function in zip(self.hidden_layers[1:],self.activation_functions[1:]):
				self.nn.add(Dense(layer_size))
				self.nn.add(Activation(activation_function))

			# Add final node
			self.nn.add(Dense(1))
			self.nn.add(Activation('linear'))
			self.nn.compile(loss = self.loss, optimizer = self.optimizer)

			logging.info("Training neural net")
			# Train neural net
			self.nn.fit(self.X, self.y, nb_epoch = self.epochs, verbose = self.verbose)
			score = self.nn.evaluate(self.X, self.y, verbose = self.verbose)
			print(score)
			#print(self.nn.score)

		def predict_ahead(self, n_ahead = 1):
			# Store predictions and predict iteratively
			self.predictions = np.zeros(n_ahead)

			for i in range(n_ahead):
				self.current_x = self.timeseries[-self.lag:]
				self.current_x = self.current_x.reshape((1, self.lag))
				self.current_x = self.scaler.transform(self.current_x)
				self.next_pred = self.nn.predict(self.current_x)
				self.predictions[i] = np.exp(self.next_pred[0, 0])
				self.timeseries = np.concatenate((self.timeseries, np.exp(self.next_pred[0,:])), axis = 0)

			return self.predictions


	hidlay = [10, 10, 30 ]
	actifunc = ['sigmoid', 'tanh', 'relu' ]
	nolag = 9
	noepoch = 10000
	noweek = 42
	#print('pd.read_csv(fname)')
	#print(pd.read_csv(fname))
	#print('pd.read_csv(fname)["x"]')
	#print(pd.read_csv(fname)["x"])
	#print('pd.read_csv(fname).iloc[:,1]')
	#print(pd.read_csv(fname).iloc[:,1])
	##cut_time_series = np.array(pd.read_csv(fname)["x"][:164])

	temp_tot_df = pd.read_csv(fname)
	temp_tot_df.columns = ["dttm","sale_count"]
	temp_df = temp_tot_df["sale_count"]

	#print(temp_df)
	temp_df[temp_df<1]=0.1
	time_series = np.array(temp_df)
	neural_net = TimeSeriesNnet(hidden_layers = hidlay, activation_functions = actifunc)
	neural_net.fit(time_series, lag = nolag, epochs = noepoch)
	neural_net.predict_ahead(n_ahead = noweek)

	end_date = temp_tot_df.iloc[temp_tot_df.shape[0] - 1]["dttm"]
	start_date = temp_tot_df.iloc[0]["dttm"]
	week_list = pd.date_range(start=end_date, freq='W', periods = noweek + 1)
	week_list = week_list[1:]
	#print(week_list)

	all_datapoints = list(neural_net.timeseries)[temp_df.shape[0]:]
	#print(list(neural_net.timeseries))
	#rint(len(list(neural_net.timeseries)))
	#print(temp_df.shape[0])
	#print(all_datapoints)

	#print(len(list(neural_net.timeseries)) - temp_df.shape[0])
	#print(len(neural_net.timeseries))
	#print(len(time_series))
	#plt.plot(range(len(cut_time_series)), cut_time_series, label = 'Original', linewidth = 3, ls = '-.')
	plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions & Forecast', linewidth=1)
	plt.plot(range(len(time_series)), time_series, '-g',  label='Original series', ls = '-.')
	figtit = "Graph for item-id : " + fname[:-4]
	figname = fname[:-4] + "_" + str(hidlay[0]) + actifunc[0][0] + str(hidlay[1]) + actifunc[1][0] + str(hidlay[2]) + actifunc[2][0] + "_" + str(noepoch) + "epch_" + str(nolag) + "lag.png"  
	plt.title(figtit)
	plt.xlabel("time")
	plt.ylabel("sales count")
	plt.legend()
	##plt.savefig(figname)
	plt.show()

	fin_df = pd.DataFrame(columns = ["date","silfra_forecast"], index=[i for i in range(noweek) ])
	fin_df["date"] = week_list
	fin_df["silfra_forecast"] = all_datapoints
	fin_df.silfra_forecast = fin_df.silfra_forecast.round()
	fin_df['silfra_forecast'] = fin_df['silfra_forecast'].astype(int)

	#print(fin_df)

	newfilename = fname[:-4] + "_silfra_forecast.csv"
	fin_df.to_csv(newfilename, sep=",", index=False)

