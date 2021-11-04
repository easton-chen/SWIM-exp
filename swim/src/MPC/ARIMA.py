import pandas
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math
import numpy as np


#traceName = './traces/clarknet-http-105m-l70.delta'
traceName = './traces/wc_day53-r0-105m-l70.delta'
trace = open(traceName,'r')
curTime = 0
curNum = 0
reqList = []
reqs = trace.readlines()
interval = 60
for req in reqs:
    time = float(req)
    if(curTime + time < interval):
        curTime += time
        curNum += 1
    else:
        curTime = time
        reqList.append(curNum / 1)
        curNum = 1


def predict(model, history):
	value = model.arparams[0] 
	for i in range(1, len(model.arparams)):
		value += model.arparams[i] * history[-i]
	for i in range(1, len(model.maparams)):
		value += model.maparams[i] * history[-i]
	value = history[-1] + value
	return value

#series = read_csv('clarknet-data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
#X = series.values
X = reqList

size = int(len(X) * 0.1)
#train, test = X[0:size], X[size:len(X)]
train = X
test = X[size:len(X)]
history_data = X[0:size]
global history
history = [x for x in history_data]
predictions = list()
model = ARIMA(train, order=(2,1,0))
global model_fit
model_fit = model.fit()
#output = model_fit.forecast(steps=len(test))
#print(len(model_fit.maparams))

for t in range(len(test)):
	yhat = predict(model_fit, history)
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


MAPE = np.mean(np.abs((np.array(predictions) - np.array(test)) / np.array(test))) * 100
print('Test MAPE: %.3f' % MAPE)

# plot
pyplot.plot(test, label='true value')
pyplot.plot(predictions, color='orange', label='prediction')
pyplot.legend(loc='lower left')
pyplot.title('WorldCup\'98')
pyplot.show()
