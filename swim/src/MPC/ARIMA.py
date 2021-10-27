import pandas
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math


#traceName = './traces/clarknet-http-105m-l70.delta'
traceName = './traces/clarknet-http-105m-l70.delta'
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
model = ARIMA(train, order=(7,1,0))
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
error = mean_squared_error(test, predictions)
RMSE = math.sqrt(error)
print('Test RMSE: %.3f' % RMSE)
# plot
pyplot.plot(test, label='true value')
pyplot.plot(predictions, color='orange', label='prediction')
pyplot.legend(loc='lower left')
pyplot.title('ARIMA')
pyplot.show()

# save result
#data = pandas.DataFrame(data = predictions)
#data.to_csv('ARIMA_res.csv')

def fun():
    pvalue = predict(model_fit,history)
    return pvalue

for t in range(len(test)):
	yhat = fun()
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))    

error = mean_squared_error(test, predictions)
RMSE = math.sqrt(error)
print('Test RMSE: %.3f' % RMSE)
# plot
pyplot.plot(test, label='true value')
pyplot.plot(predictions, color='orange', label='prediction')
pyplot.legend(loc='lower left')
pyplot.title('ARIMA')
pyplot.show()
