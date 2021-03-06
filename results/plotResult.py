import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./SWIM_SA/csv/Reactive-3.csv')
#df = pd.read_csv('./SWIM_TRAIN/csv/Train-0.csv')
df = pd.read_csv('./SWIM_TEST/csv/Test-3.csv')
df = pd.DataFrame(df, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
brownout = df.loc[df['name'] == 'brownoutFactor:vector']
serverNum = df.loc[df['name'] == 'activeServers:vector']
avgResponseTime = df.loc[df['name'] == 'avgResponseTime:vector']
avgThroughtput = df.loc[df['name'] == 'measuredInterarrivalAvg:vector']
basicMedianResponseTime = df.loc[df['name'] == 'basicMedianResponseTime:vector']
optMedianResponseTime = df.loc[df['name'] == 'optMedianResponseTime:vector']
timeoutRate = df.loc[df['name'] == 'timeoutRate:vector']
resUtil = df.loc[df['name'] == 'resUtil:vector']
#print(avgResponseTime['vecvalue'].array[0])

dataList = []
avgResponseTimeSeries = avgResponseTime['vecvalue'].array[0].split(' ')
brownoutSeries = brownout['vecvalue'].array[0].split(' ')
serverNumSeries = serverNum['vecvalue'].array[0].split(' ')
avgThroughputSeries = avgThroughtput['vecvalue'].array[0].split(' ')
brownoutSeries = brownoutSeries[1:]
serverNumSeries = serverNumSeries[1:]
basicMedianResponseTimeSeries = basicMedianResponseTime['vecvalue'].array[0].split(' ')
optMedianResponseTimeSeries = optMedianResponseTime['vecvalue'].array[0].split(' ')
timeoutRateSeries = timeoutRate['vecvalue'].array[0].split(' ')
resUtilSeries = resUtil['vecvalue'].array[0].split(' ')

#print('brownout' + '\t' + 'serverNum' + '\t' + 'avgThroughput' + 'avgResponseTime' + '\t' 
    #+ 'basicMedianResponseTime' + '\t' + 'optMedianResponseTime' + '\t' 
#    + 'timeoutRate')

tlen = len(brownoutSeries)

dimmerSeries = []
accUtility = 0
accRevenue = 0
accPenalty = 0
accCost = 0
utilitySeries = []

for i in range(tlen):
    avgThroughputSeries[i] = float(avgThroughputSeries[i]) 
    brownoutSeries[i] = float(brownoutSeries[i])
    dimmerSeries.append(1 - brownoutSeries[i])
    serverNumSeries[i] = float(serverNumSeries[i])
    timeoutRateSeries[i] = float(timeoutRateSeries[i])
    avgResponseTimeSeries[i] = float(avgResponseTimeSeries[i])
    resUtilSeries[i] = float(resUtilSeries[i])
    
    if(avgThroughputSeries[i] != 0):
        avgThroughputSeries[i] = 1 / avgThroughputSeries[i]
        #print(brownoutSeries[i] + '\t' + serverNumSeries[i] + '\t' + str(avgThroughputSeries[i]) + '\t' + avgResponseTimeSeries[i] + '\t' 
        #    + basicMedianResponseTimeSeries[i] + '\t' + optMedianResponseTimeSeries[i]+ '\t' 
        #    + timeoutRateSeries[i])

        revenue = (1 - timeoutRateSeries[i]) * avgThroughputSeries[i] * (1.5 * (1 - brownoutSeries[i]) + 1 * brownoutSeries[i]) 
        penalty = 0.5 * timeoutRateSeries[i] * avgThroughputSeries[i]
        cost = 5 * (3 - serverNumSeries[i])
        accUtility = accUtility + revenue + cost - penalty  
        accRevenue = accRevenue + revenue
        accPenalty = accPenalty + penalty
        accCost = accCost + cost 
        utilitySeries.append(revenue + cost - penalty)

print("total utility = " + str(accUtility))     
print("total revenue = " + str(accRevenue))    
print("total penalty = " + str(accPenalty))    
print("total Cost = " + str(accCost))       

fig,axarr = plt.subplots(6,1)  
fig.set_size_inches(6.4, 9.6)
plt.subplots_adjust(hspace=1)

for i in range(6):
    axarr[i].set_xlabel('t') 
    axarr[i].set_xlim(0,tlen) 

axarr[0].set_title('avgThroughput')
axarr[0].set_ylabel('avgThroughput')                          
axarr[0].set_ylim(0,max(avgThroughputSeries)) 
axarr[0].plot(range(tlen),avgThroughputSeries,linestyle='--',alpha=0.5,color='r')   #?????????linestyle?????????alpha????????????color?????????label????????????

axarr[1].set_title('resUtil')
axarr[1].set_ylabel('resUtil')                          
axarr[1].set_ylim(0,3) 
axarr[1].plot(range(tlen),resUtilSeries,linestyle='--',alpha=0.5,color='r')   #?????????linestyle?????????alpha????????????color?????????label?????????

axarr[2].set_title('dimmer')
axarr[2].set_ylabel('dimmer')                          
axarr[2].set_ylim(0,1) 
axarr[2].plot(range(tlen),dimmerSeries,linestyle='--',alpha=0.5,color='r')   #?????????linestyle?????????alpha????????????color?????????label????????????

axarr[3].set_title('server')
axarr[3].set_ylabel('server')                          
axarr[3].set_ylim(0,3) 
axarr[3].plot(range(tlen),serverNumSeries,linestyle='--',alpha=0.5,color='r')   #?????????linestyle?????????alpha????????????color?????????label????????????

axarr[4].set_title('timeoutRate')
axarr[4].set_ylabel('timeoutRate')                          
axarr[4].set_ylim(0,1) 
axarr[4].plot(range(tlen),timeoutRateSeries,linestyle='--',alpha=0.5,color='r')   #?????????linestyle?????????alpha????????????color?????????label????????????

axarr[5].set_title('Total Utility = ' + str(accUtility))
axarr[5].set_ylabel('Utility')                          
axarr[5].set_ylim(0,max(utilitySeries)) 
axarr[5].plot(range(tlen),utilitySeries,linestyle='--',alpha=0.5,color='r')   #?????????linestyle?????????alpha????????????color?????????label?????????

plt.show()

