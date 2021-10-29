import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv('./SWIM_SA/csv/Reactive-0.csv')
df = pd.read_csv('./SWIM_TEST/csv/Test-1.csv')
#df = pd.read_csv('./SWIM_TRAIN/csv/Train-0.csv')
df = pd.DataFrame(df, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
brownout = df.loc[df['name'] == 'brownoutFactor:vector']
serverNum = df.loc[df['name'] == 'activeServers:vector']
avgResponseTime = df.loc[df['name'] == 'avgResponseTime:vector']
avgThroughtput = df.loc[df['name'] == 'measuredInterarrivalAvg:vector']
basicMedianResponseTime = df.loc[df['name'] == 'basicMedianResponseTime:vector']
optMedianResponseTime = df.loc[df['name'] == 'optMedianResponseTime:vector']
timeoutRate = df.loc[df['name'] == 'timeoutRate:vector']
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

#print('brownout' + '\t' + 'serverNum' + '\t' + 'avgThroughput' + 'avgResponseTime' + '\t' 
    #+ 'basicMedianResponseTime' + '\t' + 'optMedianResponseTime' + '\t' 
#    + 'timeoutRate')

tlen = len(brownoutSeries)

accUtility = 0
utilitySeries = []

for i in range(tlen):
    avgThroughputSeries[i] = float(avgThroughputSeries[i]) 
    brownoutSeries[i] = float(brownoutSeries[i])
    serverNumSeries[i] = float(serverNumSeries[i])
    timeoutRateSeries[i] = float(timeoutRateSeries[i])
    avgResponseTimeSeries[i] = float(avgResponseTimeSeries[i])
    
    if(avgThroughputSeries[i] != 0):
        avgThroughputSeries[i] = 1 / avgThroughputSeries[i]
        #print(brownoutSeries[i] + '\t' + serverNumSeries[i] + '\t' + str(avgThroughputSeries[i]) + '\t' + avgResponseTimeSeries[i] + '\t' 
        #    + basicMedianResponseTimeSeries[i] + '\t' + optMedianResponseTimeSeries[i]+ '\t' 
        #    + timeoutRateSeries[i])

        revenue = (1 - timeoutRateSeries[i]) * avgThroughputSeries[i] * (1.5 * (1 - brownoutSeries[i]) + 1 * brownoutSeries[i]) - 0.5 * timeoutRateSeries[i] * avgThroughputSeries[i]
        cost = 5 * (3 - serverNumSeries[i])
        accUtility = accUtility + revenue + cost   
        utilitySeries.append(revenue + cost)

print("total utility = " + str(accUtility))        

fig,axarr = plt.subplots(6,1)  #开一个新窗口，并添加4个子图，返回子图数组
fig.set_size_inches(6.4, 9.6)
plt.subplots_adjust(hspace=1)

for i in range(6):
    axarr[i].set_xlabel('t') 
    axarr[i].set_xlim(0,tlen) 

axarr[0].set_title('avgThroughput')
axarr[0].set_ylabel('avgThroughput')                          
axarr[0].set_ylim(0,max(avgThroughputSeries)) 
axarr[0].plot(range(tlen),avgThroughputSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[1].set_title('brownout')
axarr[1].set_ylabel('brownout')                          
axarr[1].set_ylim(0,1) 
axarr[1].plot(range(tlen),brownoutSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[2].set_title('server')
axarr[2].set_ylabel('server')                          
axarr[2].set_ylim(0,3) 
axarr[2].plot(range(tlen),serverNumSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[3].set_title('timeoutRate')
axarr[3].set_ylabel('timeoutRate')                          
axarr[3].set_ylim(0,1) 
axarr[3].plot(range(tlen),timeoutRateSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[4].set_title('avgResponseTime')
axarr[4].set_ylabel('avgResponseTime')                          
axarr[4].set_ylim(0,1) 
axarr[4].plot(range(tlen),avgResponseTimeSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文

axarr[5].set_title('Total Utility = ' + str(accUtility))
axarr[5].set_ylabel('Utility')                          
axarr[5].set_ylim(0,max(utilitySeries)) 
axarr[5].plot(range(tlen),utilitySeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文

plt.show()

