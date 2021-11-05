import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfRe = pd.read_csv('./Reactive-1.csv')
dfCobra = pd.read_csv('./Cobra-1.csv')
dfMy = pd.read_csv('./My-1.csv')

def getData(df):
    df = pd.DataFrame(df, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
    brownout = df.loc[df['name'] == 'brownoutFactor:vector']
    serverNum = df.loc[df['name'] == 'activeServers:vector']
    avgResponseTime = df.loc[df['name'] == 'avgResponseTime:vector']
    avgThroughtput = df.loc[df['name'] == 'measuredInterarrivalAvg:vector']
    basicMedianResponseTime = df.loc[df['name'] == 'basicMedianResponseTime:vector']
    optMedianResponseTime = df.loc[df['name'] == 'optMedianResponseTime:vector']
    timeoutRate = df.loc[df['name'] == 'timeoutRate:vector']

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

            revenue = (1 - timeoutRateSeries[i]) * avgThroughputSeries[i] * (1.5 * (1 - brownoutSeries[i]) + 1 * brownoutSeries[i]) - 0.5 * timeoutRateSeries[i] * avgThroughputSeries[i]
            cost = 5 * (3 - serverNumSeries[i])
            accUtility = accUtility + revenue + cost   
            utilitySeries.append(revenue + cost)

    #print("total utility = " + str(accUtility))        
    return accUtility, avgThroughputSeries, brownoutSeries, serverNumSeries, timeoutRateSeries, avgResponseTimeSeries, utilitySeries

accUtilityRe, avgThroughputSeriesRe, brownoutSeriesRe, serverNumSeriesRe, timeoutRateSeriesRe, avgResponseTimeSeriesRe, utilitySeriesRe = getData(dfRe)
accUtilityCobra, avgThroughputSeriesCobra, brownoutSeriesCobra, serverNumSeriesCobra, timeoutRateSeriesCobra, avgResponseTimeSeriesCobra, utilitySeriesCobra = getData(dfCobra)
accUtilityMy, avgThroughputSeriesMy, brownoutSeriesMy, serverNumSeriesMy, timeoutRateSeriesMy, avgResponseTimeSeriesMy, utilitySeriesMy = getData(dfMy)

tlen = len(brownoutSeriesRe)

fig,axarr = plt.subplots(6,1)  
fig.set_size_inches(6.4, 9.6)
plt.subplots_adjust(hspace=1,right=0.75)

for i in range(6):
    axarr[i].set_xlabel('t') 
    axarr[i].set_xlim(0,tlen) 

axarr[0].set_title('avgThroughput')
axarr[0].set_ylabel('avgThroughput')                          
axarr[0].set_ylim(0,max(avgThroughputSeriesRe)) 
axarr[0].plot(range(tlen),avgThroughputSeriesRe,linestyle='--',alpha=0.5,color='r')    #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[1].set_title('brownout')
axarr[1].set_ylabel('brownout')                          
axarr[1].set_ylim(0,1) 
axarr[1].plot(range(tlen),brownoutSeriesRe,linestyle='--',alpha=0.5,color='r') 
axarr[1].plot(range(tlen),brownoutSeriesCobra,linestyle='--',alpha=0.5,color='g') 
axarr[1].plot(range(tlen),brownoutSeriesMy,linestyle='--',alpha=0.5,color='b')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[2].set_title('server')
axarr[2].set_ylabel('server')                          
axarr[2].set_ylim(0,3) 
axarr[2].plot(range(tlen),serverNumSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[2].plot(range(tlen),serverNumSeriesCobra,linestyle='--',alpha=0.5,color='g')
axarr[2].plot(range(tlen),serverNumSeriesMy,linestyle='--',alpha=0.5,color='b')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[3].set_title('timeoutRate')
axarr[3].set_ylabel('timeoutRate')                          
axarr[3].set_ylim(0,1) 
axarr[3].plot(range(tlen),timeoutRateSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[3].plot(range(tlen),timeoutRateSeriesCobra,linestyle='--',alpha=0.5,color='g')
axarr[3].plot(range(tlen),timeoutRateSeriesMy,linestyle='--',alpha=0.5,color='b')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[4].set_title('avgResponseTime')
axarr[4].set_ylabel('avgResponseTime')                          
axarr[4].set_ylim(0,1) 
axarr[4].plot(range(tlen),avgResponseTimeSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[4].plot(range(tlen),avgResponseTimeSeriesCobra,linestyle='--',alpha=0.5,color='g') 
axarr[4].plot(range(tlen),avgResponseTimeSeriesMy,linestyle='--',alpha=0.5,color='b')    #线图：linestyle线性，alpha透明度，color颜色，label图例文

axarr[5].set_title('Utility')
axarr[5].set_ylabel('Utility')                          
axarr[5].set_ylim(0,max(utilitySeriesMy)) 
axarr[5].plot(range(tlen),utilitySeriesRe,linestyle='--',alpha=0.5,color='r') 
axarr[5].plot(range(tlen),utilitySeriesCobra,linestyle='--',alpha=0.5,color='g')
axarr[5].plot(range(tlen),utilitySeriesMy,linestyle='--',alpha=0.5,color='b')  #线图：linestyle线性，alpha透明度，color颜色，label图例文

axarr[3].legend(labels=['Reactive','CobRA','New'],loc='best',bbox_to_anchor=(1.05,1.0))
plt.show()

