import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

dfRe = pd.read_csv('./all/Reactive-1.csv')
dfCobra = pd.read_csv('./all/CobRA-1.csv')
dfMy = pd.read_csv('./all/New-1.csv')

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
    dimmerSeries = brownout['vecvalue'].array[0].split(' ')
    serverNumSeries = serverNum['vecvalue'].array[0].split(' ')
    avgThroughputSeries = avgThroughtput['vecvalue'].array[0].split(' ')
    
    dimmerSeries = dimmerSeries[1:]
    serverNumSeries = serverNumSeries[1:]
    basicMedianResponseTimeSeries = basicMedianResponseTime['vecvalue'].array[0].split(' ')
    optMedianResponseTimeSeries = optMedianResponseTime['vecvalue'].array[0].split(' ')
    timeoutRateSeries = timeoutRate['vecvalue'].array[0].split(' ')

    tlen = len(dimmerSeries)

    accUtility = 0
    utilitySeries = []
    dDimmerSeries = []
    dServerNumSeries = []

    for i in range(tlen):
        avgThroughputSeries[i] = float(avgThroughputSeries[i]) 
        dimmerSeries[i] = 1 - float(dimmerSeries[i])    # change brownout value to dimmer value
        serverNumSeries[i] = float(serverNumSeries[i])
        timeoutRateSeries[i] = float(timeoutRateSeries[i])
        avgResponseTimeSeries[i] = float(avgResponseTimeSeries[i])

        if(i < tlen - 1):
            dDimmerSeries.append(abs(dimmerSeries[i] - 1 + float(dimmerSeries[i+1])))
            dServerNumSeries.append(abs(serverNumSeries[i] - float(serverNumSeries[i+1])))
        
        if(avgThroughputSeries[i] != 0):
            avgThroughputSeries[i] = 1 / avgThroughputSeries[i]

            revenue = (1 - timeoutRateSeries[i]) * avgThroughputSeries[i] * (1 * (1 - dimmerSeries[i]) + 1.5 * dimmerSeries[i]) - 0.5 * timeoutRateSeries[i] * avgThroughputSeries[i]
            cost = 5 * (3 - serverNumSeries[i])
            accUtility = accUtility + revenue + cost   
            utilitySeries.append(revenue + cost)

    #print("total utility = " + str(accUtility))        
    return accUtility, avgThroughputSeries, dimmerSeries, serverNumSeries, timeoutRateSeries, avgResponseTimeSeries, utilitySeries, dDimmerSeries, dServerNumSeries

def showStat(case, dDimmerSeries, dServerNumSeries, timeoutRateSeries, accUtility):
    dDimmerAvg = np.mean(dDimmerSeries)
    dServerNumAvg = np.mean(dServerNumSeries)
    timeoutRateSeries = timeoutRateSeries[5:]
    minY = min(timeoutRateSeries)
    maxY = max(timeoutRateSeries)
    devY = np.std(timeoutRateSeries)

    print(case + ' ' + str(dDimmerAvg) + ' ' + str(dServerNumAvg) + ' ' + str(minY) + ' ' + str(maxY)
        + ' ' + str(devY) + ' ' + str(accUtility))

accUtilityRe, avgThroughputSeriesRe, dimmerSeriesRe, serverNumSeriesRe, timeoutRateSeriesRe, avgResponseTimeSeriesRe, utilitySeriesRe, dDimmerSeriesRe, dServerNumSeriesRe = getData(dfRe)
accUtilityCobra, avgThroughputSeriesCobra, dimmerSeriesCobra, serverNumSeriesCobra, timeoutRateSeriesCobra, avgResponseTimeSeriesCobra, utilitySeriesCobra, dDimmerSeriesCobra, dServerNumSeriesCobra = getData(dfCobra)
accUtilityMy, avgThroughputSeriesMy, dimmerSeriesMy, serverNumSeriesMy, timeoutRateSeriesMy, avgResponseTimeSeriesMy, utilitySeriesMy, dDimmerSeriesMy, dServerNumSeriesMy = getData(dfMy)

tlen = len(dimmerSeriesRe)

fig,axarr = plt.subplots(6,1)  
fig.set_size_inches(6.4, 9.6)
plt.subplots_adjust(hspace=1.5,right=0.75)

for i in range(6):
    axarr[i].set_xlabel('t') 
    axarr[i].set_xlim(0,tlen) 

axarr[0].set_title('avgThroughput (ClarkNet)')
axarr[0].set_ylabel('avgThroughput')                          
axarr[0].set_ylim(0,max(avgThroughputSeriesRe)) 
axarr[0].plot(range(tlen),avgThroughputSeriesRe,linestyle='--',alpha=0.5,color='r')    #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[1].set_title('dimmer')
axarr[1].set_ylabel('dimmer')                          
axarr[1].set_ylim(0,1.1) 
y_major_locator = MultipleLocator(0.5)
axarr[1].yaxis.set_major_locator(y_major_locator)
axarr[1].plot(range(tlen),dimmerSeriesRe,linestyle='--',alpha=0.5,color='r') 
axarr[1].plot(range(tlen),dimmerSeriesCobra,linestyle='--',alpha=0.5,color='g') 
axarr[1].plot(range(tlen),dimmerSeriesMy,linestyle='--',alpha=0.5,color='b')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[2].set_title('server')
axarr[2].set_ylabel('server')                          
axarr[2].set_ylim(0,3.1) 
y_major_locator = MultipleLocator(1)
axarr[2].yaxis.set_major_locator(y_major_locator)
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

axarr[1].legend(labels=['Reactive','CobRA','New'],loc='best',bbox_to_anchor=(1.05,1.0))
plt.show()

showStat('Reactive', dDimmerSeriesRe, dServerNumSeriesRe, timeoutRateSeriesRe, accUtilityRe)
showStat('CobRA', dDimmerSeriesCobra, dServerNumSeriesCobra, timeoutRateSeriesCobra, accUtilityCobra)
showStat('New', dDimmerSeriesMy, dServerNumSeriesMy, timeoutRateSeriesMy, accUtilityMy)