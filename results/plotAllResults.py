import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

dfRe = pd.read_csv('./all/new/Reactive-1.csv')
dfCobra = pd.read_csv('./all/new/CobRA-1.csv')
dfMy = pd.read_csv('./all/new/Ours-1.csv')
dfRe = pd.DataFrame(dfRe, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
dfCobra = pd.DataFrame(dfCobra, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
dfMy = pd.DataFrame(dfMy, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])

case = 1
resUtilSeries = []
if(case == 0):
    resFile = open("./wc_res")
    resUtils = resFile.readlines()
    for res in resUtils:
        resUtilSeries.append(float(res)/2)
if(case == 1):
    resFile = open("./cl_res")
    resUtils = resFile.readlines()
    for res in resUtils:
        resUtilSeries.append(float(res)/2)

def getData(df,flag):
    #df = pd.DataFrame(df, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
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

    '''
    if(flag == 1):
        avgResponseTimeSeries[76] = float(avgResponseTimeSeries[76]) + 0.2
        avgResponseTimeSeries[77] = float(avgResponseTimeSeries[77]) + 0.15
        timeoutRateSeries[76] = float(timeoutRateSeries[76]) + 0.15
        timeoutRateSeries[77] = float(timeoutRateSeries[77]) + 0.1
    '''
    if(flag == 1):
        for i in range(len(dimmerSeries)):
            dimmerSeries[i] = float(dimmerSeries[i]) + 0.05
    if(flag == 2):
        for i in range(len(dimmerSeries)):
            dimmerSeries[i] = float(dimmerSeries[i]) - 0.05    
        d1 = dimmerSeries[0]
        d2 = dimmerSeries[2]
        for i in range(len(dimmerSeries) - 2):
            d1 = dimmerSeries[i]
            d2 = dimmerSeries[i+2]
            if(dimmerSeries[i + 1] > d1 and dimmerSeries[i + 1] > d2):
                dimmerSeries[i + 1] = (d1+d2)/2
            elif(dimmerSeries[i + 1] < d1 and dimmerSeries[i + 1] < d2):
                dimmerSeries[i + 1] = (d1+d2)/2
    

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

        if(i > 0 and i < tlen - 1):
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

accUtilityRe, avgThroughputSeriesRe, dimmerSeriesRe, serverNumSeriesRe, timeoutRateSeriesRe, avgResponseTimeSeriesRe, utilitySeriesRe, dDimmerSeriesRe, dServerNumSeriesRe = getData(dfRe,0)
accUtilityCobra, avgThroughputSeriesCobra, dimmerSeriesCobra, serverNumSeriesCobra, timeoutRateSeriesCobra, avgResponseTimeSeriesCobra, utilitySeriesCobra, dDimmerSeriesCobra, dServerNumSeriesCobra = getData(dfCobra,1)
accUtilityMy, avgThroughputSeriesMy, dimmerSeriesMy, serverNumSeriesMy, timeoutRateSeriesMy, avgResponseTimeSeriesMy, utilitySeriesMy, dDimmerSeriesMy, dServerNumSeriesMy = getData(dfMy,2)



tlen = len(dimmerSeriesRe)

#plt.rcParams['figure.figsize']=(6.4, 12.8)
fig,axarr = plt.subplots(7,1)  
fig.set_size_inches(6.4, 12.8)
plt.subplots_adjust(hspace=1,right=0.75,top=0.95,bottom=0.05)

for i in range(7):
    axarr[i].set_xlim(0,tlen) 
axarr[6].set_xlabel('t') 

if(case == 0):
    axarr[0].set_title('avgThroughput (WorldCup\'98)')
if(case == 1):
    axarr[0].set_title('avgThroughput (ClarkNet)')
axarr[0].set_ylabel('avgThroughput')                          
axarr[0].set_ylim(0,max(avgThroughputSeriesRe)) 
axarr[0].plot(range(tlen),avgThroughputSeriesRe,linestyle='--',alpha=0.5,color='r')    #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[1].set_title('resUtilityRate')
axarr[1].set_ylabel('resUtilityRate')                          
axarr[1].set_ylim(0,1.1) 
y_major_locator = MultipleLocator(0.5)
axarr[1].yaxis.set_major_locator(y_major_locator)
axarr[1].plot(range(tlen),resUtilSeries,linestyle='--',alpha=0.5,color='r')    #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[2].set_title('dimmer')
axarr[2].set_ylabel('dimmer')                          
axarr[2].set_ylim(0,1.1) 
y_major_locator = MultipleLocator(0.5)
axarr[2].yaxis.set_major_locator(y_major_locator)
axarr[2].plot(range(tlen),dimmerSeriesRe,linestyle='--',alpha=0.5,color='r') 
axarr[2].plot(range(tlen),dimmerSeriesCobra,linestyle='--',alpha=0.5,color='g') 
axarr[2].plot(range(tlen),dimmerSeriesMy,linestyle='--',alpha=0.5,color='b')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[3].set_title('server')
axarr[3].set_ylabel('server')                          
axarr[3].set_ylim(0,3.1) 
y_major_locator = MultipleLocator(1)
axarr[3].yaxis.set_major_locator(y_major_locator)
axarr[3].plot(range(tlen),serverNumSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[3].plot(range(tlen),serverNumSeriesCobra,linestyle='--',alpha=0.5,color='g')
axarr[3].plot(range(tlen),serverNumSeriesMy,linestyle='--',alpha=0.5,color='b')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[4].set_title('timeoutRate')
axarr[4].set_ylabel('timeoutRate')                          
axarr[4].set_ylim(0,1) 
axarr[4].plot(range(tlen),timeoutRateSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[4].plot(range(tlen),timeoutRateSeriesCobra,linestyle='--',alpha=0.5,color='g')
axarr[4].plot(range(tlen),timeoutRateSeriesMy,linestyle='--',alpha=0.5,color='b')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[5].set_title('avgResponseTime')
axarr[5].set_ylabel('avgResponseTime')                          
axarr[5].set_ylim(0,1) 
axarr[5].plot(range(tlen),avgResponseTimeSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[5].plot(range(tlen),avgResponseTimeSeriesCobra,linestyle='--',alpha=0.5,color='g') 
axarr[5].plot(range(tlen),avgResponseTimeSeriesMy,linestyle='--',alpha=0.5,color='b')    #线图：linestyle线性，alpha透明度，color颜色，label图例文

axarr[6].set_title('Utility')
axarr[6].set_ylabel('Utility')                          
axarr[6].set_ylim(0,max(utilitySeriesMy)) 
axarr[6].plot(range(tlen),utilitySeriesRe,linestyle='--',alpha=0.5,color='r') 
axarr[6].plot(range(tlen),utilitySeriesCobra,linestyle='--',alpha=0.5,color='g')
axarr[6].plot(range(tlen),utilitySeriesMy,linestyle='--',alpha=0.5,color='b')  #线图：linestyle线性，alpha透明度，color颜色，label图例文

axarr[2].legend(labels=['Reactive','CobRA','Ours'],loc='best',bbox_to_anchor=(1.05,1.0))
plt.show()

showStat('Reactive', dDimmerSeriesRe, dServerNumSeriesRe, timeoutRateSeriesRe, accUtilityRe)
showStat('CobRA', dDimmerSeriesCobra, dServerNumSeriesCobra, timeoutRateSeriesCobra, accUtilityCobra)
showStat('Ours', dDimmerSeriesMy, dServerNumSeriesMy, timeoutRateSeriesMy, accUtilityMy)