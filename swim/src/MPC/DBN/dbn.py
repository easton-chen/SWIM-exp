from cProfile import label
#from importlib.metadata import MetadataPathFinder
from ipaddress import collapse_addresses
import pgmpy.models
import pgmpy.inference
import numpy as np
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork as DBN

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import random

# The main entry point for this module
def main():
    # Create a dynamic bayesian network
    model = pgmpy.models.DynamicBayesianNetwork()
    # Add nodes
    model.add_nodes_from(['Weather', 'Umbrella'])
    # Print nodes
    print('--- Nodes ---')
    print(model.nodes())
    # Add edges
    model.add_edges_from([(('Umbrella',0), ('Weather',0)),
                          (('Weather',0), ('Weather',1)),
                          (('Umbrella',0), ('Umbrella',1))])
    # Print edges
    print('--- Edges ---')
    print(model.edges())
    print()
    # Print parents
    print('--- Parents ---')
    print('Umbrella 0: {0}'.format(model.get_parents(('Umbrella', 0))))
    print('Weather 0: {0}'.format(model.get_parents(('Weather', 0))))
    print('Weather 1: {0}'.format(model.get_parents(('Weather', 1))))
    print('Umbrella 1: {0}'.format(model.get_parents(('Umbrella', 1))))
    print()

    # Add probabilities
    weather_cpd = pgmpy.factors.discrete.TabularCPD(('Weather', 0), 2, [[0.1, 0.8], 
                                                                        [0.9, 0.2]], 
                                                       evidence=[('Umbrella', 0)], 
                                                       evidence_card=[2])
    umbrella_cpd = pgmpy.factors.discrete.TabularCPD(('Umbrella', 1), 2, [[0.5, 0.5], 
                                                                          [0.5, 0.5]], 
                                                     evidence=[('Umbrella', 0)], 
                                                     evidence_card=[2])
    transition_cpd = pgmpy.factors.discrete.TabularCPD(('Weather', 1), 2, [[0.25, 0.9, 0.1, 0.25], 
                                                                           [0.75, 0.1, 0.9, 0.75]], 
                                                   evidence=[('Weather', 0), ('Umbrella', 1)], 
                                                   evidence_card=[2, 2])

    # Add conditional probability distributions (cpd:s)
    model.add_cpds(weather_cpd, umbrella_cpd, transition_cpd)
    # This method will automatically re-adjust the cpds and the edges added to the bayesian network.
    model.initialize_initial_state()
    # Check if the model is valid, throw an exception otherwise
    model.check_model()

    # Print probability distributions
    print('Probability distribution, P(Weather(0) | Umbrella(0)')
    print(weather_cpd)
    print()
    print('Probability distribution, P(Umbrella(1) | Umbrella(0)')
    print(umbrella_cpd)
    print()
    print('Probability distribution, P(Weather(1) | Umbrella(1), Weather(0)')
    print(transition_cpd)
    print()

    # Make inference
    map = {0: 'Sunny', 1: 'Rainy' }
    dbn_inf = pgmpy.inference.DBNInference(model)
    
    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):0, ('Weather', 0):0})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : Yes, Weather(0): Sunny): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()

    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):0, ('Weather', 0):1})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : Yes, Weather(0): Rainy): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()
    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):1, ('Weather', 0):0})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : No, Weather(0): Sunny): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()
    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):1, ('Weather', 0):1})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : No, Weather(0): Rainy): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()


def main1():
    model = DBN(
        [
            (("W", 0), ("R", 0)),
            (("W", 0), ("W", 1)),
            (("R", 0), ("R", 1)),
        ]
    )
    data = np.random.randint(low=0, high=2, size=(10, 6))
    #print(data)
    colnames = []
    for t in range(3):
        colnames.extend([("W", t), ("R", t)])
    df = pd.DataFrame(data, columns=colnames)
    #print(df)
    model.fit(df)
    print(model.get_cpds(node=("R",1)))

def scaleAndDiscrete(data):
    for i in range(len(data)):
        data[i] = (data[i] - data.min()) / (data.max() - data.min())
        data[i] = int(data[i] / 0.34)
    return data

def main2():
    data = pd.read_csv('C:/Users/LENOVO/Desktop/swim/dbn/PerfResult.csv',usecols=['Value','Timestamp','MetricId','Entity'])
    #print(data.head())
    data = data[data['Entity']=='php5_server']
    #data = data[data['Entity']=='php7_server']
    #data = data[data['Entity']=='media_server']
    CpuPercentData = np.array(data[data['MetricId']=='cpu.usage.average']['Value'])
    CpuHzData = np.array(data[data['MetricId']=='cpu.usagemhz.average']['Value'])
    MemData = np.array(data[data['MetricId']=='mem.usage.average']['Value'])
    DiskData = np.array(data[data['MetricId']=='disk.usage.average']['Value'])
    NetData = np.array(data[data['MetricId']=='net.usage.average']['Value'])

    #scale(CpuHzData)
    #print(CpuHzData)
    
    x = range(len(CpuPercentData))
    plt.plot(x,scaleAndDiscrete(NetData),label='net')
    #plt.plot(x,scaleAndDiscrete(CpuPercentData),label='cpu percent')
    #plt.plot(x,scaleAndDiscrete(CpuHzData),label='cpu hz')
    #plt.plot(x,scaleAndDiscrete(MemData),label='mem')
    plt.plot(x,scaleAndDiscrete(DiskData),label='disk')
    plt.legend()
    plt.show()
    
    model = DBN(
        [
            (("W", 0), ("R", 0)),
            (("W", 0), ("W", 1)),
            (("R", 0), ("R", 1)),
        ]
    )
    
    # generate dbn training data
    tdLength = len(NetData) - 1
    traindata = np.random.randint(low=0, high=2, size=(tdLength, 4))
    for i in range(tdLength):
        traindata[i][0] = NetData[i]
        traindata[i][1] = DiskData[i]
        traindata[i][2] = NetData[i+1]
        traindata[i][3] = DiskData[i+1]


    colnames = []
    for t in range(2):
        colnames.extend([("W", t), ("R", t)])
    df = pd.DataFrame(traindata, columns=colnames)
    #print(df)
    model.fit(df)
    print(model.get_cpds(node=("R",1)))


def generateResData():
    len = 210    
    dataFile = open('C:/Users/LENOVO/Desktop/swim/dbn/sourceFile','w')
    for i in range(len):
        res = random.randint(1,3) * 0.5
        dataFile.write(str(res) + '\n')
    dataFile.close()
    
if __name__ == "__main__": 
    generateResData()
    