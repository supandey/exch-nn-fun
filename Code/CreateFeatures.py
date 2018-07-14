import sys
if "../Utils" not in sys.path: sys.path.append("../Utils") # Desperation

import numpy as np
import pandas as pd
import pickle

from filterData import filterDf, createSimpleCleanSamplesDf, createSimpleCleanSamplesManyInstrumentsDf


# load data

dfEsp = pd.read_pickle("../Data/esp06062018.pkl")
#dfBte = pd.read_pickle("Data/bte62018.pkl")
#dfCme = pd.read_pickle("Data/cme62018.pkl")

def createSimpleFeatures(df):
    
    df = filterDf(df)
    dataFeatures, truthData, dfSample = createSimpleCleanSamplesDf(
            df, 
            instForSignal = 4, 
            n_samplesSignal = 200, 
            n_samplesNoise = 200, 
            sampleFreq = '10L', 
            truthLookAhead = 10, 
            createDfTemp = True,
            resampleType = 'sum')
    
    pickle.dump( (dataFeatures,truthData), open( "../Data/CleanSimpleFeatures_SUM_200_06062018.pkl", "wb" ) )
    return dataFeatures, truthData, dfSample

def createSimpleFeaturesMany(df):
    df = filterDf(df)
    dataFeatures, truthData, dfSample = createSimpleCleanSamplesManyInstrumentsDf(
            df, 
            instForSignal = 4, 
            n_samplesSignal = 200, 
            n_samplesNoise = 200, 
            sampleFreq = '10L', 
            truthLookAhead = 10, 
            createDfTemp = True,
            resampleType = 'sum',
            otherFeatureInsts = [2, 3, 5])  # 5Y, 7Y and 30Y
    
    pickle.dump( (dataFeatures,truthData), open( "../Data/CleanSimpleFeaturesManyInstruments_SUM_200_06062018.pkl", "wb" ) )
    return dataFeatures, truthData, dfSample

dataFeatures, truthData, dfSample = createSimpleFeatures(dfEsp)
#dataFeaturesMany, truthDataMany, dfSampleMany = createSimpleFeaturesMany(dfEsp)
 