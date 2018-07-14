import numpy as np
import pandas as pd

from helpers import flatten

#from IPython.core.debugger import Pdb

def filterDf(df):
    # Add data
    df['pcPercent'] = (df['pA'] - df['pc'])/(df['pA'] - df['pB'])
    
    # Remove some data which we will not use for now
    df = df.drop(['eventNum', 'impQB', 'impQA', 'timeToExch', 'timeToLvlUpdt', 'workupState', 'workupPrice', 'ytrdPrice', 'ytrdOpenQty', 'ytrdHiddenQty', 'numInPacket', 'seqNumInst', 'seqNumUpdate', 'numLineInEvt', 'numOrdInTrade'], axis=1)

    return df

def createSimpleCleanSamplesDf(
        df, 
        instForSignal = 4, 
        n_samplesSignal = 20, 
        n_samplesNoise = 20, 
        sampleFreq = '10L', 
        truthLookAhead = 10, 
        createDfTemp = False,
        resampleType = 'last'):
    ''' 
    Only use data from instForJump for features. Only 2 features so can view results easily. Feature is pcPercent 10ms and 20ms in the past.
    
    Sampled at some frequency (ie 10ms) in the past. 
    
    Inputs
       instForSignal:   index intrument we are intrested in
       n_samplesSignal: samples where truth is -1 or 1
       n_samplesNoise:  samples where truth is 0
       sampleFreq:      sample data on fixed grid. Units are below.
       truthLookAhead:  how many sampleFreq points to look for signal. ie if sampleFreq=10L and truthLookAhead = 10 we look 100ms ahead for a change in price.
       createDfTemp:    create debug output (slow)
       resampleType:    last, mean, sum (might capture many updates)
    
    Outputs 
        dataFeatures: [n_samples, n_features].    features = pcPercent 10 and 20 ms in the past before truth event. 
        truthData:    [n_samples].   0,1,-1 for no change, change down, change up
        dfTemp:       subset of df we used to create features. For debugging.
    
    
    Assume if I miss an instrument when resampling I will see the change eventually.
    
    H   hourly frequency
    T   minutely frequency
    S   secondly frequency
    L   milliseconds
    U   microseconds
    
    TODO: 
         Check for only one change in signal area to avoid flickering?
         Should feature be mean or min/max rather than ffill?
        
        
    '''
    
    idxStart = []
    idxSeen = []
    dataFeatures = []
    truthData = []
    numNoJump = 0
    numJump = 0
    
    dfTemp = pd.DataFrame()
    dfSample = df[df['instNum'] == instForSignal]               # only look at data for instForSignal 
    dfSample = dfSample.resample(sampleFreq).last().ffill()
    
    dfSampleForOutput = pd.DataFrame()
    if resampleType == 'last':
        dfSampleForOutput = dfSample.resample(sampleFreq).last().ffill()
    elif resampleType == 'mean':
        dfSampleForOutput = dfSample.resample(sampleFreq).mean().ffill()
    elif resampleType == 'sum':
        dfSampleForOutput = dfSample.resample(sampleFreq).sum().ffill()
    
    np.random.seed(1001)
    indexRandom = np.random.randint(dfSample.shape[0]-truthLookAhead*10, size=500000)  #create a lot of candidates
    
    # Check for each indexRandom that we try to find a non-overlapping n_samplesSignal and n_samplesNoise
    
    npts = 2                                # number of time steps to look for features
    totNpts = npts + truthLookAhead         # number of time steps to look for signal
    
    for i in range(indexRandom.shape[0]):
        idx = indexRandom[i]
        
        dfFirstSegmentInst = dfSample.iloc[idx:idx+npts][['pB', 'pA']]        
        dfSecondSegmentInst = dfSample.iloc[idx+npts:idx+totNpts][['pB', 'pA']]
        
        #Pdb().set_trace()
            
        pBStart = dfFirstSegmentInst.iloc[0]['pB']
        pAStart = dfFirstSegmentInst.iloc[0]['pA']
        
        noChangeFirst = True
        if np.sum(np.diff(dfFirstSegmentInst['pB'])) > 1.e-8 or np.sum(np.diff(dfFirstSegmentInst['pA'])) > 1.e-8:
            noChangeFirst = False
            
        noChangeSecond = True
        if np.sum(np.diff(dfSecondSegmentInst['pB'])) > 1.e-8 or np.sum(np.diff(dfSecondSegmentInst['pA'])) > 1.e-8:
            noChangeSecond = False
            
        # Noise
        if (idxSeen.count(idx) == 0 and idxSeen.count(idx+totNpts-1) == 0 and   # no overlapping segments
            numNoJump < n_samplesNoise and 
            noChangeFirst == True and 
            noChangeSecond == True):
           
           numNoJump = numNoJump + 1 
           features = flatten(dfSampleForOutput.iloc[idx:idx+npts][['pcPercent']].values.tolist())
           dataFeatures.append(features)
           truthData.append(0)
           #Pdb().set_trace()
           idxStart.append(idx)
           for z in range(idx,idx+totNpts): idxSeen.append(z)
           if createDfTemp: dfTemp = dfTemp.append(dfSample.iloc[idx:idx+totNpts])
           
        # Signal
        if (idxSeen.count(idx) == 0 and idxSeen.count(idx+totNpts-1) == 0 and   # no overlapping segments
             numJump < n_samplesSignal and noChangeFirst == True and
             (pBStart < dfSecondSegmentInst['pB'].max() - 1.e-8 and pAStart < dfSecondSegmentInst['pA'].max() - 1.e-8) or 
             (pBStart > dfSecondSegmentInst['pB'].min() + 1.e-8 and pAStart > dfSecondSegmentInst['pA'].min() + 1.e-8)):
            
            numJump = numJump + 1 
            features = flatten(dfSampleForOutput.iloc[idx:idx+npts][['pcPercent']].values.tolist())
            dataFeatures.append(features)
            if pBStart < dfSecondSegmentInst['pB'].max() - 1.e-8: truthData.append(1)
            if pBStart > dfSecondSegmentInst['pB'].min() + 1.e-8: truthData.append(-1)
            idxStart.append(idx)
            for z in range(idx,idx+npts): idxSeen.append(z)
            if createDfTemp: dfTemp = dfTemp.append(dfSample.iloc[idx:idx+totNpts])
            print('numJump = {} nonJump = {}\n'.format(numJump, numNoJump))
                    
        if numJump >= n_samplesSignal and numNoJump >= n_samplesNoise:
            break
        
        if (i%10000 == 0):
            print('i: {}\n'.format(i))
    
    return dataFeatures, truthData, dfTemp

def createSimpleCleanSamplesManyInstrumentsDf(
        df, 
        instForSignal = 4, 
        n_samplesSignal = 20, 
        n_samplesNoise = 20, 
        sampleFreq = '10L', 
        truthLookAhead = 10, 
        createDfTemp = False,
        resampleType = 'last',
        otherFeatureInsts = [0 , 1, 2, 3, 5]):
    ''' 
    Only use data from all instruments for features. Only 2 features per instrument. Feature is pcPercent 10ms and 20ms in the past.
    
    Sampled at some frequency (ie 10ms) in the past. 
    
    Inputs
       instForSignal:       index intrument we are intrested in
       n_samplesSignal:     samples where truth is -1 or 1
       n_samplesNoise:      samples where truth is 0
       sampleFreq:          sample data on fixed grid. Units are below.
       truthLookAhead:      how many sampleFreq points to look for signal. ie if sampleFreq=10L and truthLookAhead = 10 we look 100ms ahead for a change in price.
       createDfTemp:        create debug output (slow)
       resampleType:        last, mean, sum (might capture many updates)
       otherFeatureInsts:   add same features fomr other instruments
    
    Outputs 
        dataFeatures: [n_samples, n_features].    features = pcPercent 10 and 20 ms in the past before truth event. 
        truthData:    [n_samples].   0,1,-1 for no change, change down, change up
        dfTemp:       subset of df we used to create features. For debugging.
    
    
    Assume if I miss an instrument when resampling I will see the change eventually.
    
    H   hourly frequency
    T   minutely frequency
    S   secondly frequency
    L   milliseconds
    U   microseconds
    
    TODO: 
         Check for only one change in signal area to avoid flickering?
         Should feature be mean or min/max rather than ffill?
        
        
    '''
    
    idxStart = []
    idxSeen = []
    dataFeatures = []
    truthData = []
    numNoJump = 0
    numJump = 0
    
    dfTemp = pd.DataFrame()
    dfSample = df[df['instNum'] == instForSignal]               # only look at data for instForSignal 
    dfSample = dfSample.resample(sampleFreq).last().ffill()
    
    dfSampleForOutput = pd.DataFrame()
    if resampleType == 'last':
        dfSampleForOutput = dfSample.resample(sampleFreq).last().ffill()
    elif resampleType == 'mean':
        dfSampleForOutput = dfSample.resample(sampleFreq).mean().ffill()
    elif resampleType == 'sum':
        dfSampleForOutput = dfSample.resample(sampleFreq).sum().ffill()
    
    # crude way to see every instrument at the same time snapshot
    dfMap = {}
    for instNum in otherFeatureInsts:
        dfInst = df[df['instNum'] == instNum]   
        
        if resampleType == 'last':
            dfInst = dfInst.resample(sampleFreq).last().ffill()
        elif resampleType == 'mean':
            dfInst = dfInst.resample(sampleFreq).mean().ffill()
        elif resampleType == 'sum':
            dfInst = dfInst.resample(sampleFreq).sum().ffill()
        
        dfMap[instNum] = dfInst
    
    np.random.seed(1001)
    indexRandom = np.random.randint(dfSample.shape[0]-truthLookAhead*10, size=500000)  #create a lot of candidates
    
    # Check for each indexRandom that we try to find a non-overlapping n_samplesSignal and n_samplesNoise
    
    npts = 2                                # number of time steps to look for features
    totNpts = npts + truthLookAhead         # number of time steps to look for signal
    
    for i in range(indexRandom.shape[0]):
        idx = indexRandom[i]
        
        dfFirstSegmentInst = dfSample.iloc[idx:idx+npts][['pB', 'pA']]        
        dfSecondSegmentInst = dfSample.iloc[idx+npts:idx+totNpts][['pB', 'pA']]
        
        #Pdb().set_trace()
            
        pBStart = dfFirstSegmentInst.iloc[0]['pB']
        pAStart = dfFirstSegmentInst.iloc[0]['pA']
        
        noChangeFirst = True
        if np.sum(np.diff(dfFirstSegmentInst['pB'])) > 1.e-8 or np.sum(np.diff(dfFirstSegmentInst['pA'])) > 1.e-8:
            noChangeFirst = False
            
        noChangeSecond = True
        if np.sum(np.diff(dfSecondSegmentInst['pB'])) > 1.e-8 or np.sum(np.diff(dfSecondSegmentInst['pA'])) > 1.e-8:
            noChangeSecond = False
            
        # Noise
        if (idxSeen.count(idx) == 0 and idxSeen.count(idx+totNpts-1) == 0 and   # no overlapping segments
            numNoJump < n_samplesNoise and 
            noChangeFirst == True and 
            noChangeSecond == True):
           
            numNoJump = numNoJump + 1 
          
            features = dfSampleForOutput.iloc[idx:idx+npts][['pcPercent']].values.tolist()
            for instNum, dfInstNum in dfMap.items():
                features.append(dfInstNum.iloc[idx:idx+npts][['pcPercent']].values.tolist())
           
            dataFeatures.append(flatten(features))
            
            truthData.append(0)
            #Pdb().set_trace()
            idxStart.append(idx)
            for z in range(idx,idx+totNpts): idxSeen.append(z)
            if createDfTemp: dfTemp = dfTemp.append(dfSample.iloc[idx:idx+totNpts])
           
        # Signal
        if (idxSeen.count(idx) == 0 and idxSeen.count(idx+totNpts-1) == 0 and   # no overlapping segments
            numJump < n_samplesSignal and noChangeFirst == True and
            (pBStart < dfSecondSegmentInst['pB'].max() - 1.e-8 and pAStart < dfSecondSegmentInst['pA'].max() - 1.e-8) or 
            (pBStart > dfSecondSegmentInst['pB'].min() + 1.e-8 and pAStart > dfSecondSegmentInst['pA'].min() + 1.e-8)):
                
            numJump = numJump + 1 
            
            features = dfSampleForOutput.iloc[idx:idx+npts][['pcPercent']].values.tolist()
            for instNum, dfInstNum in dfMap.items():
                features.append(dfInstNum.iloc[idx:idx+npts][['pcPercent']].values.tolist())
           
            dataFeatures.append(flatten(features))
            
            if pBStart < dfSecondSegmentInst['pB'].max() - 1.e-8: truthData.append(1)
            if pBStart > dfSecondSegmentInst['pB'].min() + 1.e-8: truthData.append(-1)
            idxStart.append(idx)
            for z in range(idx,idx+npts): idxSeen.append(z)
            if createDfTemp: dfTemp = dfTemp.append(dfSample.iloc[idx:idx+totNpts])
            print('numJump = {} nonJump = {}\n'.format(numJump, numNoJump))
                    
        if numJump >= n_samplesSignal and numNoJump >= n_samplesNoise:
            break
        
        if (i%10000 == 0):
            print('i: {}\n'.format(i))
    
    return dataFeatures, truthData, dfTemp

