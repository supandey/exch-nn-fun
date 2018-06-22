import numpy as np
import pandas as pd
#from IPython.core.debugger import Pdb

def filterDf(df):
    # Add data
    df['pcPercent'] = (df['pA'] - df['pc'])/(df['pA'] - df['pB'])
    
    # Remove some data which we will not use for now
    df = df.drop(['eventNum', 'impQB', 'impQA', 'timeToExch', 'timeToLvlUpdt', 'workupState', 'workupPrice', 'ytrdPrice', 'ytrdOpenQty', 'ytrdHiddenQty', 'numInPacket', 'seqNumInst', 'seqNumUpdate', 'numLineInEvt', 'numOrdInTrade'], axis=1)

    return df

def createCleanSamplesDf(df, freq = '10L', segmentSize = 20, numberSegments = 1000, instForJump = 4):
    # Output: dataFeatures: nxm matrix of features. m = numberSegments. n = inputs (qB,qA,...) sampled at freq in the past.
    # Output: truthData: m vector of 0/1.
    
    # Assume if I miss an instrument when resampling I will see the change eventually.
    
    # For default inputs my input features are 10ms*10=100ms before jump. Then I check next 100ms if a jump occurs
    # Check does bid/ask change.
    
    # In real life as soon as I see a change I switch to prediction mode. Note otehr intrumsnts can change
    # Change in furture to allow First to have changes
    
    
#    H   hourly frequency
#    T   minutely frequency
#    S   secondly frequency
#    L   milliseconds
#    U   microseconds
    
    idxSeen = []
    dataFeatures = []
    truthData = []
    numNoJump = 0
    numJump = 0
    
    dfSample = df.resample(freq).last().ffill()
    
    indexRandom = np.random.randint(dfSample.shape[0]-segmentSize*10, size=numberSegments*100000)  #create a lot of candidates
    
    
    # Check for each indexRandom that we have no price shift down/up on bid/ask in signal section. If none we add to randomFeatures
    szSeg = int(segmentSize/2)
    for i in range(indexRandom.shape[0]):
        idx = indexRandom[i]
        
        dfFirstSegment = df.iloc[idx:idx+szSeg][['instNum','pB', 'pA']]
        dfFirstSegmentInst = dfFirstSegment[dfFirstSegment['instNum'] == instForJump]
        
        dfSecondSegment = df.iloc[idx+szSeg:idx+segmentSize][['instNum','pB', 'pA']]
        dfSecondSegmentInst = dfSecondSegment[dfSecondSegment['instNum'] == instForJump]
        
        if (dfFirstSegmentInst.shape[0] > 0 and dfSecondSegmentInst.shape[0] > 0):  # make sure at least two changes in inst
            
            pBStart = dfFirstSegmentInst.iloc[0]['pB']
            pAStart = dfFirstSegmentInst.iloc[0]['pA']
            
            noChangeFirst = True
            if np.sum(np.diff(dfFirstSegmentInst['pB'])) > 1.e-8 or np.sum(np.diff(dfFirstSegmentInst['pA'])) > 1.e-8:
                noChangeFirst = False
                
            noChangeSecond = True
            if np.sum(np.diff(dfSecondSegmentInst['pB'])) > 1.e-8 or np.sum(np.diff(dfSecondSegmentInst['pA'])) > 1.e-8:
                noChangeSecond = False
                
            if (idxSeen.count(idx) == 0 and idxSeen.count(idx+segmentSize-1) == 0 and   # no overlapping segments
                numNoJump < numberSegments and 
                noChangeFirst == True and 
                noChangeSecond == True):
               numNoJump = numNoJump + 1 
               dataFeatures.append(df.iloc[idx:idx+szSeg][['instNum','qB', 'pB', 'pA', 'qA', 'pcPercent', 'numPartB', 'numPartA']].values.tolist())
               truthData.append(0)
               #Pdb().set_trace()
               #idxSeen.append([z for z in range(idx,idx+segmentSize)])
               for z in range(idx,idx+segmentSize): idxSeen.append(z)
               
            
            if (idxSeen.count(idx) == 0 and idxSeen.count(idx+segmentSize-1) == 0 and   # no overlapping segments
                numJump < numberSegments and noChangeFirst == True and
                (pBStart < dfSecondSegmentInst.iloc[0]['pB'].max() - 1.e-8 and pAStart < dfSecondSegmentInst.iloc[0]['pA'].max() - 1.e-8) or 
                (pBStart > dfSecondSegmentInst.iloc[0]['pB'].min() + 1.e-8 and pAStart > dfSecondSegmentInst.iloc[0]['pA'].min() + 1.e-8)):
                    numJump = numJump + 1 
                    dataFeatures.append(df.iloc[idx:idx+szSeg][['instNum','qB', 'pB', 'pA', 'qA', 'pcPercent', 'numPartB', 'numPartA']].values.tolist())
                    if pBStart < dfSecondSegmentInst.iloc[0]['pB'].max() - 1.e-8: truthData.append(1)
                    if pBStart > dfSecondSegmentInst.iloc[0]['pB'].min() + 1.e-8: truthData.append(-1)
                    #idxSeen.append([z for z in range(idx,idx+segmentSize)])
                    for z in range(idx,idx+segmentSize): idxSeen.append(z)
                    print('numJump = {}\n'.format(numJump))
                    
            if numJump > numberSegments and numNoJump > numberSegments:
                break
            
#            if ((numJump%10 == 0 and numJump < numberSegments and numJump > 1) or (numNoJump%10 == 0 and numNoJump < numberSegments)):
#                print('number: {} {} \n'.format(numJump, numNoJump))
            if (i%10000 == 0):
                print('i: {}\n'.format(i))
    
    return dataFeatures, truthData