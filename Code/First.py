import sys
if "../Utils" not in sys.path: sys.path.append("../Utils") # Desperation

import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import dates
import pandas as pd
import pickle

from scipy import stats, integrate
import seaborn as sns
sns.set_style("whitegrid")

from filterData import filterDf, createCleanSamplesDf


# load data

dfEsp = pd.read_pickle("../Data/esp06062018.pkl")
#dfBte = pd.read_pickle("Data/bte62018.pkl")
#dfCme = pd.read_pickle("Data/cme62018.pkl")

dfEsp = filterDf(dfEsp)
dataFeatures, truthData, idxStart, dfSample = createCleanSamplesDf(dfEsp, freq = '10L', segmentSize = 20, numberSegments = 500, instForJump = 4)
pickle.dump( (dataFeatures,truthData), open( "../Data/CleanFeatures_500.pkl", "wb" ) )
 