import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import dates
import pandas as pd

dfEsp = pd.read_pickle("Data/esp06062018.pkl")


#----------------------------- --------------
#  Plots
#----------------------------- --------------

instNum = 4

maskInst4 = dfEsp['instNum'] == instNum
dfInst = dfEsp.loc[maskInst4,:]
maskTrd = abs(dfInst['vol'].pct_change() ) > 1.e-8
spread = dfInst['pA'] - dfInst['pB'] 
minIncr = np.min(spread[spread>1.e-8])

plt.figure(1)
plt.plot(dfInst.index, dfInst['pB'], label='pB', drawstyle='steps')
plt.plot(dfInst.index, dfInst['pA'], label='pA', drawstyle='steps')
plt.plot(dfInst.index, dfInst['pc'], label='pc', drawstyle='steps')       # NOTE pc is grabbed from ds
plt.plot(dfInst.index[maskTrd] .values, dfInst['pT'][maskTrd].values, 'ys', label='Trd')
plt.ylim((min(dfInst['pB']) - 3*minIncr, max(dfInst['pA']) + 3*minIncr))
plt.title('%s: %s' % (dfInst['Name'][0], dfInst.index[0].strftime('%Y-%m-%d')), fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Prices', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc='upper left')
plt.gcf().set_size_inches(15, 8)
ax = plt.gca()
#ax.get_yaxis().get_major_ formatter().set_scientific( False)
ax.get_yaxis().get_major_formatter().set_useOffset( False)
plt.grid()
plt.show()