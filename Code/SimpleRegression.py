import sys
if "../Utils" not in sys.path: sys.path.append("../Utils") # Desperation

import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import dates
import pandas as pd

from plotHelpers import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

# load data

#(X,y) = pd.read_pickle("../Data/CleanSimpleFeatures_LAST_200_06062018.pkl")
(X,y) = pd.read_pickle("../Data/CleanSimpleFeatures_SUM_200_06062018.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
 
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#-----------------------------------------------------------------
# Perceptron
#-----------------------------------------------------------------

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

nTst = len(y_test)
nTot = len(y)

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(nTot-nTst, nTot))
plt.xlabel('pcPercent (10ms) [standardized]')
plt.ylabel('pcPercent (20ms) [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
plt.show()

#-----------------------------------------------------------------
# LogisticRegression
#-----------------------------------------------------------------

#lr = LogisticRegression(C=1000.0, random_state=0)
#lr.fit(X_train_std, y_train)
#
#plot_decision_regions(X_combined_std, y_combined,
#                      classifier=lr, test_idx=range(105, 150))
#plt.xlabel('petal length [standardized]')
#plt.ylabel('petal width [standardized]')
#plt.legend(loc='upper left')
#plt.tight_layout()
## plt.savefig('./figures/logistic_regression.png', dpi=300)
#plt.show()