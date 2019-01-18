import logging

from colorama import Fore 

from sklearn.datasets import fetch_20newsgroups

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, f1_score

logging.basicConfig(level = 'DEBUG')

# Imports -------------------------------------------
logging.debug(Fore.BLUE + 'Getting data' + Fore.RESET)

cats = ['alt.atheism','soc.religion.christian']

tr_data = fetch_20newsgroups(subset = 'train',categories = cats)
ts_data = fetch_20newsgroups(subset = 'test',categories = cats)

# Vectorization -------------------------------------

logging.debug(Fore.BLUE + 'Vectorizing' + Fore.RESET)
cv = CountVectorizer()
tr_mat = cv.fit_transform(tr_data.data)

# Model Selection -----------------------------------

logging.debug(Fore.BLUE + 'Performing parameter search' + Fore.RESET)
gs_mod = GridSearchCV(SVC(),n_jobs = 7, scoring = 'f1',
                      param_grid = {'C' : [0.001,0.01,0.1,1,10]})
gs_mod.fit(tr_mat,tr_data.target)
mod = gs_mod.best_estimator_

pl = Pipeline([('cv',cv),('svm',mod)])

# Evaluation ----------------------------------------
pred = pl.predict(ts_data.data)

f1 = f1_score(pred,ts_data.target)
logging.info(Fore.GREEN + str(f1) + Fore.RESET)

print(confusion_matrix(pred,ts_data.target))

# Serialization -------------------------------------

