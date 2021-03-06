
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import math

divide = math.floor(len(df2)*0.8)
target = 'is_duplicate'
nb_train = df2[:int(divide)]
nb_test = df2[int(divide):]

model = GaussianNB()
predictions = model.fit(nb_train[features], nb_train[target]).predict(nb_test[features])


metrics.accuracy_score(nb_test[target], predictions)


#Prediction accuracy: 67%

