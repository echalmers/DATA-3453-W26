import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

x_variable = 'sepal length (cm)'
y_variable = 'sepal width (cm)'

# load the "iris" dataset
df = pd.read_csv('./data/iris.csv', index_col=None)
# convert the class into numbers this time
df['class'] = df['class'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# we're only going to use a subset of the features for now
df = df[[x_variable, y_variable, 'class']]

# fit a model to the data
# try the following models:
# LogisticRegression()
# KNeighborsClassifier(n_neighbors=your_choice)
# SVC(gamma=your_choice)
# DecisionTreeClassifier(max_depth=3)
model = 
# YOUR CODE HERE

# create a grid of test points covering the whole space
x_min, x_max = df[x_variable].min() - 1, df[x_variable].max() + 1
y_min, y_max = df[y_variable].min() - 1, df[y_variable].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# use the model to make predictions at every point
preds = model.predict(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))))
preds = preds.reshape(xx.shape)

# plot the predictions at every point, with the original data overlayed
plt.contourf(xx, yy, preds, alpha=0.3, cmap='jet')
plt.scatter(df[x_variable], df[y_variable], c=df['class'], cmap='jet', edgecolors='k')
plt.show()