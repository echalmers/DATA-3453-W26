import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


# load the dataset
df = pd.read_csv('./data/labour.csv', index_col=None)
x = df.drop('class', axis=1)
y = df['class']
# (should we normalize X?)


# fit a logistic regression model
model = LogisticRegression()
model.fit(x, y)

# get the model's coeficients for each feature in X
coefs = pd.Series(index=x.columns, data=model.coef_[0]).sort_values()
print(coefs)

# visualize coefficients
# YOUR CODE HERE


# now build a SVC model. It has no coeffients per se, so let's use a model-agnostic method of measuring feature importance
model2 = SVC(gamma=10)
model2.fit(x, y)
r = permutation_importance(model2, x, y)

# Inspect r - it contains importance score info. Visualize it!
# YOUR CODE HERE
