import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# load the "iris" dataset
df = pd.read_csv('./data/iris.csv', index_col=None)

# we're only going to use a subset of the data for now
df = df[['petal length (cm)', 'sepal width (cm)', 'class']]

# plot the data
sns.scatterplot(df, x='petal length (cm)', y='sepal width (cm)', hue='class')

# split the dataframe into X (the petal length & width columns) and Y (the class column)
x = # YOUR CODE HERE
y = # YOUR CODE HERE

# fit the LogisticRegression model to the data
model = LogisticRegression()
model.fit(x, y)

# use the model to make a prediction about a new, hypothetical flower
new_flower = [3, 2] #  <----  try different petal lengths and widths here
plt.scatter(*new_flower, marker='x', c='k')
prediction = model.predict([new_flower])
plt.annotate(text=f'predicted {prediction}', xy=new_flower)


plt.show()