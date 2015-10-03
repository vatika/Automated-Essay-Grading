
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('features_3.csv', index_col=0)
data.head()

colnames = ['id', 'set', 'score', 'x', 'y','z']



scores = list(data.score)
x = list(data.x)
y = list(data.y)

plt.plot(x,scores)
plt.show()
