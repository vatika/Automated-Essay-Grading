
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('features_3.csv', index_col=0)
data.head()

colnames = ['id', 'set', 'human_score', 'misspell_words']



scores = list(data.human_score)
x = list(data.misspell_words)

plt.scatter(x, scores)
plt.show()
