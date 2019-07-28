import numpy as np; np.random.seed(22)
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.lines as mlines
point=pd.read_csv('NewsCombined.csv')
plt.figure(figsize=(14,6))
ax = sns.tsplot(point['CNNPoints'],color='green')
ax = sns.tsplot(point['FoxNewsPoints'],color='red')
ax.set_title('TextBlob Sentiment Score')
handles, labels = ax.get_legend_handles_labels()
cnn = mlines.Line2D([], [], color='green',
                          markersize=15, label='CNN')
fox = mlines.Line2D([], [], color='red',
                          markersize=15, label='Fox')
plt.legend(handles=[cnn,fox])

plt.show()
