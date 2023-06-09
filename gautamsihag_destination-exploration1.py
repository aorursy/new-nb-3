import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context("poster")
sns.set(color_codes=True)
destination_features = pd.read_csv("../input/destinations.csv")
destination_features.shape
test_destinations = pd.read_csv("../input/test.csv", usecols=['srch_destination_id'])
a,b = np.unique(test_destinations, return_counts=True)
len(sorted(b))
srch_destinations, count = a,b
a = range(0,len(a),200)
count.sum()
1.0 * np.array(sorted(count)).cumsum()/count.sum()
fig, ax = plt.subplots(ncols=2, sharex=True)
ax[0].semilogy(sorted(count))
ax[1].plot(1.0 * np.array(sorted(count)).cumsum()/count.sum())
ax[0].set_xticks(range(0, len(srch_destinations), 10000))
ax[1].set_ylabel('Cumulative sum')
ax[0].set_ylabel('Search destination counts in test set (log scale)')
#frequent_destinations = srch_destinations[count >= 10]
print (1. * count[count >= 10].sum() / count.sum())
len(count[count>=10])
frequent_destinations = srch_destinations[count >= 10]
frequent_destinations
frequent_destination_features = destination_features[destination_features['srch_destination_id'].isin(frequent_destinations)]
frequent_destination_features.info()
frequent_destination_features = frequent_destination_features.drop('srch_destination_id', axis=1)
print(frequent_destination_features.shape)
correlations = frequent_destination_features.corr()
correlations.shape
correlations.tail()
f = plt.figure()
ax = sns.heatmap(correlations)
ax.set_xticks([])
ax.set_yticks([])
plt.title('Tartan or correlation matrix')
f.savefig('tartan.png', dpi=600)
plt.show()
correlations.values.reshape(correlations.size).shape
fig=plt.figure()
sns.distplot(correlations.values.reshape(correlations.size), bins=50, color='g')
plt.title('Correlation values')
plt.show()
fig.savefig('CorrelationHist')
g = sns.clustermap(correlations)
g.ax_heatmap.set_xticks([])
g.ax_heatmap.set_yticks([])
g.savefig('clustermap.png', dpi=300)
np.array(g.dendrogram_col.reordered_ind)
p = [8, 102, 120, 127, 74]
q = [70, 78, 10, 55, 19]
top_plot = frequent_destination_features[frequent_destination_features.columns[a]].sample(4000)
end_plot = frequent_destination_features[frequent_destination_features.columns[q]].sample(4000)
g = sns.PairGrid(top_plot, size=4)
g.map_upper(plt.scatter, s=6, alpha=0.4)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_diag(sns.kdeplot, legend=False, shade=True)
plt.suptitle('Top features graphically')
g.savefig('cluster_1.png', dpi=400) 
def blue_kde_hack(x, color, **kwargs):
    sns.kdeplot(x, color='b', **kwargs)
g = sns.PairGrid(end_plot, size=4)
g.map_upper(plt.scatter, s=6, alpha=0.4, color='b')
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_diag(blue_kde_hack, legend=False, shade=True)
plt.suptitle('5 correlated features from the end of the list')
g.savefig('cluster_end.png', dpi=400)
middle = [89, 69, 115, 105, 71]
def green_kde_hack(x, color, **kwargs):
    sns.kdeplot(x, color='g', **kwargs)
middle_plot = frequent_destination_features[frequent_destination_features.columns[middle]].sample(5000)
g = sns.PairGrid(middle_plot, size=4)
g.map_upper(plt.scatter, s=6, alpha=0.4, color='g')
g.map_lower(sns.kdeplot, cmap="Greens_d")
g.map_diag(green_kde_hack, legend=False, shade=True)
plt.suptitle('Middle features depicting correlations')
g.savefig('cluster_middle.png', dpi=400)