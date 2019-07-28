import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from apyori import apriori
from datetime import datetime
from pandas.tools.plotting import scatter_matrix



#read all data
foxdf=pd.read_csv('foxnews.csv')
cnndf=pd.read_csv('cnn.csv')
ixicdf=pd.read_csv('IXIC.csv')#nasdaq
golddf=pd.read_csv('gold.csv')
gspcdf=pd.read_csv('GSPC.csv')#sp500
foxCleandf=pd.read_csv('foxnews_cleaned.csv')
cnnCleandf=pd.read_csv('cnn_cleaned.csv')
ixicCleandf=pd.read_csv('IXIC_cleaned.csv')#nasdaq
goldCleandf=pd.read_csv('gold_cleaned.csv')
gspcCleandf=pd.read_csv('GSPC_cleaned.csv')#sp500
combinedf=pd.read_csv('clean-combined.csv')

print()
print("******************** Basic Statistical Analysis and data cleaning insight **********************")
print()

 #%%
#determine the mean median, and standard deviation of the five data sets. 

print("Data summary using describe method (stats about each column :")

print('-----------------------------------------------')
print("summary of Fox dataset:")
print(foxCleandf.describe())

print('-----------------------------------------------')
print("summary of CNN dataset:")
print(cnnCleandf.describe())

print('-----------------------------------------------')
print("summary of Nasdaq dataset:")
print(ixicCleandf.describe())

print('-----------------------------------------------')
print("summary of Gold dataset:")
print(goldCleandf.describe())

print('-----------------------------------------------')
print("summary of S&P500 dataset:")
print(gspcCleandf.describe())

 #%%

#============================================Detect potential outliers=======================================
 
plt.plot(gspcdf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with s&p 500')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.show ()
########We found that there are 4 outliers for s&p 500 dataset from the plot since the price just dramatically dropped.


plt.plot(ixicdf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with Nasdaq')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.show ()
#####By observing the plot, there is no outliers appeared with the dataset Nasdaq.


plt.plot(golddf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with gold')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.show ()
#####By observing the plot, there is 11 outliers appeared with the dataset Gold.

plt.boxplot(cnndf['retweet_count'].values)
plt.title ('the distribution of the counts of cnn twitter get retweets ')
plt.show ()

plt.boxplot(cnndf['favorite_count'].values)
plt.title ('the distribution of the counts of cnn twitter get favorites ')
plt.show ()
#####there are 1 or 2 potential outliers, but also maybe there is some major breaking news happened. 


plt.boxplot(foxdf['retweet_count'].values)
plt.title ('the distribution of the counts of fox twitter get retweets ')
plt.show ()

plt.boxplot(foxdf['favorite_count'].values)
plt.title ('the distribution of the counts of fox twitter get favorites ')
plt.show ()
#####there are 1 or 2 potential outliers, but also maybe there is some major breaking news happened. 
###we are going to detect the content with the large retweet counts.



###when the adjusted_price is equal to zero, which means there is no transactions on that day.
###we decided to remove the rows that contains value of zero.

print('we can fix and the amount that is not significant, delete the data that the closed price equals to 0')
golddf=golddf[golddf.adjusted_close!=0]
plt.plot(golddf['adjusted_close'])
plt.title ('adjusted_close price with gold dataset after cleanning')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.show ()

gspcdf=gspcdf[gspcdf.adjusted_close!=0]
plt.figure()
plt.plot(gspcdf['adjusted_close'])
plt.title ('adjusted_close price with s&p 500 dataset after cleanning')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.show ()

####Need to clean the CNN dataset, because we want to have the same time length for each dataset
cnnCleandf=cnnCleandf.iloc[0:364]

print()
print("****************************** Histogram and Correlations *******************************")
print()

#%%

#===============================================Histogram================================================


plt.figure()
plt.hist(goldCleandf['adjusted_close'])
plt.title("Gold Dataset Histogram with variable adjusted_close")
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.show ()



plt.figure()
plt.hist(goldCleandf['open'])
plt.title("Gold Dataset Histogram with variable open")
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.show ()


plt.figure()
plt.hist(goldCleandf['volume'])
plt.title("Gold Dataset Histogram with variable volume")
plt.xlabel("size of volumes")
plt.ylabel("Frequency")
plt.show ()



plt.figure()
plt.hist(ixicCleandf['adjusted_close'])
plt.title("Nasdaq Dataset Histogram with variable adjusted_close")
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.show ()

plt.figure()
plt.hist(ixicCleandf['volume'])
plt.title("Nasdaq Dataset Histogram with variable volume")
plt.xlabel("size of volumes")
plt.ylabel("Frequency")
plt.show ()

#%%

#=======================================Scatterplot Subplots & Correlations==============================

###To find the correlations and form a table:



print('----------------------------------------------------------------------')
print("correlation between all the pairs of these quantity variables for CNN Dataset:")
print(cnnCleandf[['favorite_count','retweet_count','point']].corr())




print('----------------------------------------------------------------------')
print("Plot the Scatterplot Subplots:")
###### Plot the Scatterplot Subplots:

names2= ['favorite_count','retweet_count','point']

cnnCleandf = cnnCleandf[names2]

scatter_matrix(cnnCleandf, alpha=0.7, figsize=(6, 6),diagonal='kde')
plt.show()







print()
print("****************************** Cluster Analysis *******************************")
print()
#read all data
fdf=pd.read_csv('FoxNews_cleaned.csv')
cdf=pd.read_csv('CNN_cleaned.csv')
idf=pd.read_csv('ixic_cleaned.csv')#nasdaq
godf=pd.read_csv('gold_cleaned.csv')
gsdf=pd.read_csv('gspc_cleaned.csv')#sp500

#function to do the kmeans clustering. It return a list of labels and  silhouette_score
#k is number of culster
#myData is the data we used to do the clustering
#verbose is a boolean value, if True it will print the silhouette_score and give a plot
def KMEANS(k,myData,verbose):
	
	
	x = myData.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	normalizedDataFrame = pd.DataFrame(x_scaled)
	
	kmeans = KMeans(n_clusters=k).fit(normalizedDataFrame)
	labels=kmeans.labels_ 
	# Determine if the clustering is good
	silhouette_avg = silhouette_score(normalizedDataFrame, labels)
	
	if verbose==True:
		print("Kmeans: For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
		print("****************************************************")
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#####
		# PCA
		# Let's convert our high dimensional data to 3 dimensions
		# using PCA
		pca3D = decomposition.PCA(3)

		# Turn the data into three columns with PCA
		plot_columns = pca3D.fit_transform(normalizedDataFrame)

		# Plot using a scatter plot and shade by cluster label
		ax.scatter(xs=plot_columns[:,0], ys=plot_columns[:,1],zs=plot_columns[:,2], c=labels)
		
		plt.show()
	return [labels,silhouette_avg]

#function to do the hierarchical clustering. It return a list of labels and  silhouette_score
#k is number of culster
#myData is the data we used to do the clustering
#verbose is a boolean value, if True it will print the silhouette_score and give a plot
def HIERAR(k,myData,verbose):
	
	
	x = myData.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	normalizedDataFrame = pd.DataFrame(x_scaled)
	ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(normalizedDataFrame)
	labels=ward.labels_
	# Determine if the clustering is good
	silhouette_avg = silhouette_score(normalizedDataFrame, labels)
	
	if verbose==True:
		print("Hierarchical: For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
		print("****************************************************")
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#####
		# PCA
		# Let's convert our high dimensional data to 3 dimensions
		# using PCA
		pca3D = decomposition.PCA(3)

		# Turn the data into three columns with PCA
		plot_columns = pca3D.fit_transform(normalizedDataFrame)

		# Plot using a scatter plot and shade by cluster label
		ax.scatter(xs=plot_columns[:,0], ys=plot_columns[:,1],zs=plot_columns[:,2], c=labels)
		#plt.savefig("scatter plot ")
		
		plt.show()
	return [labels,silhouette_avg]


#function to do the DBscan clustering. It return a list of labels and  silhouette_score
#myData is the data we used to do the clustering
#verbose is a boolean value, if True it will print the silhouette_score and give a plot
#eps and min_samples are parameters for DBSCAN and they have decault value of 0.1 and 5
def DBS(myData,verbose,eps=0.1,min_samples=5):
	
	
	x = myData.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	normalizedDataFrame = pd.DataFrame(x_scaled)
	db = DBSCAN(eps, min_samples).fit(normalizedDataFrame)
	labels=db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	# Determine if the clustering is good
	silhouette_avg = silhouette_score(normalizedDataFrame, labels)
	
	
	if verbose==True:
		print("DBscan: For n_clusters =", n_clusters_, "The average silhouette_score is :", silhouette_avg)	
		print("****************************************************")
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#####
		# PCA
		# Let's convert our high dimensional data to 3 dimensions
		# using PCA
		pca3D = decomposition.PCA(3)

		# Turn the data into three columns with PCA
		plot_columns = pca3D.fit_transform(normalizedDataFrame)

		# Plot using a scatter plot and shade by cluster label
		ax.scatter(xs=plot_columns[:,0], ys=plot_columns[:,1],zs=plot_columns[:,2], c=labels)
		
		
		plt.show()
	return [labels,silhouette_avg]


#convert our stock data and leave the three values we will use in our clustering
idf['nas_rate'] = idf['adjusted_close'].diff()/idf['adjusted_close']
idf=idf.dropna()
idf['diff']=idf['close']-idf['open']
idf=idf[['diff','volume','nas_rate']]
idf=idf.dropna()

gsdf['sp5_rate'] = gsdf['adjusted_close'].diff()/gsdf['adjusted_close']
gsdf=gsdf.dropna()
gsdf['diff']=gsdf['close']-gsdf['open']
gsdf=gsdf[['diff','volume','sp5_rate']]
gsdf=gsdf.dropna()

#function to find the best parameter for the clustering methods
def findbestpara(df):
	i=2
	memo=0
	imemo=0
	jmemo=0
	print('DBSCAN--------------------------------')
	while i<50:
		j=0.1
		while j<0.4:
			#print(i,j)
			if DBS(df,False,j,i)[1]>memo:
				memo=DBS(df,False,j,i)[1]
				imemo=i
				jmemo=j
			j+=0.1
		i+=1
	print('eps and min-sample with the highest silhouette_avg:')	
	print([jmemo,imemo])

	print('Kmeans--------------------------------')
	i=2	
	memo=0
	kmemo=0
	while i<30:
		if KMEANS(i,df,False)[1]>memo:
			memo=KMEANS(i,df,False)[1]
			kmemo=i	
		i+=1
	print('K with the highest silhouette_avg:')
	print(kmemo)
	print('Hierar--------------------------------')	
	i=2	
	memo=0
	kmemo=0
	while i<30:
		if HIERAR(i,df,False)[1]>memo:
			memo=HIERAR(i,df,False)[1]
			kmemo=i	
		i+=1
	print('K with the highest silhouette_avg:')
	print(kmemo)
#find the best parameter
print("Finding the best parameter...")
print('nasdaq:')
findbestpara(idf)
print('sp500:')
findbestpara(gsdf)

print('Calculate the score and plot the 3D scatter graph for the best parameter...')
print('nasdaq')
iKMlables=KMEANS(2,idf,True)[0]
iHIlables=HIERAR(2,idf,True)[0]
iDBlables=DBS(idf,True,0.3,2)[0]


idf['KMlables']=iKMlables
idf['HIlables']=iHIlables
idf['DBlables']=iDBlables

print('sp500')
gKMlables=KMEANS(2,gsdf,True)[0]
gHIlables=HIERAR(2,gsdf,True)[0]
gDBlables=DBS(gsdf,True,0.3,2)[0]


gsdf['KMlables']=gKMlables
gsdf['HIlables']=gHIlables
gsdf['DBlables']=gDBlables

print('Generate 2 csv with cluster labels')
idf.to_csv('nas-cluster.csv')
gsdf.to_csv('sp5-cluster.csv')
print()
print("****************************** Association Rules *******************************")
print()

#read all data again
fdf=pd.read_csv('FoxNews_cleaned.csv')
cdf=pd.read_csv('CNN_cleaned.csv')
idf=pd.read_csv('ixic_cleaned.csv')#nasdaq
godf=pd.read_csv('gold_cleaned.csv')
gsdf=pd.read_csv('gspc_cleaned.csv')#sp500




#Here is the function to calculate the sentimental test score for everyday, 
#by the formula: 
#sumperday([(favorite_count+retweet_count)/sumperday(favorite_count+retweet_count)]*point)=result
def combinpoint(df):
    df['New'] = df['favorite_count']+df['retweet_count']
    Group = df.groupby('created_at')

    def func(x):
        x['New2'] = x['New']/x['New'].sum()
        x['result'] = x['New2']*x['point']
        return x['result'].sum()

    temp = Group.apply(func)

    dic = {"Date":df['created_at'].unique(),
            "Result": temp.tolist()}
    try:
        NewData = pd.DataFrame(data=dic)
    except:
        NewData=None
    return NewData


#function to change the date format
def regudate (of):

	dt = datetime.strptime(of, '%Y/%m/%d')
	return(dt.strftime('%Y-%m-%d'))

fdf["created_at"]=fdf.created_at.apply(lambda x: regudate(x) )
cdf["created_at"]=cdf.created_at.apply(lambda x: regudate(x) )

#function to add increase_rate column to stock data
def addrate(d):
	d['increase_rate'] = d['adjusted_close'].diff()/d['adjusted_close']
	d=d.dropna()
	return d

#add increase_rate column to nasdaq data
idf['nas_rate'] = idf['adjusted_close'].diff()/idf['adjusted_close']
idf=idf.dropna()
#add increase_rate column to sp500 data
gsdf['sp_rate'] = gsdf['adjusted_close'].diff()/gsdf['adjusted_close']
gsdf=gsdf.dropna()
#keep useful columns and delete all others
idf=idf[['timestamp','nas_rate']]
gsdf=gsdf[['timestamp','sp_rate']]

#calculate per day sentimental test socre for cnn news
cnn=combinpoint(cdf)
#keep useful rows of cnn news
cnn=cnn.iloc[0:364]
#keep useful columns and delete all others
cnn.columns=['Date','cnnpoints']

#calculate per day sentimental test socre for fox news
fox=combinpoint(fdf)
#keep useful columns and delete all others
fox.columns=['Date','foxpoints']
#combine useful columns by date
news=cnn.join(fox.set_index('Date'),on='Date')
df=idf.join(news.set_index('Date'),on='timestamp')
df=gsdf.join(df.set_index('timestamp'),on='timestamp')
df = df[np.isfinite(df['cnnpoints'])]
df = df[np.isfinite(df['foxpoints'])]
#use only 4 columns
df=df[['cnnpoints','foxpoints','nas_rate','sp_rate']]
#generate csv for further use
df.to_csv('clean-combined.csv')
#binning news by positive and negative. binning stocks by increasing and decreasing
df['cnn']=pd.cut(df.cnnpoints,bins=[-1.1,0,1],labels=['cneg','cpos'])
df['fox']=pd.cut(df.foxpoints,bins=[-1.1,0,1],labels=['fneg','fpos'])
df['nas']=pd.cut(df.nas_rate,bins=[-1.1,0,1],labels=['nasde','nasin'])
df['sp5']=pd.cut(df.sp_rate,bins=[-1.1,0,1],labels=['sp5de','sp5in'])
df=df[['cnn','fox','nas','sp5']]
df.cnn=df.cnn.shift(-1)#Let's consider last day's cnn news' effect on one day's stock increase rate
df.fox=df.fox.shift(-1)#Let's consider last day's fox news' effect on one day's stock increase rate
df=df.dropna()
#generate value array
transactions = df.values
#do the associated rule test
results = pd.DataFrame(apriori(transactions))
results=results[results['support']>=0.3].reset_index()#let minSupport=0.3
results=results[['items','support','ordered_statistics']]
#generate csv 'result.csv'
results.to_csv('results.csv')
print(results)
