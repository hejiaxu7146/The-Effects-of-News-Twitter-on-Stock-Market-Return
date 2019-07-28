
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
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

import warnings
warnings.filterwarnings('ignore')








print("****************************** Topic Modeling *******************************")
print()

#read all data again
fdf=pd.read_csv('FoxNews_cleaned.csv')
cdf=pd.read_csv('CNN_cleaned.csv')


idf=pd.read_csv('ixic_cleaned.csv')#nasdaq
godf=pd.read_csv('gold_cleaned.csv')
gsdf=pd.read_csv('gspc_cleaned.csv')#sp500




################################## modeling #############################################


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic " + str(topic_idx))
        #print (topic.argsort())
        print (" "+str([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

documents = cdf['text'].tolist()
documents1 = fdf['text'].tolist()



no_features = 1000

# tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()



num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()


cdf['topic']=clusters
#cdf.to_csv('cnn_topic.csv')





#cnn.to_csv("cnn_topic.csv")

tfidf_vectorizer1 = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf1 = tfidf_vectorizer1.fit_transform(documents1)

tfidf_feature_names1 = tfidf_vectorizer1.get_feature_names()



km1 = KMeans(n_clusters=num_clusters)
km1.fit(tfidf1)
clusters = km1.labels_.tolist()


fdf['topic']=clusters
#fdf.to_csv('fox_topic.csv')






print("CNN News Top terms per topic:")
print()

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
#print(order_centroids)




for i in range(num_clusters):
    print()
    print("Cluster",i, "words: ")
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster		
        print(tfidf_feature_names[ind]," ", end='')
print()
print("Fox News Top terms per topic:")
print()
order_centroids1 = km1.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print()
    print("Cluster",i, "words: ")
    for ind in order_centroids1[i, :10]: #replace 6 with n words per cluster
        print(tfidf_feature_names1[ind]," ",end='')
#cdf=pd.read_csv('cnn_topic.csv')
#fdf=pd.read_csv('fox_topic.csv')




################################## topic ################################################
cdf0=cdf[cdf['topic']==0]
cdf1=cdf[cdf['topic']==1]
cdf2=cdf[cdf['topic']==2]
cdf3=cdf[cdf['topic']==3]
cdf4=cdf[cdf['topic']==4]


fdf0=fdf[fdf['topic']==0]
fdf1=fdf[fdf['topic']==1]
fdf2=fdf[fdf['topic']==2]
fdf3=fdf[fdf['topic']==3]
fdf4=fdf[fdf['topic']==4]










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


cdf0["created_at"]=cdf0.created_at.apply(lambda x: regudate(x) )
cdf1["created_at"]=cdf1.created_at.apply(lambda x: regudate(x) )
cdf2["created_at"]=cdf2.created_at.apply(lambda x: regudate(x) )
cdf3["created_at"]=cdf3.created_at.apply(lambda x: regudate(x) )
cdf4["created_at"]=cdf4.created_at.apply(lambda x: regudate(x) )

fdf0["created_at"]=fdf0.created_at.apply(lambda x: regudate(x) )
fdf1["created_at"]=fdf1.created_at.apply(lambda x: regudate(x) )
fdf2["created_at"]=fdf2.created_at.apply(lambda x: regudate(x) )
fdf3["created_at"]=fdf3.created_at.apply(lambda x: regudate(x) )
fdf4["created_at"]=fdf4.created_at.apply(lambda x: regudate(x) )

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



def gefinaldata(newsdf,idf,gsdf):
	newsdf=combinpoint(newsdf)
	df=idf.join(newsdf.set_index('Date'),on='timestamp')
	df=gsdf.join(df.set_index('timestamp'),on='timestamp')
	return df
	
cnn0=gefinaldata(cdf0,idf,gsdf).dropna()
cnn1=gefinaldata(cdf1,idf,gsdf).dropna()
cnn2=gefinaldata(cdf2,idf,gsdf).dropna()
cnn3=gefinaldata(cdf3,idf,gsdf).dropna()
cnn4=gefinaldata(cdf4,idf,gsdf).dropna()

fox0=gefinaldata(fdf0,idf,gsdf).dropna()
fox1=gefinaldata(fdf1,idf,gsdf).dropna()
fox2=gefinaldata(fdf2,idf,gsdf).dropna()
fox3=gefinaldata(fdf3,idf,gsdf).dropna()
fox4=gefinaldata(fdf4,idf,gsdf).dropna()

print(cnn0)
print(cnn4)
print(fox0)
print(fox4)
print()
print("CNN topic1 vs S&P500")
print(cnn0['sp_rate'].corr(cnn0['Result']))
print("CNN topic1 vs nasdaq")
print(cnn0['nas_rate'].corr(cnn0['Result']))
print()
print("CNN topic2 vs S&P500")
print(cnn1['sp_rate'].corr(cnn1['Result']))
print("CNN topic2 vs nasdaq")
print(cnn1['nas_rate'].corr(cnn1['Result']))
print()
print("CNN topic3 vs S&P500")
print(cnn2['sp_rate'].corr(cnn2['Result']))
print("CNN topic3 vs nasdaq")
print(cnn2['nas_rate'].corr(cnn2['Result']))
print()
print("CNN topic4 vs S&P500")
print(cnn3['sp_rate'].corr(cnn3['Result']))
print("CNN topic4 vs nasdaq")
print(cnn3['nas_rate'].corr(cnn3['Result']))
print()
print("CNN topic5 vs S&P500")
print(cnn4['sp_rate'].corr(cnn4['Result']))
print("CNN topic5 vs nasdaq")
print(cnn4['nas_rate'].corr(cnn4['Result']))
print()

print("Fox topic1 vs S&P500")
print(fox0['sp_rate'].corr(fox0['Result']))
print("Fox topic1 vs nasdaq")
print(fox0['nas_rate'].corr(fox0['Result']))
print()
print("Fox topic2 vs S&P500")
print(fox1['sp_rate'].corr(fox1['Result']))
print("Fox topic2 vs nasdaq")
print(fox1['nas_rate'].corr(fox1['Result']))
print()
print("Fox topic3 vs S&P500")
print(fox2['sp_rate'].corr(fox2['Result']))
print("Fox topic3 vs nasdaq")
print(fox2['nas_rate'].corr(fox2['Result']))
print()
print("Fox topic4 vs S&P500")
print(fox3['sp_rate'].corr(fox3['Result']))
print("Fox topic4 vs nasdaq")
print(fox3['nas_rate'].corr(fox3['Result']))
print()
print("Fox topic5 vs S&P500")
print(fox4['sp_rate'].corr(fox4['Result']))
print("Fox topic5 vs nasdaq")
print(fox4['nas_rate'].corr(fox4['Result']))
print()
