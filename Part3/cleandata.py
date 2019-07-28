import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from textblob import TextBlob
from datetime import datetime
from dateutil.parser import parse
import re


f = open('rawdata-info.txt','w')
old=sys.stdout
sys.stdout=f

foxdf=pd.read_csv('foxnews.csv')
cnndf=pd.read_csv('cnn.csv')
ixicdf=pd.read_csv('IXIC.csv')#nasdaq
golddf=pd.read_csv('gold.csv')
gspcdf=pd.read_csv('GSPC.csv')#sp500




#Visually inspect
print()
print('foxnews data head:')
print(foxdf.head())
print()
print('CNN data head:')
print(cnndf.head())
print()
print('Gold data head:')
print(golddf.head())
print()
print('NASDAQ data head:')
print(ixicdf.head())
print()
print('S&P 500 data head:')
print(gspcdf.head())
print()
print()
print('foxnews data shape:')
oldfox=foxdf.shape[0]
print(oldfox)
print(foxdf.shape)
print()
print("CNN data shape:")
oldcnn=cnndf.shape[0]
print(oldcnn)
print(cnndf.shape)
print()
print("gold data shape:")
oldgold=golddf.shape[0]
print(golddf.shape)
print()
print("NASDAQ data shape:")
oldixic=ixicdf.shape[0]
print(ixicdf.shape)
print()
print("S&P 500 data shape:")
oldgspc=gspcdf.shape[0]
print(gspcdf.shape)
print()
print()
print('foxnews data info:')
print(foxdf.info())
print()
print("CNN data info:")
print(cnndf.info())
print()
print("gold data info:")
print(golddf.info())
print()
print("NASDAQ data info:")
print(ixicdf.info())
print()
print("S&P 500 data info:")
print(gspcdf.info())
print()
print()
print('foxnews data describe:')
print(foxdf.describe())
print()
print("CNN data describe:")
print(cnndf.describe())
print()
print("gold data describe:")
print(golddf.describe())
print()
print("NASDAQ data describe:")
print(ixicdf.describe())
print()
print("S&P 500 data describe:")
print(gspcdf.describe())
print()

sys.stdout=old

######From the result above, we detect there are a few issues.




###detect any outliers with the data from plot. 
plt.plot(gspcdf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with s&p 500')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.savefig('rawsp500.png')
plt.show ()
###we found that there are 4 outliers for s&p 500 dataset from the plot since the price just dramatically dropped. 


plt.plot(ixicdf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with Nasdaq')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.savefig('rawsnasdaq.png')
plt.show ()
###there is no outliers with dataset Nasdaq.


plt.plot(golddf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with gold')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.savefig('rawgold.png')
plt.show ()
###there is no outliers with dataset gold.


plt.boxplot(cnndf['retweet_count'].values)
plt.title ('the distribution of the counts of cnn twitter get retweet ')
plt.savefig('rawcnn.png')
plt.show ()

plt.boxplot(foxdf['retweet_count'].values)
plt.title ('the distribution of the counts of fox twitter get retweet ')
plt.savefig('rawfox.png')
plt.show ()
###could be potential outliers, but also maybe there is some major breaking news happened. 
###we are going to detect the content with the large retweet counts.


print('Visually inspect--see rawdata-info.txt')
print()
print("Cleaning Process:")
print("drop duplicate for those data:")
golddf=golddf.drop_duplicates()
ixicdf=ixicdf.drop_duplicates()
gspcdf=gspcdf.drop_duplicates()
foxdf=foxdf.drop_duplicates()
cnndf=cnndf.drop_duplicates()
print('-----------------------------------------------')
print("clean gold:")
print('find data with volume =0]')
print(golddf.loc[golddf['adjusted_close']==0])
print('we cant fix and the amount is not significant, delete data close=0')
#we can't fix and the amount is not significant, delete data close=0
golddf=golddf[golddf.adjusted_close!=0] 

print('find rows with high=low')
print(golddf.loc[golddf['high']==golddf['low']])

print('There is no rows with high=low')


print('----------------------------------------------')
print("clean sp500:")
#find data with 'volume'==0)
print('find data with volume=0])')
print(gspcdf.loc[gspcdf['adjusted_close']==0])
print('we cant fix and the amount is not significant, delete data close=0')
#we can't fix and the amount is not significant, delete data close=0
gspcdf=gspcdf[gspcdf.adjusted_close!=0] 
#find data low=high
print()
print('find number of data with low=high')
print(gspcdf.loc[gspcdf['high']==gspcdf['low']].shape[0])
print()
#delete data whose high=low
print()
print('delete data whose high=low')
gspcdf=gspcdf[gspcdf.high!=gspcdf.low] 
#make sure no low=high
print()
print('make sure no low=high')
print(gspcdf.loc[gspcdf['high']==gspcdf['low']])
#make sure no volumn=0
print()
print('make sure no volumn=0')
print(gspcdf.loc[gspcdf['volume']==0])


print('----------------------------------------------')
print("clean Nasdaq:")
#find rows with volume=0
print()
print("find rows with volume=0")
print(ixicdf.loc[ixicdf['volume']==0])
#delete data with volume=0
print()
print('fix data with volume=0 with mean volume')
ixicdf.loc[ixicdf.volume==0,['volume']]=np.mean(ixicdf.volume)
print("check the data we fixed:")
print(ixicdf.iloc[19])
print(ixicdf.iloc[607])


#make sure no data with volume=0
print()
print('make sure no data with volume=0')
print(ixicdf.loc[ixicdf['volume']==0])
#find rows with high=low
print()
print('find rows with high=low')
print(ixicdf.loc[ixicdf['high']==ixicdf['low']])






print('--------------------------------------------')
print("clean twiteer raw data:")

#def function to change twitter date format to %Y/%m/%d
def regudate (of):
	#of="Wed Jul 26 18:01:34 +0000 2017"
	m=re.search(' \d\d\d\d',of)
	s=re.search(' \w\w\w \d\d',of)
	m=m.group(0)
	s=s.group(0)
	m=m+s




	dt = datetime.strptime(m, ' %Y %b %d')
	return(dt.strftime('%Y/%m/%d'))
print("change date format and sort data by date")	
foxdf["created_at"]=foxdf.created_at.apply(lambda x: regudate(x) )
foxdf = foxdf.sort_values(by='created_at',ascending=False)
cnndf["created_at"]=cnndf.created_at.apply(lambda x: regudate(x) )
cnndf = cnndf.sort_values(by='created_at',ascending=False)



#def a function that can remove nonsense (weblink and messy code) from text.
def cleantxt(of):
    #eg. of = "Las Vegas suspect's brother: 'We are completely dumbfounded' https://t.co/6sJOB3kk3A"
    of1 = re.sub(r"http\S+", "", of)
    of2 = re.sub('[^A-Za-z0-9]+', ' ', of1)
    return(of2)

#def a function that can do sentiment analysis(return a polarity coefficient from -1 to 1)	
def textpoint(text):
    blob = TextBlob(text)
    blob.tags           # [('The', 'DT'), ('titular', 'JJ'),
                    #  ('threat', 'NN'), ('of', 'IN'), ...]

    blob.noun_phrases   # WordList(['titular threat', 'blob',
                    #            'ultimate movie monster',
                    #            'amoeba-like mass', ...])
    for sentence in blob.sentences:
        return(sentence.sentiment.polarity)
        
#Clean all the contents and deliever to a new csv file
print("clean nonsense in twitter data's text column")
foxdf['text']=foxdf.text.apply(lambda x: cleantxt(x) )  
cnndf['text']=cnndf.text.apply(lambda x: cleantxt(x) )
print("give a sentiment test score to each tweet, take few minutes to process")
foxdf['point']=foxdf.text.apply(lambda x: textpoint(x) ) ########## time spender
cnndf['point']=cnndf.text.apply(lambda x: textpoint(x) ) ##########


print('find number of replys')
print(foxdf[foxdf.in_reply_to_screen_name.notnull()].shape)
print(cnndf[cnndf.in_reply_to_screen_name.notnull()].shape)
print('delete replys:')
#delete replys which is not related to our research
foxdf=foxdf[foxdf.in_reply_to_screen_name.isnull()]
cnndf=cnndf[cnndf.in_reply_to_screen_name.isnull()]
print(foxdf.shape )
print(cnndf.shape)




print('delete unralated variables source, in_reply_to_screen-name,is_retweet,id_str:')
foxdf=foxdf[['favorite_count','text','created_at','retweet_count','point']]#####
cnndf=cnndf[['favorite_count','text','created_at','retweet_count','point']]#####

print('drop rows with point==null')
foxdf=foxdf[foxdf.point.notnull()]
cnndf=cnndf[cnndf.point.notnull()]






##############################################################################
#Generate the data info after cleaning then we can see the result of our data cleaning

f1 = open('cleaneddata-info.txt','w')
sys.stdout=f1


#Visually inspect
print()
print('foxnews data head:')
print(foxdf.head())
print()
print('CNN data head:')
print(cnndf.head())
print()
print('Gold data head:')
print(golddf.head())
print()
print('NASDAQ data head:')
print(ixicdf.head())
print()
print('S&P 500 data head:')
print(gspcdf.head())
print()
print()
print('foxnews data shape:')
print(foxdf.shape)
print()
print("CNN data shape:")
print(cnndf.shape)
print()
print("gold data shape:")
print(golddf.shape)
print()
print("NASDAQ data shape:")
print(ixicdf.shape)
print()
print("S&P 500 data shape:")
print(gspcdf.shape)
print()
print()
print('foxnews data info:')
print(foxdf.info())
print()
print("CNN data info:")
print(cnndf.info())
print()
print("gold data info:")
print(golddf.info())
print()
print("NASDAQ data info:")
print(ixicdf.info())
print()
print("S&P 500 data info:")
print(gspcdf.info())
print()
print()
print('foxnews data describe:')
print(foxdf.describe())
print()
print("CNN data describe:")
print(cnndf.describe())
print()
print("gold data describe:")
print(golddf.describe())
print()
print("NASDAQ data describe:")
print(ixicdf.describe())
print()
print("S&P 500 data describe:")
print(gspcdf.describe())
print()

sys.stdout=old

######From the result above, we detect there are a few issues.



###detect any outliers with the data from plot. 
plt.plot(gspcdf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with s&p 500 after cleaning')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.savefig('cleanedsp500.png')
plt.show ()
###we found that there are 4 outliers for s&p 500 dataset from the plot since the price just dramatically dropped. 


plt.plot(ixicdf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with Nasdaq after cleaning')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.savefig('cleanednasdaq.png')
plt.show ()
###there is no outliers with dataset Nasdaq.


plt.plot(golddf['adjusted_close'])
plt.title ('the outlook of variable adjusted_close price with gold after cleaning')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.savefig('cleanedold.png')
plt.show ()
###there is no outliers with dataset gold.


plt.boxplot(cnndf['retweet_count'].values)
plt.title ('the distribution of the counts of cnn twitter get retweet after cleaning')
plt.savefig('cleanedcnn.png')
plt.show ()

plt.boxplot(foxdf['retweet_count'].values)
plt.title ('the distribution of the counts of fox twitter get retweet after cleaning')
plt.savefig('cleanedfox.png')
plt.show ()


print("=============================================================")
#we use 1-((raw rows numer-new rows number)/raw rows number) to measure the quality of cleanness for the raw dataset
print("raw data cleaness scores:")
print("gold:")
print(1-((oldgold-golddf.shape[0])/oldgold))
print("ixic:")
print(1-((oldixic-ixicdf.shape[0])/oldixic))
print("gspc:")
print(1-((oldgspc-gspcdf.shape[0])/oldgspc))
print("FoxNews:")
print(1-((oldfox-foxdf.shape[0])/oldfox))
print("CNN:")
print(1-((oldcnn-cnndf.shape[0])/oldcnn))


#select data for one year & delete unuseful values dividend amount and split coefficient
print('select stock and gold data for one year')
golddf['timestamp']=pd.to_datetime(golddf['timestamp'])
golddf = golddf[(golddf['timestamp'] > '2016/10/3') & (golddf['timestamp'] <= '2017/10/4')]
golddf=golddf[['timestamp','open','high','low','close','adjusted_close','volume']]

ixicdf['timestamp']=pd.to_datetime(ixicdf['timestamp'])
ixicdf = ixicdf[(ixicdf['timestamp'] > '2016/10/3') & (ixicdf['timestamp'] <= '2017/10/4')]
ixicdf=ixicdf[['timestamp','open','high','low','close','adjusted_close','volume']]

gspcdf['timestamp']=pd.to_datetime(gspcdf['timestamp'])
gspcdf = gspcdf[(gspcdf['timestamp'] > '2016/10/3') & (gspcdf['timestamp'] <= '2017/10/4')]
gspcdf=gspcdf[['timestamp','open','high','low','close','adjusted_close','volume']]

print("generate 5 cleaned csv")
golddf.to_csv('gold_cleaned.csv',index=False)
ixicdf.to_csv('ixic_cleaned.csv',index=False)
gspcdf.to_csv('gspc_cleaned.csv',index=False)
foxdf.to_csv('FoxNews_cleaned.csv', index=False)
cnndf.to_csv('CNN_cleaned.csv',index =False)




