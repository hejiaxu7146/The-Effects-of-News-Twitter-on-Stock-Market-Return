import pandas as pd
from datetime import datetime

#Create the new column increase rate which directly shows how much it increase everyday
d=pd.read_csv('gold_cleaned.csv')
d['increase_rate'] = d['adjusted_close'].diff()/d['adjusted_close']
d=d.dropna()
d.to_csv('goldfinal.csv', index=False)
d1=pd.read_csv('gspc_cleaned.csv')
d1['increase_rateg'] = d1['adjusted_close'].diff()/d1['adjusted_close']
d1=d1.dropna()
d1.to_csv('gspcfinal.csv', index=False)
d2=pd.read_csv('ixic_cleaned.csv')
d2['increase_ratei'] = d2['adjusted_close'].diff()/d2['adjusted_close']
d2=d2.dropna()
d2.to_csv('ixicfinal.csv', index=False)

#change the increase rate or decrease rate from numerical to categorical
df11 = pd.read_csv('goldfinal.csv')
df111 = pd.read_csv('goldfinal.csv')
df11.increase_rate[df111.increase_rate >= 0] = 'increase'
df11.increase_rate[df111.increase_rate < 0] = 'decrease'
df11 = df11.rename(columns = {'timestamp':'Date'})
#df11.to_csv('goldml.csv', index=False)

df22 = pd.read_csv('gspcfinal.csv')
df222 = pd.read_csv('gspcfinal.csv')
df22.increase_rateg[df222.increase_rateg >= 0] = 'increase'
df22.increase_rateg[df222.increase_rateg < 0] = 'decrease'
df22 = df22.rename(columns = {'timestamp':'Date'})
#df22.to_csv('gspcml.csv', index=False)

df33 = pd.read_csv('ixicfinal.csv')
df333 = pd.read_csv('ixicfinal.csv')
df33.increase_ratei[df333.increase_ratei >= 0] = 'increase'
df33.increase_ratei[df333.increase_ratei < 0] = 'decrease'
df33 = df33.rename(columns = {'timestamp':'Date'})
#df33.to_csv('ixicml.csv', index=False)

def main(df):
    df['New'] = df['favorite_count']+df['retweet_count']
    Group = df.groupby('created_at')

    def func(x):
        x['New2'] = x['New']/x['New'].sum()
        x['result'] = x['New2']*x['point']
        return x['result'].sum()
    
    def func2(x):
        return x['New'].sum()

    temp = Group.apply(func)
    temp2 = Group.apply(func2)

    dic = {"Date":df['created_at'].unique(),
            "Result": temp.tolist(), "Favorite&retweet": temp2.tolist()}
    try:
        NewData = pd.DataFrame(data=dic)
    except:
        NewData=None
    return NewData

#test to see the main function works correctly
if __name__ == '__main__':
    CNN_clean = pd.read_csv('CNN_cleaned.csv')
    FoxNews = pd.read_csv('FoxNews_cleaned.csv')
    print(main(CNN_clean).head(6))

#apply the function
p1 = pd.read_csv('CNN_cleaned.csv')
p2 = pd.read_csv('FoxNews_cleaned.csv')
p11 = main(p1)
p22 = main(p2)

#change columns' names
p11 = p11.rename(columns = {'Favorite&retweet':'CNNF&R'})
p11 = p11.rename(columns = {'Result':'CNNPoints'})
p22 = p22.rename(columns = {'Favorite&retweet':'FoxNewsF&R'})
p22 = p22.rename(columns = {'Result':'FoxNewsPoints'})

#Combine them into one csv file
p11['FoxNewsF&R'] = p22['FoxNewsF&R']
p11['FoxNewsPoints'] = p22['FoxNewsPoints']

p11 = p11.dropna()

p11.to_csv('NewsCombined.csv', index=False)

#Create new csv file for one way anova test for first hypothesis
Anovafile = pd.DataFrame()
Anovafile1 = pd.DataFrame()
Anovafile2 = pd.DataFrame()
Anova1 = p11
Anova2 = p22
Anova1 = Anova1.assign(Group='CNN')
Anova2 = Anova2.assign(Group='FoxsNews')
Anovafile1['Points'] = Anova1['CNNPoints']
Anovafile1['Group'] = Anova1['Group']
Anovafile2['Points'] = Anova2['FoxNewsPoints']
Anovafile2['Group'] = Anova2['Group']
Anovafile =pd.concat([Anovafile1, Anovafile2])
Anovafile.to_csv('Anovafile.csv', index=False)



#Changed the function main a little bit to create new csv files for both news.
#The new file willl contain only counts of favorites, retweets and Absolutely value of Points.
#It is the preparation for the second hypothesis test, which is to test the realtionship of variables inside News.

def mainchanged(df):
    df['New'] = df['favorite_count']+df['retweet_count']
    Group = df.groupby('created_at')

    def func(x):
        x['New2'] = x['New']/x['New'].sum()
        x['result'] = x['New2']*x['point']
        return x['result'].sum()
    
    def func2(x):
        return x['favorite_count'].sum()
    
    def func3(x):
        return x['retweet_count'].sum()

    temp = Group.apply(func)
    temp2 = Group.apply(func2)
    temp3 = Group.apply(func3)

    dic = {"Date":df['created_at'].unique(),
            "Points": temp.tolist(), "Favorites": temp2.tolist(), "Retweets": temp3.tolist()}
    try:
        NewData = pd.DataFrame(data=dic)
    except:
        NewData=None
    return NewData
pcnn = mainchanged(p1)
pfox = mainchanged(p2)

pcnn['Points'] = pcnn['Points'].abs()
pfox['Points'] = pfox['Points'].abs()

pcnn.drop('Date', axis=1, inplace=True)
pfox.drop('Date', axis=1, inplace=True)

pcnn.to_csv('CNN_test2.csv',index=False)
pfox.to_csv('FoxNews_test2.csv',index=False)



# clean column'Date' format to make sure they can easily combine with other dataframes
def regudate (of):
    dt = datetime.strptime(of, '%Y/%m/%d')
    return(dt.strftime('%Y-%m-%d'))

p11['Date'] = p11.Date.apply(lambda x: regudate(x))


#Drop unnecessary columns
goldml = df11
goldml.drop('open', axis=1, inplace=True)
goldml.drop('high', axis=1, inplace=True)
goldml.drop('low', axis=1, inplace=True)
goldml.drop('close', axis=1, inplace=True)
goldml.drop('adjusted_close', axis=1, inplace=True)
goldml.drop('volume', axis=1, inplace=True)

gspcml = df22
gspcml.drop('open', axis=1, inplace=True)
gspcml.drop('high', axis=1, inplace=True)
gspcml.drop('low', axis=1, inplace=True)
gspcml.drop('close', axis=1, inplace=True)
gspcml.drop('adjusted_close', axis=1, inplace=True)
gspcml.drop('volume', axis=1, inplace=True)

ixicml = df33
ixicml.drop('open', axis=1, inplace=True)
ixicml.drop('high', axis=1, inplace=True)
ixicml.drop('low', axis=1, inplace=True)
ixicml.drop('close', axis=1, inplace=True)
ixicml.drop('adjusted_close', axis=1, inplace=True)
ixicml.drop('volume', axis=1, inplace=True)

#Generate the files that are ready for predictive models to do the tests
NGold=p11.join(goldml.set_index('Date'),on='Date')
NGold = NGold.dropna()
NGold.drop('Date', axis=1, inplace=True)
NGold.to_csv('News_vs_Gold.csv',index=False)
NGspc=p11.join(goldml.set_index('Date'),on='Date')
NGspc = NGspc.dropna()
NGspc.drop('Date', axis=1, inplace=True)
NGspc.to_csv('News_vs_Gspc.csv',index=False)
NIxic=p11.join(goldml.set_index('Date'),on='Date')
NIxic = NIxic.dropna()
NIxic.drop('Date', axis=1, inplace=True)
NIxic.to_csv('News_vs_Ixic.csv',index=False)

