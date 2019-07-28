import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Load a numpy record array from yahoo csv data with fields date, open, close,
# volume, adj_close from the mpl-data/example directory. The record array
# stores the date as an np.datetime64 with a day unit ('D') in the date column.
#with cbook.get_sample_data('goog.npz') as datafile:
#    price_data = np.load(datafile)['price_data'].view(np.recarray)
#price_data = price_data[-250:]  # get the most recent 250 trading days

df = pd.read_csv("goldfinal.csv")
price_data = df

delta1 = np.diff(price_data.adjusted_close) / price_data.adjusted_close[:-1]
price = price_data.iloc[0:294, 5] # 1st, 4th, 7th, 25th row + 1st 6th 7th columns.

# Marker size in units of points^2
volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

fig, ax = plt.subplots()
ax.scatter(delta1[:-1], price, c=close, s=volume, alpha=0.5)

ax.set_xlabel('Increase rate of the day', fontsize=15)
ax.set_ylabel('Adjusted value of the day', fontsize=15)
ax.set_title('Volume and Price percent change for Gold in each day of the year')
plt.annotate('Color: Close price; Bubble Size： Volumes', (0,0), (0, -20))

ax.grid(True)
fig.tight_layout()

plt.show()

df = pd.read_csv("ixicfinal.csv")
price_data = df

delta1 = np.diff(price_data.adjusted_close) / price_data.adjusted_close[:-1]

# Marker size in units of points^2
volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

fig, ax = plt.subplots()
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and Price percent change for Nasdaq in each day of the year')

ax.grid(True)
fig.tight_layout()

plt.show()

df = pd.read_csv("gspcfinal.csv")
price_data = df

delta1 = np.diff(price_data.adjusted_close) / price_data.adjusted_close[:-1]

# Marker size in units of points^2
volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

fig, ax = plt.subplots()
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and Price percent change for S&P500 in each day of the year')

ax.grid(True)
fig.tight_layout()

plt.show()


d1 = pd.read_csv("goldfinal.csv")
d2 = pd.read_csv("gspcfinal.csv")
d3 = pd.read_csv("ixicfinal.csv")

###seperate the volume sum by three dates 6/30/2017, 3/31/2017, 12/30/2016
### And start data 10/4/2016, end date 10/3/2017
gold1 = d1.loc[0:79, ["volume"]].sum()
gold2 = d1.loc[80:155, ["volume"]].sum()
gold3 = d1.loc[156:226, ["volume"]].sum()
gold4 = d1.loc[227:296, ["volume"]].sum()

gspc1 = d2.loc[0:66, ["volume"]].sum()
gspc2 = d2.loc[67:129, ["volume"]].sum()
gspc3 = d2.loc[130:191, ["volume"]].sum()
gspc4 = d2.loc[192:252, ["volume"]].sum()

ixic1 = d3.loc[0:66, ["volume"]].sum()
ixic2 = d3.loc[67:129, ["volume"]].sum()
ixic3 = d3.loc[130:191, ["volume"]].sum()
ixic4 = d3.loc[192:252, ["volume"]].sum()

print(gold1, gold2, gold3, gold4)
data = [[6251262980, gspc4, ixic4],
        [6823799480, gspc3, ixic3],
        [7177958826, gspc2, ixic2],
        [9555776363, gspc1, ixic1]]

columns = ('Gold', 'S&P500', 'Nasdaq')
rows = ['Until 10/3/2017', 'Until 6/30/2017', 'Until 3/31/2017', 'Until 12/30/2016']

values = np.arange(0, 1000, 200)
value_increment = 1000000000

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.4f' % (x / 1000000000000.0000) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("volume in billions")
#plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Sum of the volume in the Year Start from 10/4/2016')

plt.show()


df = pd.read_csv("CNN_test2.csv")
fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
# Make data.
X = df["Favorites"]
Y = df["Retweets"]
#X, Y = np.meshgrid(X, Y)
Z = df["Points"]

# Plot the surface.
ax.scatter(X, Y, Z, c="b", marker = "o")

ax.set_xlabel("Number of Favorites")
ax.set_ylabel("Number of Retweets")
ax.set_zlabel("Text Blob Points")
ax.set_title('CNN')
plt.show()


df = pd.read_csv("FoxNews_test2.csv")
fig = plt.figure(figsize=(12, 9))
ax = fig.gca(projection='3d')
# Make data.
X = df["Favorites"]
Y = df["Retweets"]
#X, Y = np.meshgrid(X, Y)
Z = df["Points"]

# Plot the surface.
ax.scatter(X, Y, Z, c="r", marker = "o")

ax.set_xlabel("Number of Favorites")
ax.set_ylabel("Number of Retweets")
ax.set_zlabel("Text Blob Points")
ax.set_title('FoxNews')
plt.show()

from sklearn import cross_validation

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


#Third hypothesis test
#Using five predictve models to test

def function(myData):
    valueArray = myData.values
    X = valueArray[:,0:4]
    Y = valueArray[:,4]
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

    num_folds = 10
    num_instances = len(X_train)
    seed = 7
    scoring = 'accuracy'
    
    
    #Normalize the Data
    X = preprocessing.normalize(X)
    
######################################################
# Use different algorithms to build models
######################################################

# Add each algorithm and its name to the model array
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF',RandomForestClassifier()))
    
# Evaluate each model, add results to a results array,
# Print the accuracy results (remember these are averages and std
    results =[]
    names = []
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)
  
######################################################
# For the best model, see how well it does on the
# validation test.  This is for KNeighborsClassifier
######################################################
# Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validate)
    accuracy = pd.DataFrame()
    accuracy = accuracy.append({'KNN': accuracy_score(Y_validate, predictions)}, ignore_index = True)


######################################################
# For the best model, see how well it does on the
# validation test. This is for DecisionTreeClassifier
######################################################
# Make predictions on validation dataset
    cart = DecisionTreeClassifier()
    cart.fit(X_train, Y_train)
    predictions = cart.predict(X_validate)
    accuracy = accuracy.append({'Cart': accuracy_score(Y_validate, predictions)}, ignore_index = True)

######################################################
# For the best model, see how well it does on the
# validation test. This is for GaussianNB
######################################################
# Make predictions on validation dataset
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    predictions = nb.predict(X_validate)
    accuracy = accuracy.append({'NB': accuracy_score(Y_validate, predictions)}, ignore_index = True)


######################################################
# For the best model, see how well it does on the
# validation test. This is for SVM
######################################################
# Make predictions on validation dataset
    svm = SVC()
    svm.fit(X_train, Y_train)
    predictions = svm.predict(X_validate)
    accuracy = accuracy.append({'SVC': accuracy_score(Y_validate, predictions)}, ignore_index = True)


######################################################
# For the best model, see how well it does on the
# validation test. This is for RandomForestClassifier
######################################################
# Make predictions on validation dataset
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_validate)
    accuracy = accuracy.append({'RF': accuracy_score(Y_validate, predictions)}, ignore_index = True)
    accuracy.Cart=accuracy.Cart.shift(-1)
    accuracy.NB=accuracy.NB.shift(-2)
    accuracy.SVC=accuracy.SVC.shift(-3)
    accuracy.RF=accuracy.RF.shift(-4)
    accuracy = accuracy.dropna()
    return accuracy


d1 = pd.read_csv('News_vs_Gspc.csv')
d2 = pd.read_csv('News_vs_Gold.csv')
d3 = pd.read_csv('News_vs_Ixic.csv')
t1 = function(d1)
t2 = function(d2)
t3 = function(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='Combined News Prediction Results in the same day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

d1.increase_rateg=d1.increase_rateg.shift(+1)
d2.increase_rate=d2.increase_rate.shift(+1)
d3.increase_ratei=d3.increase_ratei.shift(+1)
d1 = d1.dropna()
d2 = d2.dropna()
d3 = d3.dropna()
t1 = function(d1)
t2 = function(d2)
t3 = function(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='Combined News Prediction Results for the next trading day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

d1.increase_rateg=d1.increase_rateg.shift(+1)
d2.increase_rate=d2.increase_rate.shift(+1)
d3.increase_ratei=d3.increase_ratei.shift(+1)
d1 = d1.dropna()
d2 = d2.dropna()
d3 = d3.dropna()
t1 = function(d1)
t2 = function(d2)
t3 = function(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='Combined News Prediction Results for the third trading open day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

def function1(myData):
    valueArray = myData.values
    X = valueArray[:,0:2]
    Y = valueArray[:,2]
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

    num_folds = 10
    num_instances = len(X_train)
    seed = 7
    scoring = 'accuracy'
    
    
    #Normalize the Data
    X = preprocessing.normalize(X)
    
######################################################
# Use different algorithms to build models
######################################################

# Add each algorithm and its name to the model array
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RF',RandomForestClassifier()))
    
# Evaluate each model, add results to a results array,
# Print the accuracy results (remember these are averages and std
    results =[]
    names = []
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)
  
######################################################
# For the best model, see how well it does on the
# validation test.  This is for KNeighborsClassifier
######################################################
# Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validate)
    accuracy = pd.DataFrame()
    accuracy = accuracy.append({'KNN': accuracy_score(Y_validate, predictions)}, ignore_index = True)


######################################################
# For the best model, see how well it does on the
# validation test. This is for DecisionTreeClassifier
######################################################
# Make predictions on validation dataset
    cart = DecisionTreeClassifier()
    cart.fit(X_train, Y_train)
    predictions = cart.predict(X_validate)
    accuracy = accuracy.append({'Cart': accuracy_score(Y_validate, predictions)}, ignore_index = True)

######################################################
# For the best model, see how well it does on the
# validation test. This is for GaussianNB
######################################################
# Make predictions on validation dataset
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    predictions = nb.predict(X_validate)
    accuracy = accuracy.append({'NB': accuracy_score(Y_validate, predictions)}, ignore_index = True)


######################################################
# For the best model, see how well it does on the
# validation test. This is for SVM
######################################################
# Make predictions on validation dataset
    svm = SVC()
    svm.fit(X_train, Y_train)
    predictions = svm.predict(X_validate)
    accuracy = accuracy.append({'SVC': accuracy_score(Y_validate, predictions)}, ignore_index = True)


######################################################
# For the best model, see how well it does on the
# validation test. This is for RandomForestClassifier
######################################################
# Make predictions on validation dataset
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_validate)
    accuracy = accuracy.append({'RF': accuracy_score(Y_validate, predictions)}, ignore_index = True)
    accuracy.Cart=accuracy.Cart.shift(-1)
    accuracy.NB=accuracy.NB.shift(-2)
    accuracy.SVC=accuracy.SVC.shift(-3)
    accuracy.RF=accuracy.RF.shift(-4)
    accuracy = accuracy.dropna()
    return accuracy
d1 = pd.read_csv('News_vs_Gspc.csv')
d2 = pd.read_csv('News_vs_Gold.csv')
d3 = pd.read_csv('News_vs_Ixic.csv')
d1.drop('FoxNewsF&R', axis=1, inplace=True)
d1.drop('FoxNewsPoints', axis=1, inplace=True)
d2.drop('FoxNewsF&R', axis=1, inplace=True)
d2.drop('FoxNewsPoints', axis=1, inplace=True)
d3.drop('FoxNewsF&R', axis=1, inplace=True)
d3.drop('FoxNewsPoints', axis=1, inplace=True)
t1 = function1(d1)
t2 = function1(d2)
t3 = function1(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='CNN Prediction Results in the same day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

d1.increase_rateg=d1.increase_rateg.shift(+1)
d2.increase_rate=d2.increase_rate.shift(+1)
d3.increase_ratei=d3.increase_ratei.shift(+1)
d1 = d1.dropna()
d2 = d2.dropna()
d3 = d3.dropna()
t1 = function1(d1)
t2 = function1(d2)
t3 = function1(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='CNN Prediction Results for the next trading day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

d1.increase_rateg=d1.increase_rateg.shift(+1)
d2.increase_rate=d2.increase_rate.shift(+1)
d3.increase_ratei=d3.increase_ratei.shift(+1)
d1 = d1.dropna()
d2 = d2.dropna()
d3 = d3.dropna()
t1 = function1(d1)
t2 = function1(d2)
t3 = function1(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='CNN Prediction Results for the third trading open day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

#Fox News
d1 = pd.read_csv('News_vs_Gspc.csv')
d2 = pd.read_csv('News_vs_Gold.csv')
d3 = pd.read_csv('News_vs_Ixic.csv')
d1.drop('CNNF&R', axis=1, inplace=True)
d1.drop('CNNPoints', axis=1, inplace=True)
d2.drop('CNNF&R', axis=1, inplace=True)
d2.drop('CNNPoints', axis=1, inplace=True)
d3.drop('CNNF&R', axis=1, inplace=True)
d3.drop('CNNPoints', axis=1, inplace=True)
t1 = function1(d1)
t2 = function1(d2)
t3 = function1(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='FoxNews Prediction Results in the same day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

d1.increase_rateg=d1.increase_rateg.shift(+1)
d2.increase_rate=d2.increase_rate.shift(+1)
d3.increase_ratei=d3.increase_ratei.shift(+1)
d1 = d1.dropna()
d2 = d2.dropna()
d3 = d3.dropna()
t1 = function1(d1)
t2 = function1(d2)
t3 = function1(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='FoxNews Prediction Results for the next trading day')
ax.set_ylabel("Prediction Accuracy")
plt.show()

d1.increase_rateg=d1.increase_rateg.shift(+1)
d2.increase_rate=d2.increase_rate.shift(+1)
d3.increase_ratei=d3.increase_ratei.shift(+1)
d1 = d1.dropna()
d2 = d2.dropna()
d3 = d3.dropna()
t1 = function1(d1)
t2 = function1(d2)
t3 = function1(d3)
frames = [t1, t2, t3]
result = pd.concat(frames)
print(result)
result["Stocks"]=["S&P500", "Gold", "Nasdaq"]
ax = result.set_index('Stocks').plot.bar(title='FoxNews Prediction Results for the third trading open day')
ax.set_ylabel("Prediction Accuracy")
plt.show()


df = pd.read_csv("NewsCombined.csv")
ax = df.boxplot(column = ['CNNF&R', 'FoxNewsF&R'], figsize=(11,11))
ax.set_title("Comparing Number of Favorites and Retweets of two News")
ax.set_ylabel("Number of Favorites and Retweets")
ax.set_xlabel("Two News Resources")
plt.show()
