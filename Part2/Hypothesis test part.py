import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as sm1
from scipy.stats import ttest_ind
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.multiclass import OneVsRestClassifier 
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

#Anova test for first hypothesis
data = pd.read_csv('Anovafile.csv')
mod = ols('Points ~ Group', data=data).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print (aov_table)
esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
print(esq_sm)


#Second hypothesis test
#To test the linear relationship between the txt blog points and Favorites and Retweets
#Test separately for CNN and FoxNews
cnn = pd.read_csv('CNN_test2.csv')
foxnews = pd.read_csv('FoxNews_test2.csv')
linear_model = sm1.ols(formula='Points ~ Favorites+Retweets',data=cnn)
results = linear_model.fit()
print(results.summary())
linear_model1 = sm1.ols(formula='Points ~ Favorites+Retweets',data=foxnews)
results1 = linear_model1.fit()
print(results1.summary())

#Third hypothesis test
#Using five predictve models to test

def function(myData):
    print(myData.head(20))
    print()

# Summary of data
    print(myData.describe())
    print()

# Look at the number of instances of each class
# class distribution
    print(myData.groupby('increase_rate').size())

# Box and whisker plots
    myData.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

# Histogram
    myData.hist()
    plt.show()

# Scatterplots to look at 2 variables at once
# scatter plot matrix
    scatter_matrix(myData)
    plt.show()

######################################################
# Evaluate algorithms
######################################################

# Separate training and final validation data set. First remove class
# label from data (X). Setup target class (Y)
# Then make the validation set 20% of the entire
# set of labeled data (X_validate, Y_validate)
    valueArray = myData.values
    X = valueArray[:,0:4]
    Y = valueArray[:,4]
    test_size = 0.20
    seed = 7
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Setup 10-fold cross validation to estimate the accuracy of different models
# Split data into 10 parts
# Test options and evaluation metric
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
        print(msg)

  
######################################################
# For the best model, see how well it does on the
# validation test.  This is for KNeighborsClassifier
######################################################
# Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validate)

    print()
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))

######################################################
# For the best model, see how well it does on the
# validation test. This is for DecisionTreeClassifier
######################################################
# Make predictions on validation dataset
    cart = DecisionTreeClassifier()
    cart.fit(X_train, Y_train)
    predictions = cart.predict(X_validate)

    print()
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))

######################################################
# For the best model, see how well it does on the
# validation test. This is for GaussianNB
######################################################
# Make predictions on validation dataset
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    predictions = nb.predict(X_validate)

    print()
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))

######################################################
# For the best model, see how well it does on the
# validation test. This is for SVM
######################################################
# Make predictions on validation dataset
    svm = SVC()
    svm.fit(X_train, Y_train)
    predictions = svm.predict(X_validate)

    print()
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))

######################################################
# For the best model, see how well it does on the
# validation test. This is for RandomForestClassifier
######################################################
# Make predictions on validation dataset
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_validate)

    print()
    print(accuracy_score(Y_validate, predictions))
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))

def function2(myData):
# Separate training and final validation data set. First remove class
# label from data (X). Setup target class (Y)
# Then make the validation set 20% of the entire
# set of labeled data (X_validate, Y_validate)
    valueArray = myData.values
    X = valueArray[:,0:4]
    Y = valueArray[:,4]
    test_size = 0.20
    seed = 7


#Binarize the class
    Y = label_binarize(Y, classes=['increase', 'decrease'])
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
#Add each algorithm and its name to the model array
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF',RandomForestClassifier()))
    models.append(('SVM', SVC()))
    
#Predict using each method and plot the roc curve
    for name, model in models[:4]:
        classifier=OneVsRestClassifier(model)
        classifier.fit(X_train, Y_train)
        scorey=classifier.predict_proba(X_validate)
        plt.figure()
        xx = []
        yy = []
        xx,yy,_=roc_curve(Y_validate,scorey[:,1])
        plt.plot(xx,yy)
        plt.plot(np.linspace(0,1,50),np.linspace(0,1,50))
        title1=name+' '+'ROC Curve'
        plt.title(title1)
#Use decision function For SVM case
    classifier=OneVsRestClassifier(SVC())  
    classifier.fit(X_train, Y_train)
    socrey=classifier.decision_function(X_validate)
    plt.figure()
    xx = []
    yy = []
    xx,yy,_=roc_curve(Y_validate,socrey)
    plt.plot(xx,yy)
    plt.plot(np.linspace(0,1,50),np.linspace(0,1,50))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    title1='SVM'+' '+'ROC Curve'
    plt.title(title1)



#read the file and apply the both functions to get the result for the third hypothesis
d1 = pd.read_csv('News_vs_Gspc.csv')
d2 = pd.read_csv('News_vs_Gold.csv')
d3 = pd.read_csv('News_vs_Ixic.csv')
function(d1)
function2(d1)
function(d2)
function2(d2)
function(d3)
function2(d3)
