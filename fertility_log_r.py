import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

#Input variables
#Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1) nominal categorical data
#Age at the time of analysis. 18-36 (0, 1) 
#Childish diseases (ie , chicken pox, measles, mumps, polio)	1) yes, 2) no. (0, 1)  nominal categorical data
#Accident or serious trauma 1) yes, 2) no. (0, 1) nominal categorical data
#Surgical intervention 1) yes, 2) no. (0, 1) nominal categorical data
#High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1) ordinal categorical data
#Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1), ordinal categorical data 
#Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1) ordinal categorical data
#A Number of hours spent sitting per day ene-16	(0, 1) 
#Output: Diagnosis	normal (N), altered (O)	

#Predict y - will the diagnosis be normal or altered (binary)

def main():
    #downloaded the data from .csv file
    data_F=pd.read_csv('/users/alyonarodin/Desktop/fertility.csv')

    #deleted NA values
    print(data_F.isnull().sum())
    data_F=data_F.dropna()

    #printed the the first few rows of the table to see
    print(data_F.head())
    
    #example how logistic regrestic looks like, plotting sigmoid function
    #x values range from -10 to 10
    z = np.arange(-10, 10, 0.1)
    
    # plug into sigmoid function values x
    phi_z = sigmoid(z)
   
    #plot
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.axhline(y=0.0, ls='dotted', color='k')
    plt.axhline(y=1.0, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()
    
    #converted 'Yes' and 'No' values to 0 and 1, dropped Diagnosis column
    LE_Diagnosis = LabelEncoder()
    data_F["Diagnosis_code"] = LE_Diagnosis.fit_transform(data_F["Diagnosis"])
    data_F[["Diagnosis", "Diagnosis_code"]].head(11)
    data_F=data_F.drop(['Diagnosis'],axis=1)
    print(data_F)

    #to check if any column names has empty spaces that can give an error
    print ("<{}>".format(data_F.columns[9]))

    #placed Diagnosis_code to the first column
    D_C = data_F['Diagnosis_code']
    data_F.drop(labels=['Diagnosis_code'], axis=1,inplace = True)
    data_F.insert(0, 'Diagnosis_code', D_C)
    print(data_F.head())
    
    #placed Age to the second column
    A=data_F['Age']
    data_F.drop(labels=['Age'], axis=1,inplace = True)
    data_F.insert(1, 'Age', A)
    print(data_F.head())
    
    #assigned to X values which are not dummy 
    X = data_F[['Diagnosis_code','Age','High_fevers','Alcohol_consumption','Smoking','Sitting' ]]
    
    #prepared dummy and new dataset
    new_data=dummy_new_table(X,data_F)
    print(new_data.head())
    
    #checked for correlation
    corr=corr_fun(new_data)
    print(corr)
    
    #shuffled data
    new_data=shuffle_data(new_data)
    print(new_data.head())
    
    #converted to matrix and returned two matrixes A and b (A*x=b)
    data,target= matrix_data(new_data)
    
    #run logistic regressions
    log_reg(data,target)
    
    new_data2=new_data.drop(['Accident_trauma_0'],axis=1)
    new_data2=new_data2.drop(['Childish_diseases_0'], axis=1)
    new_data2=new_data2.drop(['Surgical_intervention_0'], axis=1)
    #print(new_data2)
    
    X_test_std2=log_reg2(new_data2)
    
    #created confusion matrix
    conf_mat(X_test_std2)
    
def shuffle_data(new_data):
    new_data = shuffle(new_data, random_state=0)
    return(new_data)
    
def dummy_new_table(X,data_F):
    dummy_Season=pd.get_dummies(data_F['Season'],prefix='Season')
    #print(dummy_Season)
    dummy_Childish_diseases=pd.get_dummies(data_F['Childish_diseases'],prefix='Childish_diseases')
    #print(dummy_Childish_diseases)
    dummy_Accident_trauma=pd.get_dummies(data_F['Accident_trauma'],prefix='Accident_trauma')
    #print(dummy_Accident_trauma)
    dummy_Surgical_intervention=pd.get_dummies(data_F['Surgical_intervention'],prefix='Surgical_intervention')
    #print(dummy_Surgical_intervention)

    new_data=pd.concat([X, dummy_Season,dummy_Childish_diseases,dummy_Accident_trauma,dummy_Surgical_intervention], axis=1)
    return(new_data)

def corr_fun(new_data):
    sns.heatmap(new_data.corr())
    corr = new_data.corr()
    return(corr)
    
def matrix_data(new_data) :   
    data_F_matrix = new_data.as_matrix()
    data=data_F_matrix[:,1:] 
    target=data_F_matrix[:,0]
    return (data, target)
    

def log_reg(data,target):  
    X = data

    y = target

    np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #print(X_train, X_test, y_train, y_test)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    LogisticRegression(penalty='l1')
    lr = LogisticRegression(penalty='l1', C=0.5)
    lr.fit(X_train_std, y_train)
    print('Training accuracy:', lr.score(X_train_std, y_train))
    print('Test accuracy:', lr.score(X_test_std, y_test))
    print(lr.intercept_)
    print(lr.coef_)

    lr.predict_proba(X_test_std[0,:])

    #first way to choose parameters 
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan','maroon','magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue','gray', 'indigo', 'orange','orchid']
    weights, params = [], []
    for c in np.arange(-4, 6,  dtype=float):
        lr=LogisticRegression(penalty='l1', C=10**c, random_state=0)
        lr.fit(X_train_std,y_train)
        weights.append(lr.coef_[0])
        params.append(10**c)
    
    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:,column],label=new_data.columns[column+1],color=color)

    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center',
    bbox_to_anchor=(1.38, 1.03),
    ncol=1, fancybox=True)
    plt.show()

def log_reg2(new_data2):

    sns.heatmap(new_data2.corr())
    corr = new_data2.corr()
    print(corr)

    data_F_matrix2 = new_data2.as_matrix()
    data2=data_F_matrix2[:,1:] 
    target2=data_F_matrix2[:,0]

    X2 = data2
    #print(X)

    y2 = target2

    np.unique(y2)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=0)
    #print(X_train, X_test, y_train, y_test)
    sc2 = StandardScaler()
    X_train_std2 = sc2.fit_transform(X_train2)
    X_test_std2 = sc2.transform(X_test2)
    
    lr2 = LogisticRegression(penalty='l1', C=0.5)
    lr2.fit(X_train_std2, y_train2)
    print('Training accuracy:', lr2.score(X_train_std2, y_train2))
    print('Test accuracy:', lr2.score(X_test_std2, y_test2))
    print(lr2.intercept_)
    print(lr2.coef_)
    predictions=lr2.predict_proba(X_test_std2)
    print(predictions)
    return(X_test_std2)

#function for confusion matrix
def conf_mat(X_test_std2):
    y_pred = lr2.predict(X_test_std2)
    confmat = confusion_matrix(y_true=y_test2, y_pred=y_pred)
    print(confmat)



    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

#function for sigmoid
def sigmoid(z):
   return (1.0 / (1.0 + np.exp(-z)))

main()







