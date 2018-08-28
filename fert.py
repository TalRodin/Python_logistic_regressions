import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
    #print(data_F.isnull().sum())
    #data_F=data_F.dropna()
    
    #save table for Tableau in the json format
    table_fert=data_F.to_json(orient='table')
    outfile = open("/Users/alyonarodin/Desktop/Python_log_project/table_fert.json", "w")
    outfile.write(table_fert)
    outfile.close()
    
    #funtion of plotting amount N vs O (binary outcome)
    count_N,count_O=plot_N_O(data_F)
    
    #function that adds data based on the results of previous function plot_N_O().
    #if count_N > count_O then the N value (one of the vinary outcome we try to predict based on the 
    # variables we have) will be added otherwise O value. Added into seperate list each column.
    #Separated N value from O value.
    Season=add_data(data_F['Season'],count_N,count_O,data_F)
    Age=add_data(data_F['Age'],count_N,count_O,data_F)
    Childish_diseases=add_data(data_F['Childish diseases'],count_N,count_O,data_F)
    Accident=add_data(data_F['Accident'],count_N,count_O,data_F)
    Surgical_intervention=add_data(data_F['Surgical intervention'],count_N,count_O,data_F)
    High_fevers=add_data(data_F['High fevers'],count_N,count_O,data_F)
    Frequency=add_data(data_F['Frequency'],count_N,count_O,data_F)
    Smoking=add_data(data_F['Smoking'],count_N,count_O,data_F)
    sitting=add_data(data_F['sitting'],count_N,count_O,data_F)
    Diagnosis=add_data(data_F['Diagnosis'],count_N,count_O,data_F) 
    
    #combines the lists we created 
    df_O_data=pd.DataFrame(list(zip(Season,Age,Childish_diseases,Accident,Surgical_intervention,High_fevers,Frequency,Smoking,sitting,Diagnosis)),columns=['Season','Age','Childish diseases','Accident','Surgical intervention','High fevers','Frequency','Smoking','sitting','Diagnosis'])
   
    #checking if we have NA values
    #print(df_O_data.isnull().sum())
    
    #concated into one table the initial table with additional values we created based on what value is greater
    # N or O
    n_data=pd.concat([data_F, df_O_data,df_O_data,df_O_data,df_O_data,df_O_data,df_O_data])
    #print(n_data.isnull().sum())
    
    #shuffled data
    new_data=shuffle_data(n_data)
    #print(new_data.isnull().sum())
    
    #we used again the function plot_N_O to see if it looks better and the outcome approximately the 
    #same, otherwise the results will be leaning towards the the value that has the greater number
    count_N,count_O=plot_N_O(new_data)
    
    #Converted diagnoses into values 1 and 0
    data_F=prep_data(new_data)
    #print(data_F.isnull().sum())    
    
    #preparing dummy values
    X = data_F[['Diagnosis_code','Age','High fevers','Frequency','Smoking','sitting' ]]
    
    #prepared dummy and new dataset
    #print(X.dtypes)
    new_data=dummy_new_table(X,data_F)
    #print(new_data.isnull().sum())
    
    #checked for correlation
    corr=corr_fun(new_data)
    print(corr)
    
    #dropped few columns to eliminate the multicolinearity due to dummy variabels
    new_data2=new_data.drop(['Accident_trauma_0'],axis=1)
    new_data2=new_data2.drop(['Childish_diseases_0'], axis=1)
    new_data2=new_data2.drop(['Surgical_intervention_0'], axis=1)
    new_data2=new_data2.drop(['Season_-1.0'], axis=1)
    
    #seperated data into X (all rows and columns from 2 and furhter)
    #y dataset contains the diagnoses values only
    X, y=new_data2.iloc[:,1:].values, new_data2.iloc[:,0].values
    
    #one more time to check correlation to make sure there is no multicollinearity
    new_corr=corr_fun(new_data2)
    print(new_corr)
    
    #splitted dataset into test and train as 0.7/0.3
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=0)
    
    #transfered values into standard scaler 
    stdsc=StandardScaler()
    X_train_std=stdsc.fit_transform(X_train)
    X_test_std=stdsc.transform(X_test)
    
    
    
    #runned logistic regressions on training data
    #penalty --> l1 regularization
    # c=0.2 regularization parameter 
    lr=LogisticRegression(penalty='l1', C=0.2)
    lr.fit(X_train_std, y_train)
    print('Training accuracy:', lr.score(X_train_std, y_train))
    print('Test accuracy:', lr.score(X_test_std, y_test))
    #shows intercepts and coefficients
    print(lr.intercept_)
    print(lr.coef_)
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan','maroon','magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue','gray', 'indigo', 'orange','orchid']
    weights, params = [], []
    for c in np.arange(-5, 6,  dtype=float):
        lr=LogisticRegression(penalty='l1', C=10**c, random_state=0)
      
        lr.fit(X_train_std,y_train)
        weights.append(lr.coef_[0])
        params.append(10**c)
    
    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:,column],label=new_data2.columns[column+1],color=color)

    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.xlim([10**(-3), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center',bbox_to_anchor=(1.38, 1.03),ncol=1, fancybox=True)
    plt.show()
    
    feat_labels=new_data2.columns[1:]
    forest=RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(X_train_std, y_train)
    importances=forest.feature_importances_
    indices=np.argsort(importances)[::-1]
    for f in range(X_train_std.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,feat_labels[f],importances[indices[f]]))

    plt.title('Feature Importances')    
    plt.bar(range(X_train_std.shape[1]),importances[indices], color='lightblue', align='center')  
    plt.xticks(range(X_train_std.shape[1]),feat_labels, rotation=90)
    plt.xlim([-1, X_train_std.shape[1]])    
    plt.tight_layout()   
    plt.show()
    
    #confusion matrix
    lr_pred_prob=lr.predict_proba(X_test_std)
    
    print(np.around(lr_pred_prob, decimals=3))
    
    y_pred = lr.predict(X_test_std)
    
    print(y_pred)
    
    print(y_test)
    
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Greens, alpha=0.5)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
    
    
def plot_N_O(data_F):
    
    count_N=0
    count_O=0
    for i in data_F['Diagnosis']:
        if i=='N':
            count_N+=1        
        else:
            count_O+=1
    print('N: ',count_N)
    print('O: ',count_O)    
    
    #or

    data_F['Diagnosis'].value_counts()
    sns.countplot(x='Diagnosis', data=data_F, palette="coolwarm")
    plt.show()
    return(count_N,count_O)

#since we have unequival data - outcome - solution to increse the data of N
#we increase by 7 times
def add_data(x,count_N,count_O,data_F):
    list_=[]
    if count_N > count_O:
        for i in range(len(data_F['Diagnosis'])):
            if data_F['Diagnosis'][i]=='O':
                list_.append(x[i])
    if count_O>count_N:
        for i in range(len(data_F['Diagnosis'])):
            if data_F['Diagnosis'][i]=='N':
                list_.append(x[i])
    return(list_)

def shuffle_data(new_data):
    new_data = shuffle(new_data, random_state=0)
    return(new_data)

def prep_data(data_F):
    #converted Diagnosis values 'Yes' and 'No' to 0 and 1, dropped Diagnosis column
    LE_Diagnosis = LabelEncoder()
    data_F["Diagnosis_code"] = LE_Diagnosis.fit_transform(data_F["Diagnosis"])
    data_F[["Diagnosis", "Diagnosis_code"]].head()
    data_F=data_F.drop(['Diagnosis'],axis=1)
    print(data_F.head())

    #placed Diagnosis_code to the first column
    D_C = data_F['Diagnosis_code']
    data_F.drop(labels=['Diagnosis_code'], axis=1,inplace = True)
    data_F.insert(0, 'Diagnosis_code', D_C)
    #print(data_F.head())
    
    #placed Age to the second column
    A=data_F['Age']
    data_F.drop(labels=['Age'], axis=1,inplace = True)
    data_F.insert(1, 'Age', A)
    #print(data_F.head())
    return(data_F)

def dummy_new_table(X,data_F):
    dummy_Season=pd.get_dummies(data_F['Season'],prefix='Season')
    #print(dummy_Season)
    dummy_Childish_diseases=pd.get_dummies(data_F['Childish diseases'],prefix='Childish_diseases')
    #print(dummy_Childish_diseases)
    dummy_Accident_trauma=pd.get_dummies(data_F['Accident'],prefix='Accident_trauma')
    #print(dummy_Accident_trauma)
    dummy_Surgical_intervention=pd.get_dummies(data_F['Surgical intervention'],prefix='Surgical_intervention')
    #print(dummy_Surgical_intervention)

    new_data=pd.concat([X, dummy_Season,dummy_Childish_diseases,dummy_Accident_trauma,dummy_Surgical_intervention], axis=1)
    return(new_data)  
   


def corr_fun(new_data):
    sns.heatmap(new_data.corr())
    corr = new_data.corr()
    return(corr) 



main()    
    
    