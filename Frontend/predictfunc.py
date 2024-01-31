#Import Libraries
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import io
from dtreeviz.trees import *
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier


# function for random forest importance inside a pipeline
# unsing n_estimor = 100
class RF_Feat_Selector(BaseEstimator, TransformerMixin):

    # class constructor
    # make sure class attributes end with a "_"
    # per scikit-learn convention to avoid errors
    def __init__(self, n_features_=15):
        self.n_features_ = n_features_
        self.fs_indices_ = None

    # override the fit function
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from numpy import argsort
        model_rfi = RandomForestClassifier(n_estimators=100)
        model_rfi.fit(X, y)
        self.fs_indices_ = argsort(model_rfi.feature_importances_)[::-1][0:self.n_features_]
        return self

    # override the transform function
    def transform(self, X, y=None):
        return X[:, self.fs_indices_]


# custom function to format the search results as a Pandas data frame
def get_search_results(gs):

    def model_result(scores, params):
        scores = {'mean_score': np.mean(scores),
             'std_score': np.std(scores),
             'min_score': np.min(scores),
             'max_score': np.max(scores)}
        return pd.Series({**params,**scores})

    models = []
    scores = []

    for i in range(gs.n_splits_):
        key = f"split{i}_test_score"
        r = gs.cv_results_[key]
        scores.append(r.reshape(-1,1))

    all_scores = np.hstack(scores)
    for p, s in zip(gs.cv_results_['params'], all_scores):
        models.append((model_result(s, p)))

    pipe_results = pd.concat(models, axis=1).T.sort_values(['mean_score'], ascending=False)

    columns_first = ['mean_score', 'std_score', 'max_score', 'min_score']
    columns = columns_first + [c for c in pipe_results.columns if c not in columns_first]

    return pipe_results[columns]

#Input: Dataframe containing user datapoint

class ExplainableDecisonTree:

    def __init__(self,model=None,train=None,column_names=[],X_train=None,Y_train=None):
        self.Model = model
        self.Traindata = train
        self.column_names = column_names
        self.X_train=X_train
        self.y_train=Y_train
        





    def Gettraindata(self):
        # Username of your GitHub account
        username = 'Mattjben'
        # Personal Access Token (PAO) from your GitHub account
        token = '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'
        # Creates a re-usable session object with your creds in-built
        github_session = requests.Session()
        github_session.auth = (username, token)
        # Downloading the csv file from your GitHub
        url = "https://raw.githubusercontent.com/IwVr/CSIDS-Finance/main/Datasets/heloc_dataset_v1.csv" # Make sure the url is the raw version of the file on GitHub
        download = github_session.get(url).content
        # Reading the downloaded content and making it a pandas dataframe
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))
        self.Traindata = df


    def model(self):
        df = self.Traindata
        df['RiskPerformance'] = pd.get_dummies(df['RiskPerformance'], drop_first=True, dtype=np.int64)
        X = df.drop("RiskPerformance", axis = 1)
        # Dropping the target variable
        X = X.values    # Changing into numpy array
        y = df["RiskPerformance"]   # storing target variable "RiskPerformance"
        y = y.values
        num_features = 23 # 24 minus 1 for the target Variable
        model_rf = RandomForestClassifier(n_estimators=100)
        model_rf.fit(X, y)
        fs_indices_rfi = np.argsort(model_rf.feature_importances_)[::-1][0:num_features]

        X_train = df.drop("RiskPerformance", axis = 1).values
        y_train = df["RiskPerformance"].values
        X_train_final = X_train[:, np.r_[fs_indices_rfi[0:10]]]
        self.X_train = X_train_final
        self.y_train=y_train 
        clf = DecisionTreeClassifier(random_state=999,max_depth=5,min_samples_split=2)
        print("------Fitting model--------")
        model = clf.fit(X_train_final, y_train)
        print("------Fitted--------")
        X = df.drop('RiskPerformance',axis=1)
        a=fs_indices_rfi[0:10]
        self.column_names = X.columns[a].to_list()
        self.Model =model


    def frontendpred(self,datapoint,model,user):
        text = explain_prediction_path(model, datapoint,
                                    feature_names=self.column_names,
                                    explanation_type="plain_english")
        text=text.strip()
        text=text.split("\n")
        text=[i.strip() for i in text]

        fig=dtreeviz(model, 
               x_data=self.X_train,
               y_data=self.y_train,
               target_name='RiskPerformance',
               feature_names=self.column_names, 
               class_names=['Good', 'Bad'], 
               title="Decision Tree",X=datapoint,show_just_path=True,orientation='LR')
        

        fig.save('Frontend/static/dtreeviz_'+str(user)+'.svg')
        return text
    def explainoutput(arrayin):
        explanationdict= {
        'ExternalRiskEstimate':	'External Risk Estimate',
    'MSinceOldestTradeOpen':'months since oldest approved credit agreement', 
    'MSinceMostRecentTradeOpen':'months since last approved credit agreement',
    'AverageMInFile':'average Months in File',
    'NumSatisfactoryTrades':'number of credit agreements on the customers credit bureau report with on-time payments',
    'PercentTradesNeverDelq':'Percentage of credit agreements on the customers credit bureau report with on-time payments',
    'NumTotalTrades':	'total number of credit agreements the customer has made' ,
    'PercentInstallTrades':	'percent of installment trades the customer has',
    'MSinceMostRecentInqexcl7days':	'months since most recent credit inquiry into the customers credit history (excluding the last 7 days)' ,
    'NetFractionRevolvingBurden':	 'customers revolving burden (portion of credit card spending that goes unpaid at the end of a billing cycle/credit limit)' ,
    'NetFractionInstallBurden':	'customers installment burden (portion of loan that goes unpaid at the end of a billing cycle/monthly instalment to be paid)' ,
    'PercentTradesWBalance'	:'number of trades currently not fully paid off by the customer'}

        line=['Explanation:\n']
        for i,value in enumerate(arrayin):
            range = re.search(r'(\d+\.?\d*) (?:<=|<|>=|>) (\w+)\s? (?:<=|<|>=|>) (\d+\.?\d*)',value)
            greater = re.search(r'^(\d+\.?\d*) (?:<=|<) (\w+)\s?$',value)
            lesser = re.search(r'^(\w+)\s? (?:<=|<) (\d+\.?\d*)$',value)
            if range:
                line.append('\n'+ str(i+1)+') The '+str(explanationdict[range.group(2)])+' was between '+str(range.group(1))+' and '+str(range.group(3)))
            if greater:
                line.append('\n'+ str(i+1)+') The'+str(explanationdict[greater.group(2)])+' was greater than or equal to '+str(greater.group(1)))
            if lesser:
                line.append('\n'+ str(i+1)+') The'+str(explanationdict[lesser.group(1)])+' was less than or equal to '+str(lesser.group(2)))
            return line

    def main(self,datapoint,user):
        self.Gettraindata()
        self.model()
        model = self.Model
        text= self.frontendpred(np.array(datapoint),model,user)
        value = model.predict(pd.DataFrame([datapoint]))
        return text,value #pickle.dump(model,open("model.pkl","wb"))
