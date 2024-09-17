import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    report = {}

    for i in range(len(list(models))):
        model_name= (list(models.keys())[i])
        model = (list(models.values())[i])
        param= params[list(models.keys())[i]]
        gs= GridSearchCV(model,param_grid=param,cv=3,n_jobs=-1)
        gs.fit(x_train,y_train)

        model.set_params(**gs.best_params_)
        best_param=gs.best_params_
        model.fit(x_train,y_train)

        score= accuracy_score(y_test,model.predict(x_test))

        report[list(models.keys())[i]]=(score,gs.best_params_)
        
    return report


class ModelTrainer:
    def initiate_model_training(self,train_arr,test_arr):
        x_train,y_train,x_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
        models= {
                "Ada Boost Classifier": AdaBoostClassifier(),
                "K Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(verbose=0),
                "XGB Classifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier" : SVC()
            }

        params= {
                "Ada Boost Classifier":{
                    "n_estimators":[8,16,32,64,100,128,256],
                    "learning_rate":[0.01,0.1,0.5,0.7,1],
                },
                "K Neighbors Classifier":{
                    "n_neighbors":[3,5,7],
                    "weights":['distance','uniform'],
                },
                "Decision Tree Classifier":{
                    "criterion":["gini", "entropy", "log_loss"],
                    "splitter":['best','random']
                },
                "Random Forest Classifier":{
                    "n_estimators" :[8,16,32,64,100,128,256],
                    "criterion":["gini", "entropy", "log_loss"],
                    "max_features":['sqrt','log2']
                },
                "XGB Classifier":{
                    "booster":['gbtree','gblinear','dart'],
                    "learning_rate":[0.01,0.05,0.1],
                    'n_estimators': [8,16,32,64,128,256]

                },
                "CatBoosting Classifier":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations': [30, 50, 100]
                },
                'Logistic Regression':{   
                },
                "Support Vector Classifier":{
                }
            }

        report:dict= evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

        scores=[report[key][0] for key in report.keys()]
        best_score=max(sorted(scores))
        best_param=[report[key][1] for key in report.keys() if report[key][0]==best_score] 
        best_model=[key for key in report.keys() if report[key][0]==best_score]
        return best_model,best_param
    
if __name__=='__main__':
    df = pd.read_csv(r'Dataset\Heart_minor.csv',index_col=0)
    train_arr,test_arr=train_test_split(df,test_size=0.2,random_state=69)
    mdr = ModelTrainer()
    train_arr=np.array(train_arr)
    test_arr=np.array(test_arr)
    print(mdr.initiate_model_training(test_arr=test_arr,train_arr=train_arr))