from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=1234)

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

bayesian_params = {
    'max_depth': (6,16),
    'num_leaves': (24, 64),
    'min_child_samples': (10, 200),
    'min_child_weight': (1, 50),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'max_bin': (10, 500),
    'reg_lambda': (0.001, 10),
    'reg_alpha': (0.01, 50),
}

def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,
            subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {'n_estimators': 500, 'learning_rate' : 0.02,
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves),
        'min_child_samples': int(min_child_samples),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'max_bin': int(max_bin),
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
    }
    model = LGBMRegressor(**params)
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=100,
              early_stopping_rounds=100)
    y_pred = model.predict(x_test)
    results = r2_score(y_test, y_pred)
    return results

#####
lgb_bo = BayesianOptimization(lgb_hamsu, bayesian_params, random_state=1234)
lgb_bo.maximize(init_points=5, n_iter=30)
print(lgb_bo.max)