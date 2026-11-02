
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def load_data():

    df = pd.read_csv('dataset.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'Yes').astype(int)

    return df

def train_model(df):

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.churn.values
    y_val = df_val.churn.values
    y_test = df_test.churn.values

    del df_train['churn']
    del df_val['churn']
    del df_test['churn']

    df_full_train = df_full_train.reset_index(drop=True)
    y_full_train = df_full_train.churn

    numerical = ['tenure','monthlycharges','totalcharges']

    categorical = [
        'gender', 
        'seniorcitizen', 
        'partner', 
        'dependents',
        'phoneservice', 
        'multiplelines', 
        'internetservice',
        'onlinesecurity', 
        'onlinebackup', 
        'deviceprotection', 
        'techsupport',
        'streamingtv', 
        'streamingmovies', 
        'contract', 
        'paperlessbilling',
        'paymentmethod'
    ]

    X = df_full_train[numerical + categorical]
    y = y_full_train


    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(max_iter=1000)
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring="roc_auc"
    )

    print("AUC de cada vuelta:", scores)
    print(f"AUC promedio: {scores.mean():.3f}")
    print(f"Desviación estándar: {scores.std():.3f}")


    param_grid = {
        "logisticregression__C": [0.01, 0.1, 1, 10, 100]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True
    )

    search.fit(X, y)

    print("Mejor C:", search.best_params_["logisticregression__C"])
    print(f"Mejor AUC promedio: {search.best_score_:.3f}")

    best_model = search.best_estimator_

    return best_model


def save_model(filename, best_model):
    with open(filename, "wb") as archivo:
        pickle.dump(best_model, archivo)
    print(f"model saved to {filename}")

df = load_data()
model = train_model(df)
save_model("churn_pipeline.pkl", model)