# %%
print ('importando pacotes')
import sqlite3
import requests 
import pandas as pd
import numpy as np
#import time 
import pickle #salvar modelo
import joblib #pipelines leves
from sklearn.ensemble import RandomForestClassifier #modelagem
from sklearn.ensemble import GradientBoostingClassifier #modelagem
#import re #This module provides regular expression matching operations similar to those found in Perl
#import os #funções com o sistema operacional
#import glob 
from sklearn import model_selection 
from sklearn.model_selection import GridSearchCV #cross validation do modelo
import xgboost as xgb #modelo
from xgboost.sklearn import XGBClassifier #modelagem

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = "whitegrid")

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn import pipeline
from sklearn import metrics

from feature_engine import imputation
from feature_engine import encoding
from feature_engine import outliers

print("ok...")

#%%
print("Carregando ABT")
#Criando conexão
conn = sqlite3.connect("data/imdb.db")
# Cria a consulta SQL
consulta = '''SELECT * FROM tb_abt_imdb''' 
# Extrai o resultado
df = pd.read_sql_query(consulta, conn)
print(df.shape) #(213348, 37)

print("ok...")


#%%
print("Criando target")
df.loc[(df.rating < 5),"target"] = "Ruim"
df.loc[(df.rating >=5) & (df.rating < 7),"target"] = "Medio"
df.loc[(df.rating >=7),"target"] = "Bom"

print("ok...")

#%%print("Separando base oot, base para treino e base para teste")
#Nosso back-test
df_oot = df[df["ano_estreia"] == 2022].copy() #Base out of time
df_train = df[df["ano_estreia"] != 2022].copy() #Base treino

features = df_train.columns.tolist()[1:-2]
target = 'target'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train[features],
                                                                    df_train[target],
                                                                    random_state=42,
                                                                    test_size=0.3)

print('ok...')
#%%
print("separando variaveis")
dummy_features = [
 'genero_action',
 'genero_adult',
 'genero_adventure',
 'genero_animation',
 'genero_biography',
 'genero_comedy',
 'genero_crime',
 'genero_documentary',
 'genero_drama',
 'genero_family',
 'genero_fantasy',
 'genero_filmnoir',
 'genero_gameshow',
 'genero_history',
 'genero_horror',
 'genero_music',
 'genero_musical',
 'genero_mystery',
 'genero_news',
 'genero_nulo',
 'genero_realitytv',
 'genero_romance',
 'genero_scifi',
 'genero_short',
 'genero_talkshow',
 'genero_thriller',
 'genero_war',
 'genero_western']
num_features = list(set(X_train.columns) - set(dummy_features))

print("ok...")

#%%
print("Missing numerico")
is_na = X_train[num_features].isna().sum()
print(is_na[is_na>0])
#Notamos valores nulos na variavel resposta
#vou retirar essas linhas diretamente na querry da abt
#Restaram 27 anos de estreia nulos que serão substituidos por -1
#E 24948 tempos de duração que serão substituidos pela média da categoria
#(filme TV ou filme)
#Assim ficamos com 213.348 linhas

#%%
print("Imputando missings")
missing1 = ["ano_estreia",
            "numero_titulos"]
missingmedian = ['tempo_duracao'] #Vou usar a mediana ao inves da media pois temos valores muito extremos no tempo de duração


# MODIFY - modificando os valores nulos

## imputação de dados
imput_1 = imputation.ArbitraryNumberImputer(arbitrary_number=-1, variables=missing1)
imput_median = imputation.MeanMedianImputer(imputation_method = "median",variables = missingmedian)

outlier_cut = outliers.ArbitraryOutlierCapper (max_capping_dict={"tempo_duracao":300},
                                              min_capping_dict={"num_votos":30})

print("ok...")

#%%
#MODEL
print("Iniciando modelos")
##################################################################################################
###########################################Random Forest##########################################
##################################################################################################

param_grid_rf = {
'max_depth': [50 , 75, 100],
'max_features': ['auto'],
'min_samples_leaf': [0.01,0.03,0.05],
'min_samples_split': [0.01,0.03,0.05],
'n_estimators': [100,150]
                
}
    
rf=RandomForestClassifier()
    
    # Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf,
                            scoring=["accuracy"],
                            cv = 4, n_jobs = 1, verbose = 3,
                            return_train_score=True,
                            refit='accuracy')

#Define a pipeline
rf_pipe = pipeline.Pipeline(steps=[   ("Imput -1", imput_1),
                                      ("Imput mediana", imput_median),
                                      ("Outliers cut", outlier_cut),
                                      ("Modelo", grid_search_rf)] )

rf_pipe.fit(X_train,y_train)

#%%

print('Gradient boosting')
#################################################################################################
########################################Gradient Boosting#########################################
##################################################################################################
    
    #Parametros do gardient boosting    
param_grid_gb= {
                'max_depth': [50 , 75,  100],
                'max_features': ['auto'],
                'min_samples_leaf': [10,30,50],
                'min_samples_split': [10,30,50],
                'n_estimators': [100,150]
                }

gb=GradientBoostingClassifier()
    
# Instantiate the grid search model
grid_search_gb = GridSearchCV(estimator = gb, param_grid = param_grid_gb, 
                              scoring=['accuracy'],
                              cv = 4, n_jobs = 1, verbose = 3,
                              return_train_score=True,
                              refit='accuracy')

#Define a pipeline
gb_pipe = pipeline.Pipeline(steps=[   ("Imput -1", imput_1),
                                      ("Imput mediana", imput_median),
                                      ("Outliers cut", outlier_cut),
                                      ("Modelo", grid_search_gb)] )

#Fit modelo
gb_pipe.fit(X_train,y_train)
print('fim')

#%%



print("XGB")
##################################################################################################
############################################XBoosting#############################################
##################################################################################################
#Parametros do gardient boosting    
param_grid_xb = {
                'booster':['gbtree'],
                'objective':['multi:softprob'],
                'eval_metric':['accuracy'],
                'eta': [0.1,0.2,0.3,0.5],
                'gamma':  [0,0.25,0.5],
                'max_depth': [4,6,8,20],
                'min_child_weight' : [1]
                }
xb=XGBClassifier()
    

# Instantiate the grid search model
grid_search_xb = GridSearchCV(estimator = xb, param_grid = param_grid_xb, 
                              scoring=['accuracy'],
                              cv = 4, n_jobs = 1, verbose = 3,
                              return_train_score=True,
                              refit='accuracy')

#Define a pipeline
xb_pipe = pipeline.Pipeline(steps=[   ("Imput -1", imput_1),
                                      ("Imput mediana", imput_median),
                                      ("Outliers cut", outlier_cut),
                                      ("Modelo", grid_search_xb)] )

xb_pipe.fit(X_train,y_train)


#%%

rf_y_train_pred = rf_pipe.predict(X_train)
rf_y_train_proba_pred = rf_pipe.predictp(X_train)

#gb_y_train_pred = gb_pipe.predict(X_train)
#xb_y_train_pred = xb_pipe.predict(X_train)



acuracia_rf = metrics.accuracy_score(y_train, rf_y_train_pred)
#acuracia_gb = metrics.accuracy_score(y_train,y_pred=gb_y_train_pred)
#acuracia_xb = metrics.accuracy_score(y_train,y_pred=xb_y_train_pred)
print(acuracia_rf)
precision_rf = metrics.precision_score(y_train, rf_y_train_pred,average=None)
#acuracia_gb = metrics.precision_score(y_train,y_pred=gb_y_train_pred)
#acuracia_xb = metrics.precision_score(y_train,y_pred=xb_y_train_pred)
print(precision_rf)

recall_rf = metrics.recall_score(y_train, rf_y_train_pred,average=None)
#acuracia_gb = metrics.recall_score(y_train,y_pred=gb_y_train_pred)
#acuracia_xb = metrics.recall_score(y_train,y_pred=xb_y_train_pred)
print(recall_rf)