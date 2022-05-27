# %%
print ('importando pacotes')
#import time

#from asyncio.windows_utils import pipe
import sqlite3
import math
from statistics import mean
#import pycountry

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex
#import seaborn as sns
#sns.set_theme(style = "whitegrid")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
from sklearn import pipeline
from sklearn import metrics
import xgboost as xgb

from feature_engine import imputation
from feature_engine import encoding
from feature_engine import outliers

from scipy.stats import jarque_bera

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



# %%

print("Separando base oot, base para treino e base para teste")
#Nosso back-test
df_oot = df[df["ano_estreia"] == 2022].copy() #Base out of time
df_train = df[df["ano_estreia"] != 2022].copy() #Base treino

features = df_train.columns.tolist()[1:-1]
target = 'rating'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train[features],
                                                                    df_train[target],
                                                                    random_state=42,
                                                                    test_size=0.3)

# %%

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

#tratando outliers


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
rf_rgs = ensemble.RandomForestRegressor(n_estimators=80,
                                        min_samples_leaf=20,
                                        random_state=42)

#ada_rgs = ensemble.AdaBoostRegressor(n_estimators=200,
#                                      learning_rate=0.8,                                        
#                                      random_state=42)

#dt_rgs = tree.DecisionTreeRegressor(max_depth=15,
#                                     min_samples_leaf=50,
#                                     random_state=42)

xgb_rgs = xgb.XGBRegressor(tree_method="gpu_hist")

#reglin = linear_model.LinearRegression()
print("ok...")

#%%
#Definir pipeline
print("Definindo pipelines")
rf_pipe = pipeline.Pipeline(steps=[ ("imput -1", imput_1),
                                      ("imput mediana", imput_median),
                                      ("Outliers cut", outlier_cut),
                                      ("modelo", rf_rgs)] )

#ada_pipe = pipeline.Pipeline(steps=[ ("imput -1", imput_1),
#                                      ("imput mediana", imput_median),
#                                      ("modelo", ada_rgs)] )

#dt_pipe = pipeline.Pipeline(steps=[ ("imput -1", imput_1),
#                                      ("imput mediana", imput_median),
#                                      ("modelo", dt_rgs)] )

xgb_pipe = pipeline.Pipeline(steps=[ ("imput -1", imput_1),
                                      ("imput mediana", imput_median),
                                      ("Outliers cut", outlier_cut),
                                      ("modelo", xgb_rgs)] )
#rl_pipe = pipeline.Pipeline(steps=[ ("imput -1", imput_1),
#                                      ("imput mediana", imput_median),
#                                      ("modelo", reglin)] )    

models = {"Random Forest":rf_pipe,
          #"AdaBoost": ada_pipe,
          #"Decision Tree": dt_pipe,
          "XgBoost": xgb_pipe
          #,"Linear Regression": rl_pipe
           }

print("ok...")
#%%
print("Definindo função que retorna a metrica")
def train_test_report(model, X_train, ytrain, X_test, y_test, key_metric):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metric_result = key_metric(y_test, pred)
    return metric_result
print("ok...")

#%%
print("Testando todos modelos")
#for d,m in models.items():
#    result = train_test_report(m,X_train, y_train, X_test, y_test, metrics.r2_score)
#    print(f"{d}: {result}")


#Primeiros resultados
#Random Forest: 0.3757604197400064
#AdaBoost: 0.046989319553030984
#Decision Tree: 0.3382501745745554
#XgBoost: 0.3845171431009107 O melhor por enquanto
#Linear Regression: 0.26347884521853127

#Assim decidi tunar o XGBoost e o Random forest pois são os mais promissores
#Portanto vou comentar as linhas com os outros modelos
print("ok...")


#%%
#otimizando/tunando os modelos de Random forest e XGBoost utilizando o gridsearch
#Primeiro o random forest
print("configurando gridsearch")
params = {"n_estimators":[50,100,150,200],
          "min_samples_leaf":[5,10,20,50]}

grid_search = model_selection.GridSearchCV(rf_rgs,
                                           params,
                                           n_jobs=1,
                                           cv = 4,
                                           scoring=["r2","explained_variance",
                                           'neg_mean_squared_error','neg_root_mean_squared_error'],
                                           verbose=3,
                                           return_train_score=True,
                                           refit='neg_mean_squared_error')

rf_pipe = pipeline.Pipeline(steps=[   ("Imput -1", imput_1),
                                      ("Imput mediana", imput_median),
                                      ("Outliers cut", outlier_cut),
                                      ("Modelo", grid_search)] )

print("ok...")
#%%
print("Fitando o modelo")
rf_pipe.fit(X_train,y_train)
#Fitting 5 folds for each of 16 candidates, totalling 80 fits
print("Ok...")


#%%
grid_search.best_estimator_
#RandomForestRegressor(min_samples_leaf=10, n_estimators=200, random_state=42)
#%%
print("Medindo erro na base treino")
y_train_pred = rf_pipe.predict(X_train)

mse = metrics.mean_squared_error(y_train,y_train_pred)
rmse = math.sqrt(mse)
R2 = metrics.r2_score(y_train,y_train_pred)

print("mse:",mse)
print("rmse:",rmse)
print("R2:",R2)

print("ok...")

#%%
#Testando na base de teste
print("medindo erro na base  teste")
y_test_pred = rf_pipe.predict(X_test)

mse_t = metrics.mean_squared_error(y_test,y_test_pred)
rmse_t = math.sqrt(mse)
R2_t = metrics.r2_score(y_test,y_test_pred)

print("mse:",mse_t)
print("rmse:",rmse_t)
print("R2:",R2_t)

print("ok...")

#%%
#Agora o XGB



