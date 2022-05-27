#Usando pycaret
#%%
print ('importando pacotes')
import sqlite3
from pycaret.regression import *
import pandas as pd
from sklearn import model_selection 


print("ok...")


#%% importanto bibliotecas

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
print('Data for Modeling: ' + str(df_train.shape))
print('Unseen Data For Predictions: ' + str(df_oot.shape))
print('ok...')
df_treino = pd.merge(X_train,y_train,left_on=None, right_on=None, left_index=True, right_index=True)

#%%
#%% Settando o ambiente pro pycaret
seila = setup(data = df_treino, target = 'rating', session_id=123)  #não sei pq 123

#%%
best = compare_models()

#como o melhor modelo aparentemente foi o light gradient boosting machine
#vamos treinar o modelo com ele

#%%

lgb = create_model('lightgbm')



#%%
#Tunando o modelo

tuned_lgb = tune_model(lgb)

#%%
#Plotando resultado do modelo
plot_model(tuned_lgb)


#%%


plot_model(tuned_lgb, plot = 'error')


#%%

plot_model(tuned_lgb, plot='feature')


#%%
#ou usar 
evaluate_model(tuned_lgb)

#%%
predict_model(tuned_lgb)


#%%
final_lgb = finalize_model(tuned_lgb)
#
#Final K Nearest Neighbour parameters for deployment
print(final_lgb)

#%%
#Testando na base out of time

unseen_predictions = predict_model(final_lgb, data=df_oot)
unseen_predictions.head()