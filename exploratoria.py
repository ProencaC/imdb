#%%
print ('importando pacotes')
import time
import sqlite3
#import pycountry

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = "whitegrid")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
from sklearn import pipeline
from sklearn import metrics

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
print(df.head())

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



#Aqui foi separado um tempo para pensar nos filtros usados no ETL
#Então não utilizamos dados que não perteciam ao genero filme
#Não queremos dados que possuem a variavel resposta nula
#Filmes com mais de 6 horas não pertencem a essa análise
#Número minimo de votos no filme igual a 20



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
#ANALISE EXPLORATORIA
plt1 = sns.histplot(x = df.filme_tv)
#Concluimos que temos muito mais filmes normais que filmes de TV

#%%
plt2 = sns.histplot(x = df.is_adult)
#é bizarra a quantidade de filmes adultos, quase não existe
#O que é filme adulto?

#%%
plt3 = sns.histplot(x = "ano_estreia",data = df)
plt3.set_xlabel("Ano de estreia")
plt3.set_ylabel("Quantidade de filmes")
plt3.set_title("Histograma dos filmes lançados por ano")

#nota-se uma crescente em relação ao numero de filmes produzidos com o avançar dos anos,
#que teve uma queda no ano de 2020 provavelmente devido a pandemia.


#%%
plt4 = plt.hist(x = df.tempo_duracao)
plt.xlabel("Tempo de duração")
plt.ylabel("Quantidade")
plt.title("Histograma do tempo de duração dos filmes")
df.tempo_duracao.describe()
#notamos que pelo menos metade dos filmes estão entre 80 e 100 minutos
#(antigo)notamos um filme com 51420 minutos, trata-se de um outlier, porem não é um erro,
#realmente existe um filme com 857 horas. Porem que sera descartado da nossa análise
#pois não é nosso objetivo prever a nota de um caso tão extremo. Buscamos a generalização do modelo.


#%%
plt5 = sns.histplot(x = "rating", data = df)
plt5.set_xlabel("Notas")
plt5.set_ylabel("Quantidade")
plt5.set_title("Histograma das notas")


#%%
plt6 = plt.bar (x = "num_votos", data = df,bins=20)
plt.xlabel("numero votos")
plt.ylabel("Quantidade")
plt.title("Histograma da quantidade de votos")
#Outliers não deixam visualizar os outros valores :(



#%%
plt6 = sns.histplot(x = df.genero_nulo)
#poucos valores com genero nulo, parecia ser um maior numero, quando criei a abt

#%%
plot7 = plt.hist(x = df.numero_titulos)

#%%
plot9 = sns.boxplot(x = df[df.num_votos > 50].num_votos)

df[df.num_votos > 50].num_votos.describe()

#%%
corr = df[num_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#Não notamos nenhuma correlação muito grande, mas começamos a entender os dados.


#%%

#Observando relação das variveis com a variavel resposta
plt.scatter(df.ano_estreia,df.rating)
#%%
#Não parece haver relação

plt.scatter(df.tempo_duracao,df.rating)


#%%
plt.hist(df.rating)


#%%
#Salvando dados em csv 



#Não temos uma normal, logo para aplicar regressão linear precisariamos
#fazer uma transformação nos dados.

# nota <5 nota 5 e 8 médio nota > 8 muito pia
