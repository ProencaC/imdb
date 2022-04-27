# %%
# Importando bibliotecas necessarias
import re
from sys import displayhook
import time
import sqlite3
import pycountry
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.feature_extraction.text import CountVectorizer
import warnings
sns.set_theme(style = "whitegrid")


# %%
conn = sqlite3.connect("data/imdb.db")
tabelas = pd.read_sql_query("SELECT NAME AS 'Table_Name' FROM sqlite_master WHERE type = 'table'", conn)
tabelas.head(7)

# %%
# Vamos converter o dataframe em uma lista
tabelas = tabelas["Table_Name"].values.tolist()
#percorrendo a lista de tabelas no banco de dados 
# e extraindo o esquema de cada uma
for tabela in tabelas:
    consulta = "PRAGMA TABLE_INFO({})".format(tabela)
    resultado = pd.read_sql_query(consulta, conn)
    print("Esquema da tabela:", tabela)
    display(resultado)
    print("-"*100)
    print("\n")

#A tabela tb_abt_imdb foi criada com o sqlite
#Obs: tb_abt_imdb sera a tabela utilizada para prever os ratings
#Onde temos 
# title_id --------- é o ID do filme 
# filme_tv --------- é uma variavel binaria que sinaliza se o filme esta na categoria filmetv ou não (outra opção é apenas filme) 
# is_adult --------- variavel binária que sinaliza se o filme é para adultos 
# ano_estreia ------ variavel númerica que indica o Ano de Estreia do filme 
# tempo_duracao ---- tempo duração do filme em minutos 
# genero_nulo ------ variavel binária que sinaliza se o genero do filme é nulo (1) ou não (0) 
# numero_titulos --- Varivel númerica que sinaliza a quantidade de titulos diferentes que o filme teve (dependendo de em quantas 
#                    regiões ele foi assistido) 
# qt_crew Variavel - númerica que indica a quantidade de pessoas na equipe 
# num_votos número - de votos na nota do filme rating variavel --resposta, que indica a nota média do filme

#%%
# Cria a consulta SQL
consulta = '''SELECT * FROM tb_abt_imdb''' 
# Extrai o resultado
df = pd.read_sql_query(consulta, conn)
df.head()