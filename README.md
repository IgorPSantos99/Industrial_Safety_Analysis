# Modelo de Previs�o de Acidentes em Ind�strias

## 1. **Introdu��o**

Os acidentes de trabalho s�o ocorr�ncias indesejadas que podem causar les�es corporais ou doen�as aos trabalhadores durante o exerc�cio de suas atividades. Esses incidentes podem ocorrer devido a v�rias raz�es, como condi��es inseguras no local de trabalho, falta de treinamento, equipamentos inadequados e outras neglig�ncias.
A import�ncia de prever e prevenir esses acidentes � multifacetada e impacta diversos aspectos:

* **Sa�de e Seguran�a dos Trabalhadores**: Os acidentes em ind�stria colocam em risco a sa�de e a seguran�a dos trabalhadores. Quedas de altura, choques el�tricos, quedas de objetos, cortes, contus�es e amputa��es s�o alguns dos acidentes mais comuns. Essas situa��es n�o apenas afetam a integridade f�sica dos trabalhadores, mas tamb�m podem resultar em doen�as cr�nicas ou at� mesmo mortes.

* **Impacto Econ�mico**: Os custos associados aos acidentes de trabalho s�o significativos. Eles geram preju�zos para os trabalhadores, as empresas, a previd�ncia social e a economia como um todo. Vamos explorar esses custos em detalhes:

    - **Custos Diretos**:
        - Despesas m�dicas, odontol�gicas e hospitalares para atender o acidentado.
        - Cirurgias reparadoras e reabilita��o m�dica e ocupacional.
        - Transporte do acidentado durante o tratamento.
        - Seguro de acidente.
    
    - **Custos Indiretos**:
        - Sal�rios pagos durante o tempo perdido por outros trabalhadores que n�o o acidentado.
        - Atrasos na produ��o devido ao acidente.
        - Sal�rios adicionais por horas extras para compensar a produ��o perdida.
        - Custos com treinamento do substituto do acidentado.
        - Danos a m�quinas ou equipamentos envolvidos no acidente.
        - Despesas m�dicas n�o cobertas pela seguradora.
        - Diminui��o da efici�ncia do acidentado ao retornar ao trabalho.

* **Legalidade e Reputa��o**: Empresas que negligenciam a seguran�a no trabalho enfrentam custos legais, multas e processos trabalhistas. Al�m disso, a reputa��o da empresa pode ser afetada negativamente, afastando clientes e investidores.

* **Produtividade e Lucro**: Investir em seguran�a do trabalho reduz acidentes, diminui custos e aumenta a produtividade. Funcion�rios saud�veis e seguros s�o mais produtivos, e uma cultura de seguran�a melhora a reputa��o da empresa.

Em nosso estudo, abordaremos t�picos como a an�lise dos setores mais propensos a acidentes, a distribui��o temporal desses eventos, a influ�ncia de categorias espec�ficas e a constru��o de um modelo de previs�o de acidentes futuros. Acreditamos que essa pesquisa contribuir� para a promo��o de ambientes de trabalho mais seguros e saud�veis.

---

## 2. **�ndice**

1. [Introdu��o]()
2. [�ndice]()
3. [Fonte]()
4. [Habilidades Adquiridas]()
5. [Execu��o do C�digo]()


---

## 3. **Fonte**

Para este estudo, foi utilizado o dataset disponibilizado no Kaggle por IHM Stefanini em 
https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database


## 4. **Habilidades Adquiridas**

Durante a execu��o do projeto, foram adquiridas diversas compet�ncias essenciais para a manipula��o, an�lise e modelagem de dados relacionados a acidentes de trabalho. Abaixo est�o destacadas as principais habilidades adquiridas e as aprendizagens resultantes de cada uma:

* **Manipula��o de Dados com Pandas:**
   - **Exporta��o e Modifica��o de Dados:** Foi explorada a funcionalidade do Pandas para exportar dados e efetuar modifica��es, como a altera��o de t�tulos de colunas e ajuste de tipos de dados.
   - **Limpeza de Dados:** Desenvolvemos habilidades na identifica��o e tratamento de valores NaN, aprimorando a qualidade dos conjuntos de dados.

* **An�lise Explorat�ria de Dados:**
   - **Contagens e Gr�ficos com Seaborn:** Utilizamos a biblioteca Seaborn para explorar a contagem de ocorr�ncias e criar gr�ficos que forneceram insights sobre as categorias com maior incid�ncia de acidentes de trabalho.
   - **Dimensionamento de Gr�ficos:** A abordagem considerou o dimensionamento de gr�ficos conforme a quantidade de dados dispon�veis, com �nfase na diferencia��o entre dados categ�ricos nominais e ordinais.

* **Redistribui��o de Dados Desbalanceados:**
   - **Tratamento de Categorias "Others":** Desenvolvemos t�cnicas para redistribuir dados desbalanceados, substituindo a categoria "Others" na coluna "Critical Risk" por informa��es mais informativas, visando uma an�lise mais precisa.

* **An�lise Temporal de Acidentes:**
   - **Gr�ficos de Linhas Temporais:** Implementamos uma fun��o espec�fica para an�lise temporal, possibilitando a identifica��o de correla��es entre a �poca dos acidentes e suas frequ�ncias mensais, bimestrais, quadrimestrais, semestrais ou anuais.

* **An�lise Estat�stica:**
   - **Correla��es com Pearson e Chi-Squared:** Utilizamos a an�lise de Pearson para identificar correla��es fortes entre os dados, apresentando visualmente essas rela��es por meio de heatmaps. Al�m disso, empregamos o teste qui-quadrado (chi-squared) para avaliar as rela��es entre categorias.

* **Modelagem de Machine Learning:**
   - **Random Forest:** Criamos um modelo ensamblado de Random Forest, explorando diversas combina��es de �rvores de decis�o para aumentar a acur�cia do modelo.
   - **Valida��o Cruzada e Ajuste Fino:** Realizamos valida��o cruzada e ajuste fino de hiperpar�metros para melhorar a efic�cia do modelo, enfrentando desafios de desbalanceamento nas previs�es.
   - **Naive-Bayes:** Implementamos um modelo de Naive-Bayes, observando um significativo aumento na taxa de acerto. Este modelo foi escolhido para o deploy, utilizando o m�dulo nativo "pickle" para preservar a efic�cia durante a implementa��o pr�tica.

---

## 5. **Execu��o do C�digo**

Antes de executar o c�digo no Jupyter Notebook, � necess�rio garantir que as bibliotecas adequadas estejam instaladas. Utilize o seguinte comando para instalar as depend�ncias:

```bash
pip install pandas matplotlib numpy seaborn graphviz scipy scikit-learn imbalanced-learn
```

Certifique-se de ter o Python devidamente instalado em seu ambiente.

* **Bibliotecas Necess�rias**

O c�digo requer as seguintes bibliotecas Python:

```python
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import graphviz
from scipy.stats import norm, kendalltau, chi2_contingency, randint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.multiclass import unique_labels
from collections import Counter
import pickle

# Importa��es para uso do algoritmo de Random Forest
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV

# Importa��es para o uso do algoritmo de Naive-Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
```

* **Executando o C�digo**

1. Abra o Jupyter Notebook no terminal ou prompt de comando:

```bash
jupyter notebook
```

2. Navegue at� o diret�rio onde o c�digo est� localizado.

3. Abra o arquivo do notebook (`Industrial_Safety_Analysis.ipynb`) no Jupyter Notebook.

4. Execute as c�lulas de c�digo uma por uma utilizando `Shift + Enter`, ou ent�o no console do Jupyter Notebook clique em `Cell` e ent�o `Run All` para executar todas as c�lulas de uma �nica vez.

Certifique-se de que todas as depend�ncias foram instaladas corretamente e n�o h� erros durante a execu��o do c�digo.

Lembre-se de adaptar o caminho do arquivo caso esteja em um diret�rio diferente do Jupyter Notebook.
---