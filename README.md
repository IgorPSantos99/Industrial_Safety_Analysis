# Modelo de Previsão de Acidentes em Indústrias

## 1. **Introdução**

Os acidentes de trabalho são ocorrências indesejadas que podem causar lesões corporais ou doenças aos trabalhadores durante o exercício de suas atividades. Esses incidentes podem ocorrer devido a várias razões, como condições inseguras no local de trabalho, falta de treinamento, equipamentos inadequados e outras negligências.
A importância de prever e prevenir esses acidentes é multifacetada e impacta diversos aspectos:

* **Saúde e Segurança dos Trabalhadores**: Os acidentes em indústria colocam em risco a saúde e a segurança dos trabalhadores. Quedas de altura, choques elétricos, quedas de objetos, cortes, contusões e amputações são alguns dos acidentes mais comuns. Essas situações não apenas afetam a integridade física dos trabalhadores, mas também podem resultar em doenças crônicas ou até mesmo mortes.

* **Impacto Econômico**: Os custos associados aos acidentes de trabalho são significativos. Eles geram prejuízos para os trabalhadores, as empresas, a previdência social e a economia como um todo. Vamos explorar esses custos em detalhes:

    - **Custos Diretos**:
        - Despesas médicas, odontológicas e hospitalares para atender o acidentado.
        - Cirurgias reparadoras e reabilitação médica e ocupacional.
        - Transporte do acidentado durante o tratamento.
        - Seguro de acidente.
    
    - **Custos Indiretos**:
        - Salários pagos durante o tempo perdido por outros trabalhadores que não o acidentado.
        - Atrasos na produção devido ao acidente.
        - Salários adicionais por horas extras para compensar a produção perdida.
        - Custos com treinamento do substituto do acidentado.
        - Danos a máquinas ou equipamentos envolvidos no acidente.
        - Despesas médicas não cobertas pela seguradora.
        - Diminuição da eficiência do acidentado ao retornar ao trabalho.

* **Legalidade e Reputação**: Empresas que negligenciam a segurança no trabalho enfrentam custos legais, multas e processos trabalhistas. Além disso, a reputação da empresa pode ser afetada negativamente, afastando clientes e investidores.

* **Produtividade e Lucro**: Investir em segurança do trabalho reduz acidentes, diminui custos e aumenta a produtividade. Funcionários saudáveis e seguros são mais produtivos, e uma cultura de segurança melhora a reputação da empresa.

Em nosso estudo, abordaremos tópicos como a análise dos setores mais propensos a acidentes, a distribuição temporal desses eventos, a influência de categorias específicas e a construção de um modelo de previsão de acidentes futuros. Acreditamos que essa pesquisa contribuirá para a promoção de ambientes de trabalho mais seguros e saudáveis.

---

## 2. **Índice**

1. [Introdução]()
2. [Índice]()
3. [Fonte]()
4. [Habilidades Adquiridas]()
5. [Execução do Código]()


---

## 3. **Fonte**

Para este estudo, foi utilizado o dataset disponibilizado no Kaggle por IHM Stefanini em 
https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database


## 4. **Habilidades Adquiridas**

Durante a execução do projeto, foram adquiridas diversas competências essenciais para a manipulação, análise e modelagem de dados relacionados a acidentes de trabalho. Abaixo estão destacadas as principais habilidades adquiridas e as aprendizagens resultantes de cada uma:

* **Manipulação de Dados com Pandas:**
   - **Exportação e Modificação de Dados:** Foi explorada a funcionalidade do Pandas para exportar dados e efetuar modificações, como a alteração de títulos de colunas e ajuste de tipos de dados.
   - **Limpeza de Dados:** Desenvolvemos habilidades na identificação e tratamento de valores NaN, aprimorando a qualidade dos conjuntos de dados.

* **Análise Exploratória de Dados:**
   - **Contagens e Gráficos com Seaborn:** Utilizamos a biblioteca Seaborn para explorar a contagem de ocorrências e criar gráficos que forneceram insights sobre as categorias com maior incidência de acidentes de trabalho.
   - **Dimensionamento de Gráficos:** A abordagem considerou o dimensionamento de gráficos conforme a quantidade de dados disponíveis, com ênfase na diferenciação entre dados categóricos nominais e ordinais.

* **Redistribuição de Dados Desbalanceados:**
   - **Tratamento de Categorias "Others":** Desenvolvemos técnicas para redistribuir dados desbalanceados, substituindo a categoria "Others" na coluna "Critical Risk" por informações mais informativas, visando uma análise mais precisa.

* **Análise Temporal de Acidentes:**
   - **Gráficos de Linhas Temporais:** Implementamos uma função específica para análise temporal, possibilitando a identificação de correlações entre a época dos acidentes e suas frequências mensais, bimestrais, quadrimestrais, semestrais ou anuais.

* **Análise Estatística:**
   - **Correlações com Pearson e Chi-Squared:** Utilizamos a análise de Pearson para identificar correlações fortes entre os dados, apresentando visualmente essas relações por meio de heatmaps. Além disso, empregamos o teste qui-quadrado (chi-squared) para avaliar as relações entre categorias.

* **Modelagem de Machine Learning:**
   - **Random Forest:** Criamos um modelo ensamblado de Random Forest, explorando diversas combinações de árvores de decisão para aumentar a acurácia do modelo.
   - **Validação Cruzada e Ajuste Fino:** Realizamos validação cruzada e ajuste fino de hiperparâmetros para melhorar a eficácia do modelo, enfrentando desafios de desbalanceamento nas previsões.
   - **Naive-Bayes:** Implementamos um modelo de Naive-Bayes, observando um significativo aumento na taxa de acerto. Este modelo foi escolhido para o deploy, utilizando o módulo nativo "pickle" para preservar a eficácia durante a implementação prática.

---

## 5. **Execução do Código**

Antes de executar o código no Jupyter Notebook, é necessário garantir que as bibliotecas adequadas estejam instaladas. Utilize o seguinte comando para instalar as dependências:

```bash
pip install pandas matplotlib numpy seaborn graphviz scipy scikit-learn imbalanced-learn
```

Certifique-se de ter o Python devidamente instalado em seu ambiente.

* **Bibliotecas Necessárias**

O código requer as seguintes bibliotecas Python:

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

# Importações para uso do algoritmo de Random Forest
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV

# Importações para o uso do algoritmo de Naive-Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
```

* **Executando o Código**

1. Abra o Jupyter Notebook no terminal ou prompt de comando:

```bash
jupyter notebook
```

2. Navegue até o diretório onde o código está localizado.

3. Abra o arquivo do notebook (`Industrial_Safety_Analysis.ipynb`) no Jupyter Notebook.

4. Execute as células de código uma por uma utilizando `Shift + Enter`, ou então no console do Jupyter Notebook clique em `Cell` e então `Run All` para executar todas as células de uma única vez.

Certifique-se de que todas as dependências foram instaladas corretamente e não há erros durante a execução do código.

Lembre-se de adaptar o caminho do arquivo caso esteja em um diretório diferente do Jupyter Notebook.
---