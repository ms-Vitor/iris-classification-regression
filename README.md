# 🌸 Projeto Iris: Análise Exploratória, Tratamento de Dados e Modelagem Preditiva

Este projeto foi desenvolvido com o objetivo de explorar o famoso dataset *Iris*, realizando desde a análise visual e tratamento estatístico dos dados até o treinamento e validação de dois modelos distintos de Machine Learning (Classificação e Regressão) integrados em uma arquitetura de código limpa e modular.

---

## 🛠️ Tecnologias e Ferramentas

O projeto foi inteiramente desenvolvido em **Python**, utilizando as principais bibliotecas do ecossistema de Ciência de Dados e Inteligência Artificial:

* **Pandas:** Utilizado para a carga do dataset, padronização de textos, agrupamento de dados (`groupby`) e preenchimento estatístico de valores nulos.
* **Matplotlib & Seaborn:** Responsáveis pela Análise Exploratória de Dados (EDA), plotagem de gráficos de dispersão (*scatter plots*) e geração da matriz de correlação em mapa de calor (*heatmap*).
* **Scikit-Learn (Sklearn):** Engine principal de Machine Learning, utilizada para:
  * Divisão da base em treino e teste (`train_test_split`).
  * Modelagem de Classificação (`SVC` - Support Vector Classification).
  * Modelagem de Regressão (`LinearRegression`).
  * Processo de Validação Cruzada (`cross_val_score`).
* **Git & GitHub:** Ferramentas essenciais para o controle de versão e hospedagem do portfólio.

---

## 📋 Fluxo do Projeto

### 1. Análise Exploratória de Dados (EDA)
A jornada começou com a compreensão das variáveis. Utilizando o `Seaborn` e o `Matplotlib`, plotei gráficos de dispersão e uma matriz de correlação. Essa etapa foi fundamental para identificar padrões visuais e o forte comportamento linear entre alguns atributos das flores.

### 2. Tratamento de Dados (Data Wrangling)
Para simular um cenário real e robusto de Engenharia de Dados, o dataset passou por uma rigorosa etapa de limpeza:
* **Padronização de Texto:** Transformação das strings da coluna `species` para letras minúsculas (`.str.lower()`), eliminando inconsistências de digitação.
* **Imputação de Dados Nulos:** Substituição de valores ausentes pela **mediana agrupada por espécie** (`groupby`), garantindo que as características específicas de cada categoria fossem preservadas.
* **Remoção de Outliers:** Aplicação do método estatístico **IQR (Intervalo Interquartil)** na coluna `petal_length`, calculando os limites inferior e superior para remover registros discrepantes que pudessem enviesar os algoritmos.

---

## 🤖 Modelagem e Treinamento

O projeto foi estruturado de forma versátil para abordar o problema sob duas perspectivas diferentes de Inteligência Artificial:

### 🎯 Classificação (SVC - Support Vector Classification)
* **Objetivo:** Identificar a espécie da flor (Setosa, Versicolor ou Virginica) com base nas medidas de largura e comprimento de suas pétalas e sépalas.
* **Métrica de Avaliação:** **Acurácia** (proporção de acertos sobre o total de tentativas).
* **Justificativa:** As espécies atuam como rótulos categóricos (classes), tornando o algoritmo SVM com kernel RBF ideal para mapear as fronteiras de decisão.

### 📐 Regressão Linear
* **Objetivo:** Prever a largura da pétala baseado exclusivamente no comprimento da pétala.
* **Métrica de Avaliação:** **Coeficiente de Determinação ($R^2$)** (mede o quão próxima a reta do modelo ficou dos resultados reais).
* **Justificativa:** A análise gráfica inicial revelou uma forte correlação linear (0.96) entre essas duas variáveis, justificando o uso da regressão.

---

## 🧪 Validação e Testes

Para ambos os modelos, a base de dados foi dividida utilizando a proporção de **80% para treinamento e 20% para testes** (`train_test_split`), com o estado de aleatoriedade fixado para garantir a reprodutibilidade. 

Após a avaliação dos scores individuais, foi aplicada a técnica de **Validação Cruzada** (`cross_val_score`) sobre a base de treino, calculando a média dos testes para validar o desempenho real dos modelos e mitigar riscos de *overfitting*.

---

## 🧠 Considerações Finais

Este projeto foi uma oportunidade incrível para vivenciar o pipeline completo de Ciência de Dados na prática. Consegui consolidar conhecimentos fundamentais como:
* Estruturação de código limpo e reutilizável baseado em Programação Orientada a Objetos (POO).
* Manipulação e transformação estatística de dados complexos com `Pandas`.
* Entendimento prático de como diferentes problemas (classificação vs. regressão) exigem métricas de avaliação totalmente distintas (Acurácia vs. $R^2$).

Consegui colocar em prática conceitos de engenharia de atributos, pipelines de validação e modelagem estatística, construindo uma base sólida para os meus próximos passos na área de Inteligência Artificial.

*Agradecimento especial ao canal **DATA ICMC** no YouTube pelos conteúdos e ensinamentos que serviram de base para este desenvolvimento.*
