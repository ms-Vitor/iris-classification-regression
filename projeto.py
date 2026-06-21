import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

class Modelo():
    def __init__(self, path):

        self.df = pd.read_csv(path)


    def plotar_graficos(self, df):

        #Calcula a correlação de cada coluna com as outras.(1 mt correlação e -1 o contrário).
        matriz_correlacao =df[["sepal_length" ,"sepal_width" ,"petal_length","petal_width"]].corr()
        sns.heatmap(matriz_correlacao,annot=True)#annot coloca o num dentro de cada quardado
        plt.show()
        
        # O 'hue' separa as cores automaticamente por espécie
        sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")#Correlação 0.96
        plt.show()

        sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")#Correlação -0.11
        plt.show()

        sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species")#Correlação 0.87
        plt.show()

    #Aqui só preparamos os dados, não tinha muito oq mexer(os dados já estavam bem estruturados)   
    def TratamentoDeDados(self):
                
        self.df["species"] = self.df["species"].str.lower()
        #Deixa todas str da coluna minusculas

        for coluna in ["sepal_length", "sepal_width", "petal_width"]:
            self.df[coluna] = self.df.groupby(["species"],group_keys=False)[coluna].apply(lambda x: x.fillna(x.median()))
            #Coloca a mediana nos nulos dessas colunas

        q1 = self.df["petal_length"].quantile(0.25)
        q3 = self.df["petal_length"].quantile(0.75)
        IQR = q3 - q1

        limite_superior = q3 + 1.5 * IQR

        self.df = self.df[self.df["petal_length"] < limite_superior] #Eliminando outliers

        self.df.info()

        self.plotar_graficos(self.df)

    def Treinamento_Regressao_linear(self):

        x = self.df[["petal_length"]] #Nosso recurso para essa analise
        y = self.df["petal_width"] #Nosso alvo(queremos adivinhar)

        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
        #20% fica pra teste e 80% restatnte para treinamento. Random só embaralha as linhas.

        regressao = linear_model.LinearRegression()
        regressao.fit(x_treino, y_treino)

        return regressao, x_treino, y_treino, x_teste, y_teste


    def Treinamento_Classificacao_linear(self):

        #Vamos separar nossos recursos e nosso alvos para o ML
        x = self.df[["sepal_length" ,"sepal_width" ,"petal_length","petal_width"]]#Recurso
        y = self.df["species"] #Alvo(queremos adinhar mais pra frente)

        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
        #20% fica pra teste e 80% restatnte para treinamento. Random só embaralha as linhas.
 
        classificador = SVC()#Kernel padrão, 'RBF'
        classificador.fit(x_treino, y_treino)

        return classificador, x_treino, y_treino, x_teste, y_teste
    

    def Teste(self):

        for modelo in ["Classificação", "Regressão"]:

            if(modelo == "Classificação"):
                modelo_treinado, x_treino, y_treino, x_teste, y_teste = self.Treinamento_Classificacao_linear()
                metrica = "Acurácia" #Proporção de acerto sobre o total de tentativas
            
            elif(modelo == "Regressão"):
                modelo_treinado, x_treino, y_treino, x_teste, y_teste = self.Treinamento_Regressao_linear()
                metrica = "Coeficiente de Determinação" #Mede o quão proximo chega dos resultados reais

            predicao_Y = modelo_treinado.predict(x_teste)

            score = modelo_treinado.score(x_teste, y_teste)
            validacao_cruzada = cross_val_score(modelo_treinado, x_treino, y_treino).mean()

            print(f"\n\nMODELO: {modelo}\n\nPredições do nosso modelo:\n{predicao_Y}\nDados reais para comparar:\n{y_teste}\n\n{metrica} do nosso modelo: {score*100:.1f}%\nValidação Cruzada: {validacao_cruzada*100:.1f}%")

    def Train(self):

        self.TratamentoDeDados()
        self.Teste()  # Executa o treinamento do modelo


modelo = Modelo("Data/iris.csv")

modelo.Train()