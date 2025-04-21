# Predição de Turnover


Neste projeto foi abordado sobre um dos principais indicadores da Área de Recursos Humanos, Turnover.

O turnover é o número de funcionários desligados da empresa em determinado período comparado à quantidade de funcionários efetivos, o que gera o Índice de Rotatividade.

O turnover tem um grande impacto na produtividade e eficiência da organização quando ocorre de forma voluntária (quando o funcionário solicita desligamento), além de afetar o desempenho de outros funcionários.

O objetivo deste estudo é a utilização da técnica estatística de regressão logística binária, para prever quem pode solicitar desligamento. 



## Base de Dados

Para este trabalho foi utilizada uma base de dados fictícia contendo dados de funcionários fictícios. Os dados foram coletados do Kaggle, do dataset chamado “IBM HR Analytics Employee Attrition & Performance”. Este conjunto de dados oferece uma visão abrangente das informações relacionadas aos funcionários de uma organização fictícia criada por cientistas de dados da IBM.

O dataset contém 1.470 observações e 34 colunas, das quais 17 quantitativas e 17 colunas são qualitativas.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/entendimento_dados.png?raw=true">
</p>

Acesse o [Kaggle](https://www.kaggle.com/code/shwetapandey01/employee-attrition-eda-prediction-9-model) para fazer download do dataset e verificar mais informações.


## Análise e Preparação dos Dados

A coluna Attrition é a variável dependente do modelo e foi analisada para verificar o balanceamento dos dados. Para os valores de Attrition iguais a 1, ou seja, pessoas que solicitaram desligamento (que é o evento de interesse em estudo) está presente em apenas 16,12% das observações, enquanto o valor 0, ou pessoas que não solicitaram desligamento, está presente no restante das observações, 83,88%.

No gráfico abaixo é possível perceber o desbalanceamento entre os dados.

Para este estudo, não foi adotada nenhuma técnica de reamostragem para o dataset.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/Desbalanceamento_turnover.png?raw=true">
</p>


**Colunas numéricas**

Foi realizada uma análise descritiva das colunas numéricas.
A coluna EmployeeCount contém um único valor “1” para todas as observações na base de dados e é utilizada apenas para contabilizar as observações.
Analisando os valores média, mínimo e máximo da coluna StandardHours, nota-se que esta coluna possui um único valor (80) para todas as observações.

As colunas EmployeeCount e StandardHours, foram excluídas do conjunto de dados.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/colunas_numericas_turnover.png?raw=true">
</p>


**Colunas categóricas**

Para as colunas qualitativas, foi analisado o conteúdo de cada coluna no dataset.
Todas as colunas contém mais de uma categoria, com exceção da coluna Over18 que contém apenas o valor ”Y” presente em 100% das linhas do dataset, indicando que todas as observações são maiores que 18 anos.

Como o dataset já contém a coluna Age (idade dos empregados), a coluna Over18 também foi excluída do dataset, restando 30 colunas explicativas para análise do estudo.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/colunas_categoricas_turnover.png?raw=true">
</p>


**Preparação Final**

As variáveis qualitativas categóricas foram transformadas para variáveis dummy, aumentando a quantidade de colunas explicativas para aplicação da regressão logística para 61 variáveis, sendo uma variável dependente dicotômica do evento em estudo.


**Matriz de Correlação**

Foi gerada uma matriz de correlação das variáveis categóricas. Pode-se notar que existem variáveis com maiores correlações:
- MonthlyIncome and TotalWorkingYears
- YearsAtCompany and YearsInCurrentRole
- YearsAtCompany and YearsWithCurrManager

Essas variáveis foram mantidas no dataset, pois na construção do modelo foi executado um procedimento chamado Stepwise da biblioteca statsmodel, que será explicado melhor no próximo tópico.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/matriz_correlacao.png?raw=true">
</p>


## Construção do Modelo

Neste projeto foi trabalhado um modelo do tipo supervisionado que são modelos estatísticos usados quando queremos explicar ou prever dados, isso é feito com a ajuda de dados históricos que serão destinados ao treino do modelo e assim ele será capaz de prever dados de saída para novas entradas.

Para a construção dos modelo foi utilizado:

* **Logistic Regression:** A técnica de regressão logística binária é utilizada quando o fenômeno a ser estudado apresenta-se de forma qualitativa dicotômica (dois valores possíveis), representado por uma variável dummy, com o intuito de estimar a probabilidade de ocorrência do evento de interesse, considerando a chance [odds] de ocorrência do evento. A partir do logito define-se a expressão da probabilidade de ocorrência do evento em estudo, em função das variáveis explicativas. O logito é o logaritmo natural da chance de ocorrência de uma resposta do tipo “sim”.


**Logistic Regression**

Para aplicação da regressão logística, o dataset foi separado entre treino e teste, usando o percentual 80% e 20%, respectivamente.
- Treino:
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/proporcao_treino.png">
</p>

- Teste:
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/proporcao_teste.png">
</p>

Foi executado o modelo logito na base de treino, considerou-se o nível de significância de 5% e foi usada a função logit da biblioteca statsmodels.api.

Após executar o modelo, obteve-se o resultado do LLR p-value no valor de 1.207e-57 (em notação científica), ou seja, menor que 5% de nível de significância, que indicou que ao menos uma variável independente se mostrou estatisticamente relevante, ou beta [β] diferente de 0. Esse resultado mostrou que o modelo de regressão logística é adequado para seguir com a análise.

O p-value do teste Chi-Square [χ2] propicia uma verificação inicial da significância do modelo. O teste χ2 é calculado da seguinte forma: -2*(likelihood do modelo nulo – likelihood máximo). Com o resultado dessa função calcula-se o p-value do χ2, resultando no valor do log likelihood ratio test [LLR p-value] que verifica a adequação do ajuste do modelo completo em comparação com o ajuste do modelo final. Para que o teste de significância do modelo rejeite a hipótese nula (onde todos os betas são iguais a zero) ao nível de confiança de 95%, o LLR p-value deve ser menor que 5% de nível de significância para dar continuidade ao modelo de regressão logística.

```shell
          Logit Regression Results
Dep. Variable:	Attrition	  No. Observations:	1176
Model:	Logit	              Df Residuals:	1115
Method:	MLE	                Df Model:	60
Date:	Sun, 20 Apr 2025	    Pseudo R-squ.:	0.4149
Time:	14:30:55	            Log-Likelihood:	-304.29
converged:	False	          LL-Null:	-520.09
Covariance Type:	nonrobust	LLR p-value:	1.207e-57
```


**Feature Importances**

Para identificar quais variáveis foram de fato estatisticamente relevantes e que deveriam permanecer no modelo, foi executada a função stepwise da biblioteca statsmodels.process. Esta técnica mantém apenas os parâmetros estatisticamente significantes. Conforme a figura abaixo, pode-se observar que após o procedimento stepwise, 23 variáveis permaneceram no modelo, todas com o valor do p-value menor que o nível de significância de 5%.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/feature_importance.png">
</p>


## Interpretando indicadores de performance do modelo

A curva de sensibilidade é um gráfico que apresenta os valores da sensitividade e da especificidade em função dos diversos valores de cutoff. Por meio da curva de sensitividade e especificidade, é possível analisar e definir o melhor cutoff para o objetivo da análise.
O cutoff é um ponto de corte que o pesquisador escolhe o que melhor atende aos objetivos do modelo. Para definir o cutoff é necessária uma análise dos parâmetros para verificar o que melhor se encaixa para atender as necessidades do estudo. Não necessariamente, somente a acurácia deve ser levada em consideração, pois existem outros parâmetros como a sensitividade, que é a taxa de acerto do evento, e a especificidade, que é taxa de acerto do não evento. É possível escolher um cutoff que melhore mais a sensitividade, ou um cutoff que traga um melhor resultado para a especificidade.

**Curva de Sensibilidade**

Ao analisar a curva de sensitividade em relação a especificidade, é possível observar que a especificidade aumenta à medida que eleva o cutoff, ou seja, quanto maior o cutoff, maior é a taxa de acerto do não evento.  Este comportamento pode ter ocorrido devido ao desbalanceamento dos dados, levando o modelo a ter uma maior taxa de acerto para pessoas que não solicitaram desligamento, pois fazem parte da classe majoritária do dataset. Já um cutoff mais baixo, por exemplo 30%, pode-se observar que a curva de sensitividade mostra uma elevação, indicando que o modelo pode ter uma taxa melhor de acerto para os que são de fato o evento de interesse em estudo. Ao observar o ponto do cutoff de 20%, as curvas de sensitividade e especificidade se cruzam, indicando que ambas apresentam uma taxa de acerto bem próximas, entre 75% e 80%.

- Treino
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/curva_sensibilidade_treino.png" width="600" height="500">
</p>

- Teste
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/curva_sensibilidade_teste.png" width="600" height="500">
</p>


**Matriz de confusão**

A matriz de confusão é uma tabela com duas linhas e duas colunas que relata o número de falsos positivos, falsos negativos, verdadeiros positivos e verdadeiros negativos. Com base nesses resultados, calcula-se a sensitividade e a especificidade. Considerando um cutoff de 50% de acerto, a matriz de confusão do dataset de treino mostrou que o modelo atingiu 90% de acurácia, 97,26% de especificidade e 52,63% de sensitividade, enquanto a matriz de confusão após treinado o modelo no dataset de teste mostrou que o modelo atingiu 86,39% de acurácia, 95,55% de especificidade e 38,3% de sensitividade.

- Treino
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/matriz_confusao_treino_cutoff05.png">
</p>

- Teste
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/matriz_confusao_treino_cutoff03.png">
</p>

Ao considerar um cutoff de 30%, nota-se uma melhora da taxa de acerto dos que foram evento. A matriz de confusão mostrou que o modelo de treino atingiu 90,97% de especificidade e 68,42% de sensitividade, com uma acurácia de 87,33%. A matriz de confusão após treinado o modelo no dataset de teste mostrou que o modelo atingiu 84,69% de acurácia, 89,88% de especificidade e 57,45% de sensitividade

- Treino
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/matriz_confusao_teste_cutoff05.png">
</p>

- Teste
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/matriz_confusao_teste_cutoff03.png">
</p>


**Curva ROC**

A Curva ROC, ou Receiver Operating Characteristic, é uma representação gráfica que mostra o desempenho do modelo de regressão logística binária. Quanto maior a área abaixo da curva, significa que melhor é a eficiência de previsão do modelo. A curva ROC não depende do valor do cutoff, na verdade a área abaixo da curva aumenta de acordo com as variáveis preditoras estatisticamente relevantes que permanecem após o procedimento de stepwise.

Para o modelo de treino pode-se observar uma área abaixo da curva de 88,36%, enquanto o modelo de teste apresentou 81,17% de área abaixo da curva, mostrando que ambos os modelos tiveram uma ótima capacidade preditora, considerando as 18 variáveis que permaneceram após o procedimento stepwise.

- Treino
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/curva_roc_treino.png" width="600" height="500">
</p>

- Teste
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/teste/blob/main/Regress%C3%A3o%20Log%C3%ADstica/Imagens/curva_roc_teste.png" width="600" height="500">
</p>


## Considerações Finais

O modelo de regressão logística aplicado no estudo, se mostrou adequado para prever funcionários que podem solicitar desligamento, considerando dados históricos. Inicialmente o modelo foi gerado com 60 variáveis explicativas independentes. Foi realizada a análise do parâmetro LLR p-value que apresentou um valor abaixo de 0,05 de nível de significância, para análise de significância do modelo. Esse resultado mostrou que o modelo de regressão logística seria adequado para o estudo, sendo assim, foi dado continuidade com a execução do modelo realizando o procedimento stepwise. O modelo final permaneceu com 23 variáveis estatisticamente relevantes, com o p-value menor que 0,05 de nível de significância para explicar o evento de interesse.

Ao analisar o exponencial dos coeficientes das variáveis preditoras, foi possível observar algumas características que podem influenciar na decisão de funcionários solicitarem desligamento, como tais fatores: pessoas mais jovens, pessoas com salários menores, pessoas que moram mais distante do trabalho, pessoas que viajam frequentemente ou raramente a trabalho, pessoas insatisfeitas com o ambiente de trabalho. Também foi possível observar que pessoas que desempenham o cargo Sales Executive apresentaram mais chance de solicitar desligamento, indicando um possível problema no ambiente de trabalho, dentre outras análises que foi possível extrair do resultado final do modelo.
Mesmo com os dados desbalanceados e com uma classe majoritária no dataset sendo pessoas que não solicitaram desligamento, ao considerar um cutoff de 50% o modelo foi capaz de acertar os que foram evento, resultando em uma sensitividade de 52,26% e 38,3% no treino e teste do modelo, respectivamente. Ao considerar um cutoff que melhora a sensitividade, o modelo foi capaz de acertar 68,42%  e 57,45%, no treino e teste respectivamente.

A curva ROC apresentou uma excelente área abaixo da curva, resultando nos valores 88,36% e  81,17% no treino e teste respectivamente, mostrando que o modelo teve uma ótima capacidade preditora global.

Com estas análises, foi possível concluir que o uso da técnica estatística de regressão logística binária em um conjunto de dados históricos relevantes com uma ampla gama de variáveis explicativas, como características e informações contratuais dos funcionários, sua implementação se mostrou estatisticamente relevante para previsão do turnover. Sendo assim, a área de Recursos Humanos pode gerar ações para a retenção de talentos, usando análises baseadas em dados mensuráveis.



## Referências

Fávero, L. P.,  Belfiore, P. 2017. Manual de análise de dados: estatística e modelagem multivariada com Excel®, SPSS® e Stata®. Elsevier Brasil.

Fernandes, A. A. T., Filho, D. B. F., Rocha, E. C., Nascimento, W. S. 2020. Leia este artigo se você quiser aprender regressão logística. Programa de Pós-Graduação em Ciência Política, Universidade Federal de Pernambuco, Recife, PE, Brasil.
