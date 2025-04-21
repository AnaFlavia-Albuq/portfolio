# Turnover Prediction


This project addressed one of the main indicators in the Human Resources Area: Turnover.

Turnover is the number of employees who leave the company in a given period compared to the number of permanent employees, which generates the Turnover Rate.

Turnover has a major impact on the productivity and efficiency of the organization when it occurs voluntarily (when the employee requests termination), in addition to affecting the performance of other employees.

The objective of this study is to use the statistical technique of binary logistic regression to predict who may request termination.



## Database

For this work, we used a fictitious database containing data from fictitious employees. The data was collected from Kaggle, from the dataset called “IBM HR Analytics Employee Attrition & Performance”. This dataset provides a comprehensive view of information related to employees of a fictitious organization created by IBM data scientists.

The dataset contains 1,470 observations and 34 columns, of which 17 are quantitative and 17 are qualitative.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/entendimento_dados.png">
</p>

Visit [Kaggle](https://www.kaggle.com/code/shwetapandey01/employee-attrition-eda-prediction-9-model) to download the dataset and check more information.


## Data Analysis and Preparation

The Attrition column is the dependent variable of the model and was analyzed to verify the balance of the data. For Attrition values ​​equal to 1, that is, people who requested termination (which is the event of interest in the study) it is present in only 16.12% of the observations, while the value 0, or people who did not request termination, is present in the remaining observations, 83.88%.

The imbalance between the data can be seen in the graph below.

For this study, no resampling technique was adopted for the dataset.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/Desbalanceamento_turnover.png">
</p>


**Numeric columns**

A descriptive analysis of the numerical columns was performed.
The EmployeeCount column contains a single value “1” for all observations in the database and is used only to count the observations.
Analyzing the average, minimum and maximum values ​​of the StandardHours column, it is noted that this column has a single value (80) for all observations.

The EmployeeCount and StandardHours columns were excluded from the dataset.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/colunas_numericas_turnover.png">
</p>


**Categorical columns**

For the qualitative columns, the content of each column in the dataset was analyzed.
All columns contain more than one category, with the exception of the Over18 column, which contains only the value “Y”, present in 100% of the dataset rows, indicating that all observations are over 18 years old.

Since the dataset already contains the Age column (age of employees), the Over18 column was also excluded from the dataset, leaving 30 explanatory columns for analysis in the study.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/colunas_categoricas_turnover.png">
</p>


**Final Preparation**

The categorical qualitative variables were transformed into dummy variables, increasing the number of explanatory columns for applying logistic regression to 61 variables, with a dichotomous dependent variable for the event under study.

**Correlation Matrix**

A correlation matrix of the categorical variables was generated. It can be seen that there are variables with higher correlations:
- MonthlyIncome and TotalWorkingYears
- YearsAtCompany and YearsInCurrentRole
- YearsAtCompany and YearsWithCurrManager

These variables were kept in the dataset, since a procedure called Stepwise from the statsmodel library was executed in the construction of the model, which will be explained in more detail in the next topic.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/matriz_correlacao.png">
</p>


## Model Construction

This project involved working on a supervised model, which are statistical models used when we want to explain or predict data. This is done with the help of historical data that will be used to train the model, so that it will be able to predict output data for new inputs.

The following were used to build the model:

* **Logistic Regression:** The binary logistic regression technique is used when the phenomenon to be studied presents itself in a qualitative dichotomous form (two possible values), represented by a dummy variable, with the aim of estimating the probability of the event of interest occurring, considering the chance [odds] of the event occurring. The logit is used to define the expression of the probability of the event under study occurring, based on the explanatory variables. The logit is the natural logarithm of the chance of a “yes” response occurring.

**Logistic Regression**

To apply logistic regression, the dataset was separated into training and testing, using the percentages 80% and 20%, respectively.

- Train:
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/proporcao_treino.png">
</p>

- Test:
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/proporcao_teste.png">
</p>

The logit model was run on the training base, considering a significance level of 5% and using the logit function from the statsmodels.api library.

After running the model, the LLR p-value result was 1.207e-57 (in scientific notation), i.e., less than a 5% significance level, which indicated that at least one independent variable was statistically relevant, or beta [β] different from 0. This result showed that the logistic regression model is suitable for continuing with the analysis.

The p-value of the Chi-Square [χ2] test provides an initial verification of the significance of the model. The χ2 test is calculated as follows: -2*(likelihood of the null model – maximum likelihood). The result of this function is used to calculate the p-value of χ2, resulting in the value of the log likelihood ratio test [LLR p-value] that verifies the adequacy of the adjustment of the complete model in comparison with the adjustment of the final model. In order for the significance test of the model to reject the null hypothesis (where all betas are equal to zero) at the 95% confidence level, the LLR p-value must be less than the 5% significance level to continue with the logistic regression model.

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

To identify which variables were in fact statistically relevant and should remain in the model, the stepwise function from the statsmodels.process library was executed. This technique keeps only the statistically significant parameters. As shown in the figure below, it can be seen that after the stepwise procedure, 23 variables remained in the model, all with a p-value lower than the 5% significance level.

<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/feature_importance.png">
</p>


## Interpreting model performance indicators

The sensitivity curve is a graph that shows the sensitivity and specificity values ​​as a function of the various cutoff values. Using the sensitivity and specificity curve, it is possible to analyze and define the best cutoff for the purpose of the analysis.
The cutoff is a cutoff point that the researcher chooses that best meets the objectives of the model. To define the cutoff, an analysis of the parameters is necessary to verify what best fits the needs of the study. Not necessarily, only accuracy should be taken into consideration, as there are other parameters such as sensitivity, which is the event hit rate, and specificity, which is the non-event hit rate. It is possible to choose a cutoff that improves sensitivity the most, or a cutoff that yields a better result for specificity.

**Sensitivity Curve**

When analyzing the sensitivity curve in relation to specificity, it is possible to observe that specificity increases as the cutoff increases, that is, the higher the cutoff, the higher the accuracy rate for the non-event. This behavior may have occurred due to data imbalance, leading the model to have a higher accuracy rate for people who did not request termination, as they are part of the majority class in the dataset. However, with a lower cutoff, for example 30%, it can be observed that the sensitivity curve shows an increase, indicating that the model may have a better accuracy rate for those who are in fact the event of interest under study. When observing the 20% cutoff point, the sensitivity and specificity curves intersect, indicating that both have a very close accuracy rate, between 75% and 80%.

- Treino
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/curva_sensibilidade_treino.png" width="600" height="500">
</p>

- Teste
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/curva_sensibilidade_teste.png" width="600" height="500">
</p>


**Confusion Matrix**

The confusion matrix is ​​a table with two rows and two columns that reports the number of false positives, false negatives, true positives, and true negatives. Based on these results, sensitivity and specificity are calculated. Considering a cutoff of 50% accuracy, the confusion matrix of the training dataset showed that the model achieved 90% accuracy, 97.26% specificity, and 52.63% sensitivity, while the confusion matrix after training the model on the test dataset showed that the model achieved 86.39% accuracy, 95.55% specificity, and 38.3% sensitivity.

- Train
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/matriz_confusao_treino_cutoff05.png">
</p>

- Test
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/matriz_confusao_treino_cutoff03.png">
</p>

When considering a cutoff of 30%, an improvement in the accuracy rate of those that were events is noted. The confusion matrix showed that the training model achieved 90.97% specificity and 68.42% sensitivity, with an accuracy of 87.33%. The confusion matrix after training the model on the test dataset showed that the model achieved 84.69% accuracy, 89.88% specificity and 57.45% sensitivity.

- Train
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/matriz_confusao_teste_cutoff05.png">
</p>

- Test
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/matriz_confusao_teste_cutoff03.png">
</p>


**ROC Curve**

The ROC Curve, or Receiver Operating Characteristic, is a graphical representation that shows the performance of the binary logistic regression model. The larger the area under the curve, the better the model's predictive efficiency. The ROC curve does not depend on the cutoff value; in fact, the area under the curve increases according to the statistically relevant predictor variables that remain after the stepwise procedure.

For the training model, an area under the curve of 88.36% can be observed, while the test model presented an area under the curve of 81.17%, showing that both models had excellent predictive capacity, considering the 18 variables that remained after the stepwise procedure.

- Train
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/curva_roc_treino.png" width="600" height="500">
</p>

- Test
<p align="center">
  <img src="https://github.com/AnaFlavia-Albuq/portfolio/blob/main/Logistic%20Regression/Imagens/curva_roc_teste.png" width="600" height="500">
</p>


## Considerations

The logistic regression model applied in the study proved to be adequate for predicting employees who may request dismissal, considering historical data. Initially, the model was generated with 60 independent explanatory variables. The LLR p-value parameter was analyzed, which presented a value below 0.05 significance level, for analysis of the significance of the model. This result showed that the logistic regression model would be adequate for the study, therefore, the model was continued with the stepwise procedure. The final model remained with 23 statistically relevant variables, with a p-value below 0.05 significance level to explain the event of interest.

By analyzing the exponential of the coefficients of the predictor variables, it was possible to observe some characteristics that may influence the decision of employees to request termination, such as: younger people, people with lower salaries, people who live further away from work, people who travel frequently or rarely for work, people dissatisfied with the work environment. It was also possible to observe that people who hold the Sales Executive position were more likely to request termination, indicating a possible problem in the work environment, among other analyses that were possible to extract from the final result of the model.

Even with the unbalanced data and with a majority class in the dataset being people who did not request termination, when considering a cutoff of 50% the model was able to correctly identify those that were events, resulting in a sensitivity of 52.26% and 38.3% in the training and testing of the model, respectively. When considering a cutoff that improves sensitivity, the model was able to correctly identify 68.42% and 57.45%, in the training and testing respectively.

The ROC curve showed an excellent area under the curve, resulting in values ​​of 88.36% and 81.17% in training and testing, respectively, showing that the model had excellent overall predictive capacity.

With these analyses, it was possible to conclude that the use of the binary logistic regression statistical technique in a set of relevant historical data with a wide range of explanatory variables, such as employee characteristics and contractual information, proved to be statistically relevant for predicting turnover. Therefore, the Human Resources area can generate actions for talent retention, using analyses based on measurable data.



## Referências

Fávero, L. P.,  Belfiore, P. 2017. Manual de análise de dados: estatística e modelagem multivariada com Excel®, SPSS® e Stata®. Elsevier Brasil.

Fernandes, A. A. T., Filho, D. B. F., Rocha, E. C., Nascimento, W. S. 2020. Leia este artigo se você quiser aprender regressão logística. Programa de Pós-Graduação em Ciência Política, Universidade Federal de Pernambuco, Recife, PE, Brasil.
