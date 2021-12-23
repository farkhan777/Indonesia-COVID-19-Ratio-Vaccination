# Indonesia-COVID-19-Ratio-Vaccination 🚧

## Project Overview

The dataset is acquired from kaggle which entitled [COVID vaccination vs. mortality](https://www.kaggle.com/sinakaraji/covid-vaccination-vs-death).
The CSV file contains vaccination rate data and its datetime for each respective countries. In addition to that, the columns provided in this CSV are as follows:

* country 
  * provided country name
* iso_code 
  * iso_code for each country
* date
  * date that this data belong    
* total_vaccinations
  * number of all doses of COVID vaccine usage in that country
* people_vaccinated
  * number of people who got at least one shot of COVID vaccine
* peoplefullyvaccinated
  * number of people who got full vaccine shots
* New_deaths
  * number of daily new deaths
* population
  * 2021 country population  
* ratio
  * % of vaccinations in that country at that date = people_vaccinated/population * 100.

The main concern when creating our Machine Learning model are **country** and **ratio** column.

Based on current dataset, the vaccination rate from `2021-01-28` to `2021-11-20` in Indonesia is illustrated as follows:

![Vaccination Ratio](https://github.com/farkhan777/Indonesia-COVID-19-Ratio-Vaccination/blob/ilham_deploy_ML/documentation/persebaran-rasio-vaksinasi.jpg?raw=true)

## Tech Stack
<br />

![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

<br />

## Modelling

There are various options for time-series model creation. Here, we tried to implement LSTM.

LSTM is the new fix from traditional RNN whereas it allows the model to keep the important values and remove those which aren't; in long period of time by default.

As the model's complexity grows, the possibility overfitting occurence getting higher as well. Even though the architecture is suited specifically for the dataset, using correct loss function, and suitable metrics, there is a chance of data overfitting. Therefore, in addition to LSTM we also use Dropout Layer to prevent overfitting while data is being trained. Simply, dropout layer will acts as hidden layer which will be enabled-disabled while training the data. 

In this project, the dropout value is 0.5, the analogy is illustrated as follows:

![Model LSTM](https://camo.githubusercontent.com/79f4e3545a9f6b9e0987df5ed060c4f1320f3e4325bda659231ef6240e346aea/68747470733a2f2f64313769767139623772707062332e636c6f756466726f6e742e6e65742f6f726967696e616c2f61636164656d792f323032303038303331323532303262303737613132353361373764656639623965346165366235353362633163632e676966)

For thorough model creation, see more [here](https://github.com/farkhan777/Indonesia-COVID-19-Ratio-Vaccination/blob/main/Indonesia_COVID_19_Ratio_Vaccination.ipynb)

Created model will be saved into h5 format, and it will be processed further in `predict_vaccination.py`

Overall, the architecture of this model is LSTM layer as the input layer, then it goes to dropout layer with 0.5 value to increase output variance. Then Dense layer with 1 perceptron will be the its output layer.

## Model Evaluation 🚧

The prediction of vaccination rate in Indonesia

![Prediction result](https://github.com/farkhan777/Indonesia-COVID-19-Ratio-Vaccination/blob/main/documentation/prediction-result.png?raw=true)

The training and validation MAE is illustrated as follows:

![Training MAE](https://github.com/farkhan777/Indonesia-COVID-19-Ratio-Vaccination/blob/ilham_deploy_ML/documentation/train-and-validation-mae.jpg?raw=true)

## Evaluation

![MAE Formula](https://github.com/ilhamadhim/TLKM-Stock-Analysis/raw/master/assets/MAE_Formula.png?raw=true)

This metric is used to define model faults or variance of bias of trained model towards data that will be tested

## Project Parts
See other repos :
* 〽 [Frontend Implementation](https://github.com/ilhamAdhim/covid-vaccination-rate)

## Deployed by
![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)
