# Titanic-Classification
A machine learning model on the `Titanic dataset` to predict whether a passenger on the Titanic would have been survived or not using the passenger data. If you want more details then click on **[link](https://www.kaggle.com/c/titanic)**.

So the data has information about passengers on the Titanic, such as `name` , `sex`, `age`, `survival`, `economic status(class)`, etc.

## Gathering Data
We start by importing Important libraries such as `Numpy`, `Pandas`, `matplotlib.pyplot`, `seaborn`  are data manipulation libraries. For machine learning we will use classification algorithm  *Logistic Regression* , *accuracy_score* and *train_test_split*  to split the data into train/ test to check and avoid overfitting. 

>Overfitting is when the model learns the training data so well that it fails to generalize the model for the test data or unseen data. Therefore, we have very good accuracy in train data but very poor accuracy in the test data.

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

```

| PassengerId  | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |                                                            
| :--- | :--- |:--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |:--- |
| 1  | 0 | 3 | Braund, Mr. Owen Harris	| male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S
| 2  | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 3  | 1 | 3 | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 | STON/023101282 | 7.9250 | NaN | S |
| 4  | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 | S |
| 5  | 0 | 3 | Allen, Mr. William Henry	 | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S |


## Data Visualisation
>Making the count plot for Survived

![Survived](https://user-images.githubusercontent.com/123626918/227458583-bafe1c4b-ea9e-4a9e-a804-f8ceaf7d0a1a.png)

![Sex](https://user-images.githubusercontent.com/123626918/227459561-95c84452-f183-4bfb-bc75-bec7f4673f42.png)

>Examining the survival statistics, a large majority of males did not survive the ship sinking. However, a majority of females did survive the ship sinking

![Sex-Survived](https://user-images.githubusercontent.com/123626918/227459824-2dea3adc-f7aa-42a8-8140-770d1b0266d4.png)
 

![Pclass](https://user-images.githubusercontent.com/123626918/227459977-b66fdbbc-7db1-41d6-a870-fa2954c1634b.png)

![Pclass-Survivors](https://user-images.githubusercontent.com/123626918/227460086-c4d10af7-9d28-45db-867e-b675259ea800.png)


## Model Training
```bash
model = LogisticRegression()
#training the Logistic Regression model with training data
model.fit(X_train, Y_train)
```
## Model Evaluation
>Accuracy Score of Training data
```bash
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
```
**Prediction have an accuracy for training data :** 80.75%  <!--~~(0.8075842696629213)~~-->




>Accuracy Score of Testing data
```bash
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
```
**Prediction have an accuracy :** 78.21 %   <!--(0.7821229050279329)-->

## License

[MIT](https://choosealicense.com/licenses/mit/)
