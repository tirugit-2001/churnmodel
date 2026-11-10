# Salary Regression with ANN - Code

Source: `python-practice/annpractice1/salaryregression.ipynb`

Predicts `EstimatedSalary` from the churn dataset using an artificial neural network.

## 1. Imports

```python
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
```

## 2. Load the data

```python
data=pd.read_csv("Churn_Modelling.csv")
data.head()
```

## 3. Drop unneeded columns

```python
#preprocess the data 
data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)
data.head()
```

## 4. Label encode Gender

```python
#Encode categorical variables
label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
data.head()
```

## 5. One-hot encode Geography

```python
#One hot encode Geography
onehot_encoder_geo=OneHotEncoder(handle_unknown='ignore')
geo_encoded=onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df
```

## 6. Combine encoded columns with the data

```python
#Combine one-hot encoded columns with original data
data=pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
data.head()
```

## 7. Features and target

```python
X=data.drop('EstimatedSalary',axis=1)
y=data['EstimatedSalary']
```

## 8. Split and scale

```python
#split the data into training and testing sets 
X_train ,X_test ,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
```

## 9. Save the encoders and scaler

```python
# save the encoders and scaler using picke 
with open('label1_encoder_gender.pkl','wb') as f:
    pickle.dump(label_encoder_gender,f)
with open('onehot1_encoder_geo.pkl','wb') as f:
    pickle.dump(onehot_encoder_geo,f)
with open('scaler1.pkl','wb') as f:
    pickle.dump(scaler,f)
```

# ANN Regression Problem Statement

## 10. TensorFlow imports

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
```

## 11. Build and compile the model

```python
regression_model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
    Dense(32,activation='relu'),
    Dense(1)
])
regression_model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mae'])
regression_model.summary()
```

## 12. TensorBoard setup

```python
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
# Set up TensorBoard
log_dir = "regressionlogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
```

## 13. Early stopping

```python
#set up Early Stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
```

## 14. Train the model

```python
#Train the model 
history=regression_model.fit(
    X_train,y_train,
    validation_data=(X_test,y_test),
    epochs=100,
    callbacks=[early_stopping_callback,tensorboard_callback]
)
```
