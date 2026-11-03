import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler ,LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv('Churn_Modelling.csv')
print(data.head())
#remove some not important columns
data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)
## encode categorical variables
label_encoder=LabelEncoder()
data['Gender']=LabelEncoder().fit_transform(data['Gender'])
#one hot encoding for Geography
one_hot_encoder=OneHotEncoder()
geo_encoder=one_hot_encoder.fit_transform(data[['Geography']])


geo_encoded=pd.DataFrame(geo_encoder.toarray(),columns=one_hot_encoder.get_feature_names_out(['Geography']))
