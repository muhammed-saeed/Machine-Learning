import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


housing = pd.read_csv('cal_housing_clean.csv')

x_data = housing.drop(['medianHouseValue'],axis=1)
#all the input_features
y_val = housing['medianHouseValue']
#the taregt
X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.3,random_state=101)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)
#scale the feature data
X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)
#scale the feature data
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')
#create features columns for the numerical features

feat_cols = [ age,rooms,bedrooms,pop,households,income]
#create list of all the features columns

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=1000,
                                            shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)

model.train(input_fn=input_func,steps=25000)

predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size= 10,num_epochs=1,shuffle=False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

print(mean_squared_error(y_test,final_preds)**0.5)