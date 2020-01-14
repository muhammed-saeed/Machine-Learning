

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


#pima-indians-diabetes is a dataset for the indians people whom are divided into two classes part with diabetes and other part without diabetes
#we build LinearClassifier model to predict whether the person has diabetes or not
#we evaluate our linear model and then
#then we build simple neural network to predict whether person has diabetes or not

diabetes = pd.read_csv('pima-indians-diabetes.csv')
diabetes.head()
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']
#the Age going to be treated as categorical data
#the class is the ouptut
#the Group are characheters

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#normallize the continous numeric values!!
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')
#create the feature columns for the numerical continous values


assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
#create the  feature columns for categorical data


age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]
#those are the inputs to the  model will be used later with tensorflow estimators API

x_data = diabetes.drop('Class',axis=1)
labels = diabetes['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
results = model.evaluate(eval_input_func)
print(results)

pred_input_function = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size= 10, num_epochs = 1, shuffle = False )
predictions = model.predict(pred_input_function)
my_pred = list(predictions)


#note our above model hits only about 74% accuracy so we seek the power of Neural nets

embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
#pass to the embedding columns the categorical features
#the above line of code is to train the nn-model

feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,embedded_group_col, age_buckets]
#change the feature columns into embedding columns

input_func =tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size = 10, num_epochs = 1000, shuffle = True)

dnn_model = tf.estimator.DNNClassifier(hidden_units = [10,10,10], feature_columns= feat_cols, n_classes = 2)

dnn_model.train(input_fn = input_func, steps = 1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size = 10, num_epochs =1, shuffle=False )
dnn_model.evaluate(eval_input_func)
