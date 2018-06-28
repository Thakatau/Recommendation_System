"""

A model that will determine whether a student will complete their degree in minimum time given their first year marks

"""


"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Import MergedDatabase .CSV:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import pandas

mergedFinal = pandas.read_excel('/Users/popinjay/Desktop/MergedFinal.xlsx')


""" 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Takes dataframe and returns relevant columns as dataframe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

def drop_columns(df): 
    
    d1= df.loc[(df['Year Of Study'] == 'YOS 1') & (df['Program Code'] == 'EF003')]
    
    c1 = [2,4,7,16,20,21,22,26,d1.columns.get_loc("CHEM1033"),d1.columns.get_loc("ELEN1000"),d1.columns.get_loc("MATH1014"),d1.columns.get_loc("PHYS1014"),d1.columns.get_loc("PHYS1015")]
    
    d2 = d1[d1.columns[c1]]
    
    del d2['Student Number']
   
    return d2

# 

"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dropping columns which are not necessary for model:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
 
"""
gradFinal = drop_columns(mergedFinal)

print(gradFinal.describe())



# Selecting target and the dataset:

y1 = gradFinal['Completed in Minimum Time']

y2 = gradFinal['Total Years To Grad']

x_  = gradFinal.drop(['Completed in Minimum Time', 'Total Years To Grad'], axis = 1)


"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Splitting Data:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_, y1, test_size = 0.3, random_state = 42)


"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data Preprocessing Stage:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

# Determining how many nun values are in each column:

print(gradFinal.isnull().sum())


"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using Imputer:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from sklearn.preprocessing import Imputer


"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using imputer replacing nun values with the median for each column of grades
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

imputer = Imputer(strategy="median") # Imputing the mean to missing values as we don't want to lose 20 % of data:

imputer.fit(x_train)
imputer.fit(x_test)

x_train.median().values              # Ensuring Imputer is applied to all numerical values
x_test.median().values


x_trans = imputer.transform(x_train) # Implementing Imputer on the training data
x_test_tr = imputer.transform(x_test)




# Converting the numpy array back into a dataframe: This can actually be done in place.

import pandas as pd

x_transform = pd.DataFrame(x_trans, columns=x_train.columns)

x_test = pd.DataFrame(x_test_tr, columns = x_test.columns)

"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implementing Standard Scaler: not necessary, no outliers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from sklearn.preprocessing import StandardScaler

scaler =  StandardScaler()

# Applying the transformation in place:

x_transform[['Average Marks','CHEM1033','ELEN1000','MATH1014','PHYS1014','PHYS1015']] = scaler.fit_transform(x_transform[['Average Marks','CHEM1033','ELEN1000','MATH1014','PHYS1014','PHYS1015']])


"""
Training the model using a deep neural network:
"""
# Needed to define placeholders:


import tensorflow as tf

tf.reset_default_graph() 

import math 

n_inputs  = 10 # equal to the number of features
n_outputs = 4  # three outputs of 7,5,4,0
n_hidden1 = math.ceil((n_inputs+n_outputs)/2) # the number of neurons on the hidden layer is the mean of the input + output


X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int64, shape = (None), name = "y")



from tensorflow.contrib.layers import fully_connected

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope = "hidden1")
    logits  = fully_connected(hidden1, n_outputs, scope = "outputs", activation_fn = None)
    

"""
cost function: cross entropy 

"""
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name = "loss")
     
     
"""

Defining Gradient Descent that will tweak model parameters to minimize cost function:
    
"""

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
# USING ACCURACY AS A PERFORMANCE MEASURE OF MODEL:

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
# NOW FOR THE TENSORFLOW SEESSION:
init = tf.global_variables_initializer()
saver = tf.train.Saver()


"""
EXECUTION PHASE:
"""

n_epochs = 100
batch_size = 20


import numpy as np

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        epoch_loss = 0
        total_batch = int(len(x_train)/batch_size)
        
        x_batches = np.array_split(x_test_tr, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            sess.run(training_op, feed_dict = {X: batch_x, y: batch_y})
            
        
                                   
      
"""

Using the high level API:
    
"""  
import math
n_inputs  = 6 # equal to the number of features
n_outputs = 4  # three outputs of 7,5,4,0
n_hidden1 = math.ceil((n_inputs+n_outputs)/2) # the number of neurons on the hidden layer is the mean of the input + output



import tensorflow as tf

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_transform)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[n_hidden1], n_classes = 2, feature_columns = feature_columns)
dnn_clf.fit(x = x_train, y = y_train, batch_size = 20, steps = 100)


from sklearn.metrics import accuracy_score
y_pred = list(dnn_clf.predict(x_test))
print(accuracy_score(y_test, y_pred))





