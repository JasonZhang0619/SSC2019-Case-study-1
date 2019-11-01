import tensorflow as tf
import datetime
import numpy as np
import os
os.chdir("C:\\Users\\jason\\Desktop\\SSC2019CaseStudy")
import utils
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

F = 'F1'
w = 'w1'
X, df = utils.read_imgset(csv_path='train_label.csv',train=True, F=F, w=w,
                          hist = False)
X, df = shuffle(X, df, random_state=0)
kf = KFold(n_splits=10)
train, test = list(kf.split(X))[0]
X_train, X_test =X[train,] / 255.0, X[test,] / 255.0
ytrain, ytest= df['count'][train], df['count'][test]

model = tf.keras.models.Sequential([
    tf.keras.layers.AvgPool2D(pool_size=[10, 10], strides=[5, 5], padding='valid', input_shape=(520, 696, 1), data_format='channels_last'),
    tf.keras.layers.Conv2D(filters=20,kernel_size=[5,5],strides=[3,3],padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=[6, 6], strides=[3, 3], padding='valid'),
    tf.keras.layers.Conv2D(filters=7,kernel_size=[5,5],strides=[3,3],padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
  ])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

log_dir="logs\\fit\\CNN2" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
%load_ext tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(x=X_train,
          y=np.array(ytrain),
          epochs=100,
          validation_data=(X_test, np.array(ytest)),
          callbacks=[tensorboard_callback])