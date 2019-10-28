import tensorflow as tf
import numpy as np
import pandas as pd
import utils
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_all=utils.readtrainingset()

def pooling(hist, window, step):
    with tf.name_scope('pooling'):
        hist_pool = tf.nn.avg_pool(hist, ksize=[1, window, 1, 1], strides=[1, step, 1, 1], padding='VALID')
        hist_pool = tf.keras.layers.Flatten()(hist_pool)
    return hist_pool

for F in ['F1','F23','F48']:
    for w in ['w1','w2']:
        print('For img at level', F, w)
        trainsetF1w2 = utils.get_data_level(train_all, F, w)
        os.chdir('C:/Users/jason/Dropbox/Data Files_Question1_SSC2019CaseStudy')
        size_histF1w2=utils.get_cellsize_hist()
        datasetF1w2= pd.concat([trainsetF1w2,size_histF1w2], axis=1, join='inner')


        trainset = datasetF1w2[:300]
        testset = datasetF1w2[300:]

        train_hist = np.array(trainset.iloc[:, :255])
        train_cell_hist = np.array(trainset.iloc[:, -249:])
        train_counts = np.array(trainset['count'])

        test_hist = np.array(testset.iloc[:, :255])
        test_cell_hist = np.array(testset.iloc[:, -249:])
        test_counts = np.array(testset['count'])

        # regression with pooling
        # using tf ave pool so some redundant codes for tf structure
        train_hist = np.array(trainset.iloc[:, :255]).reshape([-1, 255, 1, 1])
        test_hist = np.array(testset.iloc[:, :255]).reshape([-1, 255, 1, 1])
        with tf.name_scope('inputs'):
            hist = tf.placeholder(dtype=tf.float32, shape=[None, 255, 1, 1], name='image_hist')
            features = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='features')
            count = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='counts')

        n=14
        hist_pool = pooling(hist, n, n)
        sess = tf.Session()
        #without topology
        train_hist_pool = sess.run(hist_pool, feed_dict={hist: train_hist})
        print(n)
        reg = LinearRegression().fit(train_hist_pool, train_counts)
        pred_train = reg.predict(train_hist_pool)
        train_MSE_pool = mean_squared_error(train_counts, pred_train)

        test_hist_pool = sess.run(hist_pool, feed_dict={hist: test_hist})
        pred_test = reg.predict(test_hist_pool)
        test_MSE_pool = mean_squared_error(test_counts, pred_test)

        print('MSE for train set after pooling with', n, ' :', train_MSE_pool)
        print('MSE for test set after pooling with', n, ' :', test_MSE_pool, '\n')


#with topology

train_feats=np.append(train_hist_pool,train_cell_hist,axis=1)
reg2 = LinearRegression().fit(train_feats, train_counts)
pred_train2 = reg2.predict(train_feats)

test_feats=np.append(test_hist_pool,test_cell_hist,axis=1)
pred_test2 = reg2.predict(test_feats)
test_MSE_pool = mean_squared_error(test_counts, pred_test2)
train_MSE_pool = mean_squared_error(train_counts, pred_train2)
print('MSE for train set using cell sizes with', n, ' :', train_MSE_pool)
print('MSE for test set using cell sizes with', n, ' :', test_MSE_pool, '\n')

#with cell pooling

train_cell_hist_tensor = train_cell_hist.reshape([-1, 249, 1, 1])
test_cell_hist_tensor = test_cell_hist.reshape([-1, 249, 1, 1])
with tf.name_scope('inputs'):
    cell_hist = tf.placeholder(dtype=tf.float32, shape=[None, 249, 1, 1], name='cell_size_hist')

for n2 in range(100,180,5):
    cell_hist_pool = pooling(cell_hist, n2, n2)
    train_cell_hist_pool = sess.run(cell_hist_pool, feed_dict={cell_hist: train_cell_hist_tensor})

    train_feats=np.append(train_hist_pool,train_cell_hist_pool,axis=1)
    reg3 = LinearRegression().fit(train_feats, train_counts)
    pred_train3 = reg3.predict(train_feats)
    train_MSE_pool = mean_squared_error(train_counts, pred_train3)

    test_cell_hist_pool = sess.run(cell_hist_pool, feed_dict={cell_hist: test_cell_hist_tensor})
    test_feats=np.append(test_hist_pool,test_cell_hist_pool,axis=1)
    pred_test3 = reg3.predict(test_feats)
    test_MSE_pool = mean_squared_error(test_counts, pred_test3)
    print('MSE for train set using cell sizes with', n2, ' :', train_MSE_pool)
    print('MSE for test set using cell sizes with', n2, ' :', test_MSE_pool, '\n')
