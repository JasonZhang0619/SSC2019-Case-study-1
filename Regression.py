import tensorflow as tf
import numpy as np
import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


for F in ['F1','F23','F48']:
    for w in ['w1','w2']:
        print('For img at level',F,w)

        datasetF1w1=utils.get_traindata_level(F,w)
        def pooling(hist,window,step):
            with tf.name_scope('pooling'):
                hist_pool = tf.nn.avg_pool(hist, ksize=[1, window, 1, 1], strides=[1, step, 1, 1], padding='VALID')
                hist_pool = tf.keras.layers.Flatten()(hist_pool)
            return hist_pool


        # regression using hist without pooling
        trainset=datasetF1w1[:300]
        testset=datasetF1w1[300:]
        train_hist=np.array(trainset.iloc[:,:255])
        train_counts=np.array(trainset['count'])
        test_hist=np.array(testset.iloc[:,:255])
        test_counts=np.array(testset['count'])

        reg=LinearRegression().fit(train_hist,train_counts)
        pred_train=reg.predict(train_hist)
        train_MSE=mean_squared_error(train_counts,pred_train)
        pred_test=reg.predict(test_hist)
        test_MSE=mean_squared_error(test_counts,pred_test)

        print('MSE for train set without pooling:',train_MSE)
        print('MSE for test set without pooling:',test_MSE)

        #regression with pooling
        #using tf ave pool so some redundant codes for tf structure
        train_hist=np.array(trainset.iloc[:,:255]).reshape([-1,255,1,1])
        test_hist=np.array(testset.iloc[:,:255]).reshape([-1,255,1,1])
        with tf.name_scope('inputs'):
            hist = tf.placeholder(dtype=tf.float32, shape=[None, 255, 1, 1], name='image_hist')
            features = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='features')
            count = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='counts')

        hist_pool=pooling(hist,3,3)
        sess=tf.Session()

        train_hist_pool=sess.run(hist_pool,feed_dict={hist:train_hist})
        print(train_hist_pool.shape)
        reg=LinearRegression().fit(train_hist_pool,train_counts)
        pred_train=reg.predict(train_hist_pool)
        train_MSE_pool=mean_squared_error(train_counts,pred_train)

        test_hist_pool=sess.run(hist_pool,feed_dict={hist:test_hist})
        pred_test=reg.predict(test_hist_pool)
        test_MSE_pool=mean_squared_error(test_counts,pred_test)

        print('MSE for train set after pooling:',train_MSE_pool)
        print('MSE for test set after pooling:',test_MSE_pool,'\n')