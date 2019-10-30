import utils
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

class PoolRegressor:
    def __init__(self,window=10,step=10,pool=True):
        self.window=window
        self.step=step
        self.pool=pool
        self.regressor=LinearRegression()

    def pooling(self, hist):
        hist_pool=[]
        for start in range(0,255,self.step):
            hist_pool.append(np.mean(hist[:,start:start+self.window],axis=1))
        hist_pool=np.vstack(hist_pool).T
        return hist_pool

    def train(self, Xtrain, ytrain):
        if (self.pool): Xtrain=self.pooling(Xtrain)
        self.fit = self.regressor.fit(Xtrain, ytrain)

    def predict(self,Xtest):
        if (self.pool): Xtest=self.pooling(Xtest)
        return self.fit.predict(Xtest)

#pool_dict={'F1w1':34,'F1w2':14,'F23w1':16,'F23w2':32,'F48w1':14,'F48w2':60}
#with topology

# train_feats=np.append(train_hist_pool,train_cell_hist,axis=1)
# reg2 = LinearRegression().fit(train_feats, train_counts)
# pred_train2 = reg2.predict(train_feats)
#
# test_feats=np.append(test_hist_pool,test_cell_hist,axis=1)
# pred_test2 = reg2.predict(test_feats)
# test_MSE_pool = mean_squared_error(test_counts, pred_test2)
# train_MSE_pool = mean_squared_error(train_counts, pred_train2)
# print('MSE for train set using cell sizes with', n, ' :', train_MSE_pool)
# print('MSE for test set using cell sizes with', n, ' :', test_MSE_pool, '\n')
#
# #with cell pooling
#
# train_cell_hist_tensor = train_cell_hist.reshape([-1, 249, 1, 1])
# test_cell_hist_tensor = test_cell_hist.reshape([-1, 249, 1, 1])
# with tf.name_scope('inputs'):
#     cell_hist = tf.placeholder(dtype=tf.float32, shape=[None, 249, 1, 1], name='cell_size_hist')
#
# for n2 in range(100,180,5):
#     cell_hist_pool = pooling(cell_hist, n2, n2)
#     train_cell_hist_pool = sess.run(cell_hist_pool, feed_dict={cell_hist: train_cell_hist_tensor})
#
#     train_feats=np.append(train_hist_pool,train_cell_hist_pool,axis=1)
#     reg3 = LinearRegression().fit(train_feats, train_counts)
#     pred_train3 = reg3.predict(train_feats)
#     train_MSE_pool = mean_squared_error(train_counts, pred_train3)
#
#     test_cell_hist_pool = sess.run(cell_hist_pool, feed_dict={cell_hist: test_cell_hist_tensor})
#     test_feats=np.append(test_hist_pool,test_cell_hist_pool,axis=1)
#     pred_test3 = reg3.predict(test_feats)
#     test_MSE_pool = mean_squared_error(test_counts, pred_test3)
#     print('MSE for train set using cell sizes with', n2, ' :', train_MSE_pool)
#     print('MSE for test set using cell sizes with', n2, ' :', test_MSE_pool, '\n')
