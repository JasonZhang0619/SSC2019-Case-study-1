import tensorflow as tf
import utils
import numpy as np

images=tf.placeholder(dtype=tf.float32, shape=[None,520,696,1],name='input_image')
features=tf.placeholder(dtype=tf.float32,shape=[None,3],name='factors')
counts=tf.placeholder(dtype=tf.float32,shape=[None,1],name='counts')


def inference(images):
    #shrink the images first
    images_small=tf.nn.avg_pool(images, ksize=[1, 10, 10, 1], strides=[1, 5, 5, 1], padding='VALID')
    with tf.variable_scope('cov1'):
        cov1=tf.layers.conv2d(images_small,filters=10,kernel_size=[5,5],strides=[3,3],padding='VALID',
                              activation=tf.nn.relu)
        pool1=tf.nn.max_pool(cov1,ksize=[1,6,6,1],strides=[1,3,3,1],padding='SAME')

    with tf.variable_scope('conv2'):
        conv2=tf.layers.conv2d(pool1,filters=20,kernel_size=[5,5],strides=[1,1],padding='SAME',
                               activation=tf.nn.relu)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 6, 6, 1], strides=[1, 3, 3, 1], padding='SAME')

    #funtional full connected layer
    with tf.variable_scope('fun') as scope:
        reshape=tf.keras.layers.Flatten()(pool2)
        h1 = tf.layers.dense(reshape,10,
                             activation=tf.nn.relu)

    #output layer
    with tf.variable_scope('fun2') as scope:
        y = tf.layers.dense(h1,1)
    return y

y=inference(images)
loss=tf.losses.mean_squared_error(y,counts)
tf.summary.scalar('MSE',loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/"+'train/', sess.graph)
writer_test = tf.summary.FileWriter("logs/"+'test/')
init = tf.global_variables_initializer()
sess.run(init)

#load training set
all_images, all_blur, all_stain, all_labels = utils.read_trainimgset(F='F1',w='w1')
training_images=all_images[:300]
training_labels=all_labels[:300]
test_images=all_images[300:]
test_labels=all_labels[300:]

size=100
j=0
for _ in range(1000):
    j+=1
    slice=np.random.choice(training_images.shape[0],size,replace=False)
    batch_images=training_images[slice]
    batch_labels=training_labels[slice]
    sess.run(train_step, feed_dict={images: batch_images, counts: batch_labels})
    if j % 100 == 0:
        print(sess.run(loss,feed_dict={images: batch_images, counts: batch_labels}))
        train_loss=sess.run(merged,feed_dict={images: batch_images, counts: batch_labels})
        writer.add_summary(train_loss,j)
        test_loss = sess.run(merged, feed_dict={images: test_images, counts: test_labels})
        writer_test.add_summary(test_loss,j)
