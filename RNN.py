import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from random import shuffle

data_path = "./data"
data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

layer_num = 3
batch_size = 20
n_step = 30
num_units = 50
filter_num = 10
vector_size = 3
kernel_size = 10

X = tf.placeholder(dtype=tf.float32, shape=[None, n_step, vector_size])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
global_step = tf.Variable(0, False, name='global_step')
increment_global_step_op = tf.assign(global_step, global_step+1)
with tf.name_scope('layer1'):
    RNN_list = [tf.contrib.rnn.GRUCell(num_units=num_units) for i in range(layer_num)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(RNN_list)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
with tf.name_scope('layer2'):
    state = tf.reshape(tf.concat(states, axis=1), shape=[batch_size,num_units*layer_num ,1])
    afterConv1d = tf.reshape(tf.layers.conv1d(inputs=state, filters=filter_num, kernel_size=kernel_size, padding='SAME'), shape=[batch_size, num_units*layer_num*filter_num])
    result = tf.layers.dense(afterConv1d, 1)
    answer = Y
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(answer - result))
with tf.name_scope('op'):
    optimzer = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)


train_resume = True
iter = 10
epochs = 2500

with tf.Session() as sess:
    saver= tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if train_resume:
        saver.restore(sess, "./model.ckpt")

    loss_arr = []
    step_arr = []
    current_step = sess.run(global_step)
    for epoch in range(epochs):
        print(epoch, " epoch")
        shuffle(data_files)
        total_loss = 0
        for f in data_files:
            f = data_path + "/" + f
            df = pd.read_csv(f)
            dflen = len(df)
            loss_avg = 0
            for i in range(iter):
                data = None
                for _ in range(batch_size):
                    r = np.random.randint(low=0, high=len(df) - (n_step+1))
                    datum = np.asarray(df[r:r + (n_step+1)])[:, [5, 7, 9]]
                    if data is None:
                        data = datum
                    else:
                        data = np.concatenate((data, datum))
                data = data.reshape([batch_size, (n_step+1), vector_size])
                data_stddev = np.std(data[:, :(n_step), :], axis=1).reshape([batch_size,1,  vector_size]) #(batch_size, 1, vector_size)
                data_mean = np.mean(data[:, :(n_step), :], axis=1).reshape([batch_size,1, vector_size]) #(batch_size, 1, vector_size)
                _X = (data[:, :(n_step), :] - data_mean)/data_stddev
                _Y = data[:, (n_step):(n_step+1), 1]
                _, tmp = sess.run([optimzer, loss], feed_dict={X:_X, Y:_Y})
                loss_avg += tmp
            loss_avg /= iter
            total_loss += loss_avg
        total_loss /= len(data_files)
        current_step = sess.run(global_step)
        sess.run([increment_global_step_op])
        loss_arr.append(total_loss)
        step_arr.append(current_step)
        fig, ax = plt.subplots()
        ax.plot(step_arr, loss_arr)
        ax.set(xlabel='global_steps', ylabel='average_loss', title='Loss')
        fig.savefig('loss.png')
        plt.close(fig)
        print("epoch: ", epoch, "\ntotal loss: ", total_loss)
        saver.save(sess, "./model.ckpt")
