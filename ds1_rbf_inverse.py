
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random
import math

from datetime import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops

def load_data(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    array = np.array(data.iloc[0:, 1:])

    x_train, t_train, x_test, t_test = [], [], [], []

    test_number = random.sample(range(0, array.shape[0]), int((array.shape[0] - 1) / 5))
    test_number.sort()
    idx = 0
    for i in range(array.shape[0]):  # test case
        if idx < int((array.shape[0] - 1) / 5) and i == test_number[idx]:
            idx = idx + 1
            x_test.append(array[i][:28])
            t_test.append(array[i][28:])
        else:  # train case
            x_train.append(array[i][:28])
            t_train.append(array[i][28:])

    x_train_arr = np.array(x_train)
    t_train_arr = np.array(t_train)
    x_test_arr = np.array(x_test)
    t_test_arr = np.array(t_test)

    return (x_train_arr, t_train_arr), (x_test_arr, t_test_arr), int((array.shape[0] - 1) / 5)


def load_zdata(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    cols = list(zdata.columns)
    std = zdata.iloc[0:, :31].std(ddof=0)
    print(std.shape[0])
    for col in cols:
        col_zscore = col + '_zscore'
        if zdata[col].std(ddof=0) == 0:
            zdata[col_zscore] = 0
        else:
            zdata[col_zscore] = (zdata[col] - zdata[col].mean()) / zdata[col].std(ddof=0)
    zdata = zdata.iloc[0:, 31:]
    array = np.array(zdata)

    x_train, t_train, x_test, t_test = [], [], [], []

    test_number = random.sample(range(0, array.shape[0]), int((array.shape[0] - 1) / 5))
    test_number.sort()
    idx = 0
    for i in range(array.shape[0]):  # test case
        if idx < int((array.shape[0] - 1) / 5) and i == test_number[idx]:
            idx = idx + 1
            x_test.append(array[i][:28])
            t_test.append(array[i][28:])
        else:  # train case
            x_train.append(array[i][:28])
            t_train.append(array[i][28:])

    x_train_arr = np.array(x_train)
    t_train_arr = np.array(t_train)
    x_test_arr = np.array(x_test)
    t_test_arr = np.array(t_test)

    return (x_train_arr, t_train_arr), (x_test_arr, t_test_arr), int((array.shape[0] - 1) / 5), std


def load_ndata(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    cols = list(zdata.columns)

    for col in cols:
        col_norm = col + '_norm'
        if (zdata[col].max()-zdata[col].min()) == 0:
            zdata[col_norm] = 0
        else:
            zdata[col_norm] = (zdata[col] - zdata[col].min()) / (zdata[col].max()-zdata[col].min())
    zdata = zdata.iloc[0:, 31:]
    array = np.array(zdata)

    x_train, t_train, x_test, t_test = [], [], [], []

    test_number = random.sample(range(0, array.shape[0]), int((array.shape[0] - 1) / 5))
    test_number.sort()
    idx = 0
    for i in range(array.shape[0]):  # test case
        if idx < int((array.shape[0] - 1) / 5) and i == test_number[idx]:
            idx = idx + 1
            x_test.append(array[i][:28])
            t_test.append(array[i][28:])
        else:  # train case
            x_train.append(array[i][:28])
            t_train.append(array[i][28:])

    x_train_arr = np.array(x_train)
    t_train_arr = np.array(t_train)
    x_test_arr = np.array(x_test)
    t_test_arr = np.array(t_test)

    return (x_train_arr, t_train_arr), (x_test_arr, t_test_arr), int((array.shape[0] - 1) / 5)


def preload_data(file_name):
    data = pd.read_excel(file_name)
    array = np.array(data.iloc[0:, 1:])
    x_predict, t_predict = [], []

    idx = 0
    for i in range(array.shape[0]):  # test case
        x_predict.append(array[i][:28])
        t_predict.append(array[i][28:])

    x_predict_arr = np.array(x_predict)
    t_predict_arr = np.array(t_predict)

    return x_predict_arr, t_predict_arr


def preload_zdata(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    cols = list(zdata.columns)
    std = zdata.iloc[0:, 31:].std(ddof=0)
    for col in cols:
        col_zscore = col + '_zscore'
        if zdata[col].std(ddof=0) == 0:
            zdata[col_zscore] = 0
        else:
            zdata[col_zscore] = (zdata[col] - zdata[col].mean()) / zdata[col].std(ddof=0)
    zdata = zdata.iloc[0:, 31:]
    array = np.array(zdata)
    x_predict, t_predict = [], []

    idx = 0
    for i in range(array.shape[0]):  # test case
        x_predict.append(array[i][:28])
        t_predict.append(array[i][28:])

    x_predict_arr = np.array(x_predict)
    t_predict_arr = np.array(t_predict)

    return x_predict_arr, t_predict_arr, std


def preload_ndata(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    cols = list(zdata.columns)
    min = zdata.iloc[0:, 28:].max() - zdata.iloc[0:, 28:].min()
    for col in cols:
        col_norm = col + '_norm'
        if (zdata[col].max() - zdata[col].min()) == 0:
            zdata[col_norm] = 0
        else:
            zdata[col_norm] = (zdata[col] - zdata[col].min()) / (zdata[col].max() - zdata[col].min())
    zdata = zdata.iloc[0:, 31:]
    array = np.array(zdata)
    x_predict, t_predict = [], []

    idx = 0
    for i in range(array.shape[0]):  # test case
        x_predict.append(array[i][:28])
        t_predict.append(array[i][28:])

    x_predict_arr = np.array(x_predict)
    t_predict_arr = np.array(t_predict)

    return x_predict_arr, t_predict_arr, min


# creates activation function
def gaussian_function(input_layer):
    initial = math.exp(-1*math.pow(input_layer, 2))
    return initial


np_gaussian_function = np.vectorize(gaussian_function)


def d_gaussian_function(input_layer):
    initial = -4 * input_layer * math.exp(-2*math.pow(input_layer, 2))
    return initial


np_d_gaussian_function = np.vectorize(d_gaussian_function)
np_d_gaussian_function_32 = lambda input_layer: np_d_gaussian_function(input_layer).astype(np.float32)


def tf_d_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "d_gaussian_function", [input_layer]) as name:
        y = tf.py_func(np_d_gaussian_function_32, [input_layer], [tf.float32], name=name, stateful=False)
    return y[0]


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFunGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def gaussian_function_grad(op, grad):
    input_variable = op.inputs[0]
    n_gr = tf_d_gaussian_function(input_variable)
    return grad * n_gr


np_gaussian_function_32 = lambda input_layer: np_gaussian_function(input_layer).astype(np.float32)


def tf_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "gaussian_function", [input_layer]) as name:
        y = py_func(np_gaussian_function_32, [input_layer], [tf.float32], name=name, grad=gaussian_function_grad)
    return y[0]
# end of defining activation function

#
# def rbf_network(input_layer, weights):
#     layer1 = tf.matmul(tf_gaussian_function(input_layer), weights['h1'])
#     layer2 = tf.matmul(tf_gaussian_function(layer1), weights['h2'])
#     #output = tf.matmul(tf_gaussian_function(layer1), weights['output'])
#     output = tf.matmul(layer1, weights['output']) + weights['bias']
#     return output


# Modified version of rbf_network
def rbf_network(input_layer, weights):
    layer1 = tf_gaussian_function(tf.matmul(input_layer, weights['h1']))
    layer2 = tf_gaussian_function(tf.matmul(layer1, weights['h2']))
    output = tf.nn.sigmoid(tf.matmul(tf_gaussian_function(layer1), weights['output']))
    #output = tf_gaussian_function(tf.matmul(layer2, weights['output']) + weights['bias'])
    return output

ops.reset_default_graph()

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# (x_train, t_train), (x_test, t_test), test_size = load_data("MeanData.xlsx")
(t_train, x_train), (t_test, x_test), test_size, std = load_zdata("KNN impute.xlsx")

N_INSTANCES = x_train.shape[0]
N_INPUT = x_train.shape[1]
N_CLASSES = t_train.shape[1]
TEST_SIZE = 0.2
TRAIN_SIZE = x_test.shape[0]
batch_size = 256
iters_num = 100000
learning_rate = 0.001
display_step = 20
hidden_size = 500
hidden_size2 = 500

x_data = tf.placeholder(shape=[None, N_INPUT], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, N_CLASSES], dtype=tf.float32)

weights = {
    'h1': tf.Variable(tf.random_normal([N_INPUT, hidden_size], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([hidden_size, hidden_size2], stddev=0.1)),
    'output': tf.Variable(tf.random_normal([hidden_size2, N_CLASSES], stddev=0.1)),
    'bias': tf.Variable(tf.random_normal([N_CLASSES], stddev=0.1))
}

pred = rbf_network(x_data, weights)

cost = tf.reduce_mean(tf.square(pred - y_target))
mae = tf.reduce_mean(tf.abs(pred - y_target))
my_opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_target, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

tol = 1e-1
epoch, err = 0, 1
case_cost = []
train_cost_list = []
# Training loop
while epoch <= iters_num:
    batch_mask = np.random.choice(N_INSTANCES, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    sess.run(my_opt, feed_dict={x_data: x_batch, y_target: t_batch})
    avg_cost = sess.run(cost, feed_dict={x_data: x_batch, y_target: t_batch})
    train_cost_list.append(avg_cost)
    err = avg_cost
    if epoch % 10 == 0:
        print("Epoch: {}/{} err = {}".format(epoch, iters_num, avg_cost))
        # print(weights['output'].eval())

    epoch += 1

for i in range(test_size):
    x_case = []
    t_case = []
    x_case.append(x_test[i])
    t_case.append(t_test[i])

    test_mae = sess.run(mae, feed_dict={x_data: x_case, y_target: t_case})
    case_cost.append(test_mae)

feature_MAE = [0, 0, 0]

for i in range(test_size):
    x_case = []
    t_case = []
    x_case.append(x_test[i])
    t_case.append(t_test[i])
    test_feature = sess.run(pred, feed_dict={x_data: x_case, y_target: t_case})
    Y_feature = np.absolute(test_feature - t_case)
    Y_feature = np.array(Y_feature).reshape(3)

    feature_MAE[0] += Y_feature[0] * std[28]
    feature_MAE[1] += Y_feature[1] * std[29]
    feature_MAE[2] += Y_feature[2] * std[30]
    feature_MAE = np.array(feature_MAE)

feature_MAE /= test_size
print(str((feature_MAE[0] + feature_MAE[1] + feature_MAE[2]) / 3))

ax = plt.subplot(221)
ax.set_title("Train MAE loss")
x = np.arange(1, epoch+1, 1)
graph_list = np.array(train_cost_list)
plt.plot(x, np.array(graph_list))
plt.grid(linestyle='--')
plt.legend()
ax = plt.subplot(223)
ax.set_title("Case MAE")
x = np.arange(1, test_size+1)
plt.scatter(x, np.array(case_cost), s=5, color='c')
plt.grid(linestyle='--')
plt.legend()
ax = plt.subplot(133)
ax.set_title("Feature MAE")
x = np.arange(1, 4)
plt.bar(x, np.array(feature_MAE), color='r')
for a, b in zip(x, feature_MAE):
    plt.text(a, b, str(round(b, 5)))
plt.grid(linestyle='--')
plt.legend()
plt.show()
print("End of learning process")
print("Final epoch = {}/{} ".format(epoch, iters_num))
print("Final error = {}".format(err))
while True:
    tidx = int(input("test number : "))
    testing, predicting = [], []
    testing.append(x_test[tidx])
    predicting.append(t_test[tidx])
    p_predict = sess.run(pred, feed_dict={x_data: testing, y_target: predicting})
    print(p_predict)
    print(t_test[tidx])
sess.close()