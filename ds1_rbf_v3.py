import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random
import math

from tensorflow.python.framework import ops

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


# data를 원래 값 그대로 불러오는 함수
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


# data를 z-score 표준화로 불러오는 함수
def load_zdata(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    cols = list(zdata.columns)
    mean = zdata.iloc[0:, :31].mean()
    std = zdata.iloc[0:, :31].std(ddof=0)
    dmin = np.divide(np.subtract(zdata.iloc[0:, :31].min(), mean), std)
    dmax = np.divide(np.subtract(zdata.iloc[0:, :31].max(), mean), std)
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

    return (x_train_arr, t_train_arr), (x_test_arr, t_test_arr), int((array.shape[0] - 1) / 5), std, mean, dmin, dmax


# data를 scaling 변환해 불러오는 함수
def load_ndata(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    cols = list(zdata.columns)
    std = zdata.iloc[0:, :31].max() - zdata.iloc[0:, :31].min()
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

    return (x_train_arr, t_train_arr), (x_test_arr, t_test_arr), int((array.shape[0] - 1) / 5), std


# 예측용 data에 대해(X만 존재) 불러오는 함수
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


# RBF activation function을 만드는 부분
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


# RBF network 구조를 만드는 함수
# Modified version of rbf_network
def rbf_network(input_layer, weights):
    layer1 = tf_gaussian_function(tf.matmul(input_layer, weights['h1'])) + weights['bias_h1']
    # layer2 = tf_gaussian_function(tf.matmul(layer1, weights['h2']) + weights['bias_h2'])
    # output = tf.nn.sigmoid(tf.matmul(tf_gaussian_function(layer2), weights['output']) + weights['bias']) # 2 hidden layer를 사용하는 경우
    output = tf.matmul(layer1, weights['output']) + weights['bias_output']  # zdata
    return output


ops.reset_default_graph()

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

#(x_train, t_train), (x_test, t_test), test_size, std = load_ndata("MeanData.xlsx") # scaling 으로 학습시킬 경우
(x_train, t_train), (x_test, t_test), test_size, std, mean, dmin, dmax = load_zdata("MeanData.xlsx")

print(dmin)
print(dmax)
print(std[28:])
print(mean[28:])

# 학습 parameter setting
N_INSTANCES = x_train.shape[0]
N_INPUT = x_train.shape[1]
N_CLASSES = t_train.shape[1]
TRAIN_SIZE = x_test.shape[0]
batch_size = 256
iters_num = 10000
learning_rate = 0.001
display_step = 20
hidden_size = 200
hidden_size2 = 200

# tensor graph 생성
x_data = tf.placeholder(shape=[None, N_INPUT], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, N_CLASSES], dtype=tf.float32)

weights = {
    'h1': tf.Variable(tf.random_normal([N_INPUT, hidden_size], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([hidden_size, hidden_size2], stddev=0.1)),
    'output': tf.Variable(tf.random_normal([hidden_size2, N_CLASSES], stddev=0.1)),
    'bias_h1': tf.Variable(tf.random_normal([hidden_size], stddev=0.1)),
    'bias_h2': tf.Variable(tf.random_normal([hidden_size2], stddev=0.1)),
    'bias_output': tf.Variable(tf.random_normal([N_CLASSES], stddev=0.1))
}

pred = rbf_network(x_data, weights)

cost = tf.reduce_mean(tf.square(pred - y_target))
mae = tf.reduce_mean(tf.abs(pred - y_target))
my_opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# input x에 대한 gradient 계산을 위한 layer 구조 생성
one_layer = tf.Variable(tf.ones([1, N_INPUT]), dtype=tf.float32)
y_true = tf.placeholder(shape=[None, N_CLASSES], dtype=tf.float32)
x_weight = tf.Variable(tf.linalg.diag(tf.random_normal([N_INPUT], stddev=0.5)))
x_layer = tf.matmul(one_layer, x_weight)

x_pred = rbf_network(x_layer, weights)

cost_x = tf.reduce_mean(tf.square(x_pred - y_true))
mae_x = tf.reduce_mean(tf.abs(x_pred - y_true))
opt_x = tf.train.AdamOptimizer(learning_rate).minimize(cost_x, var_list=[x_weight])
#

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

tol = 1e-1
epoch, err = 0, 1
case_cost = []
train_cost_list = []

# Training loop - X->Y에 대한 network를 학습
while epoch <= iters_num:
    batch_mask = np.random.choice(N_INSTANCES, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    sess.run(my_opt, feed_dict={x_data: x_batch, y_target: t_batch})
    avg_cost, avg_mae = sess.run([cost, mae], feed_dict={x_data: x_batch, y_target: t_batch})
    train_cost_list.append(avg_cost)
    err = avg_cost
    if epoch % 10 == 0:
        print("Epoch: {}/{} err = {}".format(epoch, iters_num, avg_cost))
        print("mae: {}".format(avg_mae))

    epoch += 1

#test set에 대해 case 각각의 MAE를 계산하는 부분
for i in range(test_size):
    x_case = []
    t_case = []
    x_case.append(x_test[i])
    t_case.append(t_test[i])

    test_mae = sess.run(mae, feed_dict={x_data: x_case, y_target: t_case})
    case_cost.append(test_mae)

feature_MAE = [0, 0, 0]

#test set에 대해 feature 각각의 MAE를 계산하는 부분
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

print("End of learning process")
print("Final epoch = {}/{} ".format(epoch, iters_num))
print("Final error = {}".format(err))

# random position X'에서 출발해 Y의 결과값을 갖는 X를 탐색하는 부분.
candidate = []
candidate_mae = []
truth_mae = []
x_epoch = 1
batch_mask = np.random.choice(N_INSTANCES, 1) # tensor shape을 맞춰주기 위한 부분([N] 배열을 [N, 1]의 모양으로)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(t_batch.shape)
# 100번의 random position sampling
for i in range(20):
    print("sample number:{}".format(i))
    rand_input = tf.random_normal([N_INPUT], stddev=1).eval()
    for i in range(16):
        print(x_batch[0][i])
        rand_input[i] = x_batch[0][i]
    sess.run(x_weight.assign(tf.linalg.diag(rand_input)))
    print("initial random x:{}".format(tf.diag_part(x_weight).eval()))
    # 각 positon X'마다 1000회 학습
    for x_epoch in range(1001):
        learning_rate = 0.01
        if x_epoch > 500:
            learning_rate = 0.001

        result, _ = sess.run([x_pred, opt_x], feed_dict={y_true: t_batch})
        avg_cost, avg_mae = sess.run([cost_x, mae_x], feed_dict={y_true: t_batch})
        train_cost_list.append(avg_cost)
        err = avg_cost
        tmp = np.array(x_weight.eval())
        tmp = tmp.reshape(-1, N_INPUT)
        for i in range(N_INPUT):
            for j in range(N_INPUT):
                if i != j:
                    tmp[i][j] = 0
                elif i == j:
                    if dmin[i] != dmin[i]:
                        tmp[i][j] = 0
                        continue
                    elif i < 16:
                        tmp[i][j] = x_batch[0][i]
                    elif tmp[i][j] <= dmin[i]:
                        tmp[i][j] = dmin[i]
                    elif tmp[i][j] >= dmax[i]:
                        tmp[i][j] = dmax[i]

        sess.run(x_weight.assign(tmp))

        if avg_mae <= 0.000001:
            print(tf.diag_part(x_weight).eval())
            print(x_batch)
            print(avg_mae)

            print("result:{}".format(result))
            print("Gtruth:{}".format(t_batch))
            break

        if x_epoch % 1000 == 0:
            print(tf.diag_part(x_weight).eval())
            print(x_batch)
            print(avg_mae)

            print("result:{}".format(result))
            print("Gtruth:{}".format(t_batch))

    x = tf.diag_part(x_weight).eval()
    for i in range(28):
        x[i] = x[i] * std[i] + mean[i]
    candidate.append(x)
    candidate_mae.append(avg_mae)
    truth_mae.append(tf.reduce_mean(tf.abs(tf.diag_part(x_weight)-x_batch.reshape(28))).eval())

# search 결과 출력
for i in range(len(candidate)):
    print("candidate:{}".format(candidate[i]))
    print("err:{}".format(candidate_mae[i]))
    print("truth_mae:{}".format(truth_mae[i]))

sess.close()