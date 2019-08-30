import random
import pandas as pd
import numpy as np
import os
import time
import gpflow as gpf
import tensorflow as tf

# gpflow는 다양한 확률 분포를 학습할 수 있도록 지원하는 외부 package

# gpflow의 경우 output feature 3개를 하나의 model로 학습하게 되면 각각의 std deviation의 분포가 같게 학습됨(y1, y2, y3의 std값이 input point X1에서 하나의 같은 값을 가짐).
# 이를 해결하기 위해 y1, y2, y3을 따로 학습하고, 이 형태에 맞게 output feature를 1개씩 나누어서 load
# 학습 정확도를 위해 z-score 표준화
def load_zdata_y1(file_name):
    data = pd.read_excel(file_name)
    zdata = data.iloc[0:, 1:]
    cols = list(zdata.columns)
    std = zdata.iloc[0:, :31].std(ddof=0)
    mean = zdata.iloc[0:, :31].mean()
    print(std.shape[0])
    for col in cols:
        col_zscore = col + '_zscore'
        if zdata[col].std(ddof=0) == 0:
            zdata[col_zscore] = 0
        else:
            zdata[col_zscore] = (zdata[col] - zdata[col].mean()) / zdata[col].std(ddof=0)
    zdata = zdata.iloc[0:, 31:]
    array = np.array(zdata)

    x_train, t_train1, t_train2, t_train3, x_test, t_test1, t_test2, t_test3 = [], [], [], [], [], [], [], []

    test_number = random.sample(range(0, array.shape[0]), int((array.shape[0] - 1) / 5))
    test_number.sort()
    idx = 0
    for i in range(array.shape[0]):  # test case
        if idx < int((array.shape[0] - 1) / 5) and i == test_number[idx]:
            idx = idx + 1
            x_test.append(array[i][:28])
            t_test1.append(array[i][28:29])
            t_test2.append(array[i][29:30])
            t_test3.append(array[i][30:31])
        else:  # train case
            x_train.append(array[i][:28])
            t_train1.append(array[i][28:29])
            t_train2.append(array[i][29:30])
            t_train3.append(array[i][30:31])

    x_train_arr = np.array(x_train)
    t_train_arr1 = np.array(t_train1)
    t_train_arr2 = np.array(t_train2)
    t_train_arr3 = np.array(t_train3)
    x_test_arr = np.array(x_test)
    t_test_arr1 = np.array(t_test1)
    t_test_arr2 = np.array(t_test2)
    t_test_arr3 = np.array(t_test3)

    return (x_train_arr, t_train_arr1, t_train_arr2, t_train_arr3), (x_test_arr, t_test_arr1, t_test_arr2, t_test_arr3), int((array.shape[0] - 1) / 5), mean, std


# CPU time check용 코드. 삭제해도 됨.
start_time = time.time()

# load하고, gpflow 에러가 나지 않도록 data type 설정해주는 부분
(x_train, y_train1, y_train2, y_train3), (x_test, y_test1, y_test2, y_test3), _, mean, std = load_zdata_y1("MeanData.xlsx")
x_train, y_train1, y_train2, y_train3 = x_train.astype(gpf.settings.float_type), y_train1.astype(gpf.settings.float_type), y_train2.astype(gpf.settings.float_type), y_train3.astype(gpf.settings.float_type)
x_test, y_test1, y_test2, y_test3 = x_test.astype(gpf.settings.float_type), y_test1.astype(gpf.settings.float_type), y_test2.astype(gpf.settings.float_type), y_test3.astype(gpf.settings.float_type)

# bagging 적용을 위한 코드. 전체 training set의 1/20만큼(1000보다 큰 값이 나올 경우 1000개)을 하나의 모델 학습에 사용.
resample_size = min(1000, (int)(x_train.shape[0]/20))
gp1_list, gp2_list, gp3_list = [], [], []
z_size = min(1000, (int)(x_train.shape[0]/10))
bagging_size = 5

tf_graph = tf.Graph()
tf_session = tf.Session(graph=tf_graph)

# 학습한 모델을 불러오는 코드. 지정된 경로에 저장된 gp model이 있는 경우를 체크하고 있으면 모델을 불러온다.
if os.path.exists('/tmp/gp0_0'):
    for i in range(3):
        for j in range(bagging_size):
            gp = gpf.saver.Saver().load('/tmp/gp{}_{}'.format(i, j))
            if i == 0:
                gp1_list.append(gp)
            elif i == 1:
                gp2_list.append(gp)
            elif i == 2:
                gp3_list.append(gp)

# 없으면 학습을 진행한다.
else:
    for i in range(bagging_size):
        print("Sample number " + str(i))
        resample_mask = np.random.choice(x_train.shape[0], resample_size)
        z_mask = np.random.choice(x_train.shape[0], z_size)
        x_resample = x_train[resample_mask]
        y_resample1 = y_train1[resample_mask]
        y_resample2 = y_train2[resample_mask]
        y_resample3 = y_train3[resample_mask]
        z_resample = x_train[z_mask]
        with gpf.defer_build():
            gp = gpf.models.SVGP(x_resample, y_resample1, kern=gpf.kernels.RBF(28), likelihood=gpf.likelihoods.Gaussian(), Z=x_resample)
            gp2 = gpf.models.SVGP(x_resample, y_resample2, kern=gpf.kernels.RBF(28), likelihood=gpf.likelihoods.Gaussian(), Z=x_resample)
            gp3 = gpf.models.SVGP(x_resample, y_resample3, kern=gpf.kernels.RBF(28), likelihood=gpf.likelihoods.Gaussian(), Z=x_resample)

        gpf.train.ScipyOptimizer().minimize(gp)
        gpf.train.ScipyOptimizer().minimize(gp2)
        gpf.train.ScipyOptimizer().minimize(gp3)

        gp1_list.append(gp)
        gp2_list.append(gp2)
        gp3_list.append(gp3)

# 엑셀 저장용 형식을 만들기 위한 array shape 지정. bagging 모델 각각의 출력값을 더한 후 모델의 개수만큼 나눠야 하므로, 예측하고 싶은 test set의 길이만큼의 0 배열을 만들어줌.
y_pred1, y_pred2, y_pred3, std1, std2, std3 = np.zeros(x_test.shape[0]), np.zeros(x_test.shape[0]), np.zeros(x_test.shape[0]), np.zeros(x_test.shape[0]), np.zeros(x_test.shape[0]), np.zeros(x_test.shape[0])

# 각 모델별로 예측값을 출력하고 배열에 더하는 부분
for gp1 in gp1_list:
    p_pred1, p_std1 = gp1.predict_y(x_test)
    p_pred1 = p_pred1.reshape(-1)
    p_std1 = p_std1.reshape(-1)
    y_pred1 = np.add(y_pred1, p_pred1)
    std1 = np.add(std1, p_std1)

for gp2 in gp2_list:
    p_pred2, p_std2 = gp2.predict_y(x_test)
    p_pred2 = p_pred2.reshape(-1)
    p_std2 = p_std2.reshape(-1)
    y_pred2 = np.add(y_pred2, p_pred2)
    std2 = np.add(std2, p_std2)

for gp3 in gp3_list:
    p_pred3, p_std3 = gp3.predict_y(x_test)
    p_pred3 = p_pred3.reshape(-1)
    p_std3 = p_std3.reshape(-1)
    y_pred3 = np.add(y_pred3, p_pred3)
    std3 = np.add(std3, p_std3)

# 더한 값들을 모델 개수만큼 나눠주면 원하는 최종값이 나온다.
y_pred1 /= len(gp1_list)
y_pred2 /= len(gp1_list)
y_pred3 /= len(gp1_list)
std1 /= len(gp1_list)
std2 /= len(gp1_list)
std3 /= len(gp1_list)

# np.hstack의 형식을 위해서 reshape해주는 부분.
std1 = std1.reshape(-1, 1)
std2 = std2.reshape(-1, 1)
std3 = std3.reshape(-1, 1)
y_test1 = y_test1.reshape(-1, 1)
y_test2 = y_test2.reshape(-1, 1)
y_test3 = y_test3.reshape(-1, 1)
y_pred1 = y_pred1.reshape(-1, 1)
y_pred2 = y_pred2.reshape(-1, 1)
y_pred3 = y_pred3.reshape(-1, 1)


# 연구 실험 분석을 위한 출력구문. 삭제해도 됨.
# MAE_y1 = tf.reduce_mean(np.abs(np.subtract(y_pred1, y_test1)))
# MAE_y2 = tf.reduce_mean(np.abs(np.subtract(y_pred2, y_test2)))
# MAE_y3 = tf.reduce_mean(np.abs(np.subtract(y_pred3, y_test3)))
# sess = tf.InteractiveSession()
#
# print("MAE error:")
#
# print(MAE_y1.eval())
# print(MAE_y2.eval())
# print(MAE_y3.eval())
#
# print(MAE_y1.eval()*std[28])
# print(MAE_y2.eval()*std[29])
# print(MAE_y3.eval()*std[30])
#
# sess.close()

# hstack으로 하나씩 예측한 결과값들을 붙임. (hstack의 경우 row가 같은 배열
# 결과 file은 28개의 input + y1 예측값(평균) + y1 std값 + y2 예측값 + y2 std값 + y3 예측값 + y3 std값 + y1 y2 y3 실제값의 모양으로 생성
print("write shape:")
x_test = np.hstack((x_test, y_pred1)) #
x_test = np.hstack((x_test, std1))
x_test = np.hstack((x_test, y_pred2))
x_test = np.hstack((x_test, std2))
x_test = np.hstack((x_test, y_pred3))
x_test = np.hstack((x_test, std3))
x_test = np.hstack((x_test, y_test1))
x_test = np.hstack((x_test, y_test2))
x_test = np.hstack((x_test, y_test3))
print(x_test.shape)

# excel 형태로 저장하는 코드
pd_data = pd.DataFrame(x_test)
writer = pd.ExcelWriter("output1.xlsx")
pd_data.to_excel(writer, sheet_name='sheet1')
writer.save()

# model save용 부분 - gpflow saver의 경우 overwrite할 때 error가 생기므로 기존에 저장된 모델을 지워야 함.
for i in range(3):
    for j in range(bagging_size):
        if os.path.exists('/tmp/gp{}_{}'.format(i, j)):
            os.remove('/tmp/gp{}_{}'.format(i, j))

saver = gpf.train.Saver()


# model save용 부분 - 각 bagging 모델별로 저장해줌
i = 0
for gp in gp1_list:
    model_name = '/tmp/gp0_{}'.format(i)
    saver.save(model_name, gp)
    i += 1

i = 0
for gp2 in gp2_list:
    model_name = '/tmp/gp1_{}'.format(i)
    saver.save(model_name, gp2)
    i += 1

i = 0
for gp3 in gp3_list:
    model_name = '/tmp/gp2_{}'.format(i)
    saver.save(model_name, gp3)
    i += 1

# CPU time check용 코드. 삭제해도 됨.
print("=== %s seconds ===" % (time.time()-start_time))
