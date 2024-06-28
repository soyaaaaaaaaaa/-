
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import sys
import h5py

# sigmoid
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

#  计算sigmoid激活函数的梯度，用于反向传播中更新权重
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


# relu函数 和反向求导
def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

#  计算relu激活函数的梯度，用于反向传播中更新权重
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# Softmax
def softmax(Z):
    A = np.exp(Z)/np.sum(np.exp(Z),axis=0)
    cache = Z
    return A, cache
# 初始化w
# def INIT_W(n_x,n_h1,n_h2,n_y):
#     W1 = np.random.randn(n_h1, n_x ) * 0.01
#     b1 = np.zeros((n_h1, 1))
#     W2 = np.random.randn(n_h2,n_h1)*0.01
#     b2 = np.zeros((n_h2,1))
#     W3 = np.random.randn(n_y, n_h2) * 0.01
#     b3 = np.zeros((n_y, 1))
#     INIT = {
#         "W1" : W1,
#         "b1" : b1,
#         "W2" : W2,
#         "b2" : b2,
#         "W3" : W3,
#         "b3" : b3
#     }
#     return INIT


def init_W(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters



# 向前
def L_forword_sum(W,A,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)
    return Z,cache

def L_activate_forworld(A_prev,W,b,activation):
    if activation == "relu":
        Z ,linear_cache =  L_forword_sum(W,A_prev,b)
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        Z, linear_cache = L_forword_sum(W, A_prev, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "softmax":
        Z, linear_cache = L_forword_sum(W, A_prev, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A,cache


def L_forword(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = L_activate_forworld(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    # 最后一层使用softmax
    AL, cache  = L_activate_forworld(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)
    return AL, caches

#计算代价
def cost(Y_out,Y):
    cost = -np.sum(np.multiply(np.log(Y_out), Y)) / Y_out.shape[1]
    cost = np.squeeze(cost)
    return cost

#线性返回
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, Y,activation="relu"):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = dA - Y
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches,case):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    if case == "softmax":
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(AL, current_cache,Y,"softmax")

    elif case  == "sigmoid":
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(AL, current_cache,Y, "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, Y ,"relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # 整除
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters






def deepnet(X, Y,net_layers,learning_rate=0.0075, num_iterations=3000,step=1, print_cost=False, isPlot=True):
    np.random.seed(1) #设计种子
    costs = [] #用于画图
    parameters = init_W(net_layers)
    for i in range(0, num_iterations):
        # 迭代
        AL, caches = L_forword(X, parameters)

        costi = cost(AL, Y) #这里的Y是标准化的Y

        grads = L_model_backward(AL, Y, caches,"softmax")

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % step == 0:
            # 记录成本
            costs.append(costi)
            # 是否打印成本值
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(costi))


    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # plt.savefig(fnme = "cast"+str(datetime.datetime.now())+".jig")
    return parameters


def predict(X, y, parameters,Y_org):

    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = L_forword(X, parameters)
    p = np.argmax(probas,axis=0)
    zql = float(np.sum((p == Y_org)) / m)
    print("准确度为: " + str(float(np.sum((p == Y_org)) / m)))

    error_list = []
    for i in range(m):
        if p[i] != Y_org[i]:
            error_list.append(i)
    return p,error_list,zql


def save_model(parameters):
    np.set_printoptions(threshold=sys.maxsize)

    model_number = 0
    f = open("model/model" + str(model_number) + ".txt", "a+")
    f.write(str(datetime.datetime.now()) + "\n")
    f.write("model_number " + str(model_number) + "\n")
    for i, j in parameters.items():
        f.write(str(i) + "\n")
        f.write(str(j) + "\n")
    f.close()
    return 0

#保存为h5数据格式
def save_h5(data,layers,zql):
    str1 = "./model/model"+str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+".h5"
    f = h5py.File(str1, "w")
    ID = ["model layer "]
    f.create_dataset("layers",data = layers)
    i = len(data) // 2
    for j in range(i):
        f.create_dataset("W"+str(j+1),data = data["W"+str(j+1)])
        f.create_dataset("b"+str(j+1),data = data["b"+str(j+1)])
    f.create_dataset("accuracy",data = zql)
    f.close()





def predict1(X, parameters):
    # 根据参数前向传播
    probas, caches = L_forword(X, parameters)
    p = np.argmax(probas,axis=0)
    return p

def read_ccs(path):
    w = h5py.File(path, "r")
    layers = w["layers"][:]
    l = len(layers)
    p = {}
    # print(l)
    for i in range(1, l):
        p["W" + str(i)] = w["W" + str(i)][:]
        p["b" + str(i)] = w["b" + str(i)][:]
    return p, layers