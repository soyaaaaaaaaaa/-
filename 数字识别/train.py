
import DeepNET
import time

import function



if __name__ == '__main__':
    # 计时开始
    start1 = time.time()
    train_image_path = './MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte'
    train_lable_path = './MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte'
    teat_image_path = './MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
    teat_lable_path = './MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
    # #加载数据
    train_image = function.jiexi_image(train_image_path)
    train_lable = function.jiexi_label(train_lable_path)
    teat_image = function.jiexi_image(teat_image_path)
    test_lable = function.jiexi_label(teat_lable_path)
    # print(train_image.shape)

    function.plot_data(train_image,train_lable,10,True)

    train_image = train_image.reshape(train_image.shape[0], -1).T / 255
    teat_image = teat_image.reshape(teat_image.shape[0], -1).T / 255
    train_lable1 = function.label_init(train_lable)
    test_lable1 =  function.label_init(test_lable)

    print(train_image.shape)

    end1 = time.time()

    start2 = time.time()
    layers = [784, 30, 15, 10]
    parameters = DeepNET.deepnet(train_image, train_lable1,layers , learning_rate=0.075, num_iterations=1000,
                                 step=100, print_cost=True, isPlot=True)
    end2 = time.time()

    p, error_list_train, zql1 = DeepNET.predict(train_image, train_lable1, parameters, train_lable)
    p0, error_list_test, zql2 = DeepNET.predict(teat_image,test_lable1,parameters,test_lable)
    zql = [[zql1], [zql2]]
    print("数据加载时间：", end1-start1," 秒")
    print("模型训练时间：", end2-start2," 秒")

    DeepNET.save_h5(parameters, layers, zql)