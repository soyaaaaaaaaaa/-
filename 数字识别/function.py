#此文件是一些函数 有加载数据模块
import datetime
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def jiexi_image(path):
    # 用二进制读取
    data = open(path, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, data, offset)
    print("魔数:%d, 图片数量: %d张, 图片大小: %d*%d" % (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def jiexi_label(path):
    data = open(path, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def plot_data(images,labels,n,issave = False):
    for i in range(n):
        print(labels[i])
        plt.imshow(images[i], cmap='gray')
        plt.show()
        # if(issave == True):
            # plt.savefig(fname = "save"+str(datetime.datetime.now())+".jpg")

    print('done')





# 说明：输入原始图像路径和新建图像文件夹名称 默认修改出长度宽度为64*64
def stdimage(pathorg, name, pathnew=None, width=64, length=64):
    # 检查文件是否建立
    if pathnew == None:  # 如果没有手动创建
        tage = os.path.exists(os.getcwd() + '\\' + name)  # 检查一下是否属实
        if not tage:  # 没有整个新文件夹
            os.mkdir(os.getcwd() + "\\" + name)  # 创建文件夹，name
        image_path = os.getcwd() + "\\" + name + "\\"
    else:  # 已经手动创建
        tage = os.path.exists(pathnew + "\\" + name)
        if not tage:
            path = os.getcwd()
            os.mkdir(path + "\\" + name)
        image_path = path + "\\" + name + "\\"

    # 开始处理
    i = 1  # 从一开始
    list_name = os.listdir(pathorg)  # 获取图片名称列表
    for item in list_name:
        # 检查是否有图片
        tage = os.path.exists(pathorg + str(i) + '.png')
        if not tage:
            image = Image.open(pathorg + '\\' + item)
            std = image.resize((width, length), Image.ANTIALIAS)
            # 模式为RGB
            if not std.mode == "RGB":
                std = std.convert('RGB')
            std.save(image_path + str(i) + '.png')
        i += 1




def label_init(lable):
    n = lable.shape[0]
    label_Y = np.zeros([10, n])
    res = lable.astype(int)
    for i in range(0, label_Y.shape[1]):
        label_Y[res[i], i] = 1
    return label_Y




def get_X(path):
    im_name_list = os.listdir(path)
    all_data = []
    for item in im_name_list:
        try:
            all_data.append(plt.imread(path + '\\' + item).tolist())
        except:
            print(item + " open error ")
    return all_data

