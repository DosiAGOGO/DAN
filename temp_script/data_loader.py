from torchvision import datasets, transforms
import torch
import pandas as pd
import math
import pickle
import os
from sklearn.model_selection import train_test_split

'''
原版加载图片的加载方法

def load_training(root_path, dir, batch_size, kwargs):
    # 定义一个transform的变换形式
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    # 定义路径
    data = datasets.ImageFolder(root=root_path + "/" + dir, transform=transform)
    # 读取路径内容并返回
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader
'''

'''
新版加载txt文档的加载方法
'''




def transform (f):
    # 划分测试集和训练集 按8：2比例划分
    X_train, X_test, Y_train, Y_test = train_test_split(f.iloc[:, 1:7], f.iloc[:, 0], train_size=0.8)

    X_train = X_train.as_matrix()
    X_train = X_train.tolist()

    X_test = X_test.as_matrix()
    X_test = X_test.tolist()

    Y_train = Y_train.as_matrix()
    rrrYTR = Y_train.shape[0]

    Y_train = Y_train.reshape(rrrYTR, 1)
    Y_train = Y_train.tolist()


    Y_test = Y_test.as_matrix().transpose()
    rrrYTE = Y_test.shape[0]
    Y_test = Y_test.reshape(rrrYTE, 1)
    Y_test = Y_test.tolist()

    return X_train, Y_train, X_test, Y_test


def load_training(path):
    # 定义一个transform的变换形式
    # pickle.dump(df,open(dump_path,'wb'))
    print(path)
    if os.path.exists(path):
        try:
            train_loader = pickle.load(open(path, 'rb'))
            # print(train_loader)
        except EOFError:
            print("EOF")
    return train_loader



def makePkl(txtName):
    # 读取txt文件并生成pkl
    rootPath = 'dataset'
    filenameDataSet = rootPath + '/raw/' + txtName + '.txt'
    print("it is", filenameDataSet)
    df = pd.read_csv(filenameDataSet, header=None)

    for iii in range(df.shape[0]):
        for jjj in range(2, df.shape[1]):
            tty = df.loc[iii, jjj]
            df.iloc[iii, jjj] = math.log(tty, 1024)

    dump1_path = rootPath + '/raw_pkl/' + txtName + '.pkl'
    print(dump1_path)
    pickle.dump(df, open(dump1_path, 'wb'))


'''
def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + "/" + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
'''