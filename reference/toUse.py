# 前提是：现在已经有一个标准数据集了，大概像这样 时间，106.4150858224653,75,22,24,101,393,229的很多行，现在要load相应的model去算准确率
from collections import Counter  # 取三次输入的最大值
import pandas as pd
import math
import tensorflow as tf
import numpy as np
import pickle
import os


sess = None
graph = None

# load_model('D:/a/Caiji/Z/FCNTrainData0823Z1.txt',)
# 定义sess
# 准备数据

dump1_path = '/Users/zzg/PycharmProjects/untitled1/DAN/dataset/raw_pkl/10001_20000.pkl'
print("asfa")
if os.path.exists(dump1_path):
    try:
        print(dump1_path)
        dg = pickle.load(open(dump1_path, 'rb'))
        print("aaaaa")
    except EOFError:
        print("EOF")

    X_Data = dg.iloc[:, 2:8]
    Y_Data = dg.iloc[:, 1]

    X_Data2 = X_Data.as_matrix()
    X_Data3 = X_Data2.tolist()

    Y_Data2 = Y_Data.as_matrix()
    rrrYTR = Y_Data2.shape[0]
    # Y_train3=torch.from_numpy(Y_train2)

    Y_Data3 = Y_Data2.reshape(rrrYTR, 1)
    Y_Data4 = Y_Data3.tolist()

    listx1 = np.array(X_Data3[0]).T
    listx21 = listx1.reshape(1, 6)

    listx2 = np.array(X_Data3[20]).T
    listx22 = listx2.reshape(1, 6)

    listx3 = np.array(X_Data3[40]).T
    listx23 = listx3.reshape(1, 6)

    listx4 = np.array(X_Data3[60]).T
    listx24 = listx3.reshape(1, 6)

    listx5 = np.array(X_Data3[80]).T
    listx25 = listx3.reshape(1, 6)

    listx6 = np.array(X_Data3[100]).T
    listx26 = listx3.reshape(1, 6)

    listuui = []
    print("11111")
    # listuui.append(classify(listx21,0))
    # listuui.append(classify(listx22,1))
    # listuui.append(classify(listx23,2))
    # listuui.append(classify(listx24,3))
    # listuui.append(classify(listx25,4))
    # listuui.append(classify(listx26,5))
    # aaa=Counter(listuui).most_common(4)
    # maxClass=aaa[0][0]
    # print(maxClass+1)

    # realNumber=maxClass+1
    yhat = None
    X = None
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    outputNodeName = 'output_node'

    # xuhao=realNumber
    import tensorflow as tf

    # 可能要新开一个或者重置什么的,不对，好像是回归模型读不出来
    tf.reset_default_graph()
    X = None  # input
    yhat = None  # output

    sess = tf.Session()
    graph = tf.get_default_graph()

    modelpath = '/Users/zzg/PycharmProjects/untitled1/DAN/dataset/model'
    print(modelpath)
    saver = tf.train.import_meta_graph(modelpath + '/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(modelpath))
    X = graph.get_tensor_by_name('input_placeholder_x:0')
    Y = graph.get_tensor_by_name('ouput_placeholder_y:0')
    ouputNode = str(outputNodeName) + ':0'
    yhat = graph.get_tensor_by_name(ouputNode)
    aac = sess.run(yhat, feed_dict={X: X_Data3})

    # 验证准确率

    k = 0
    for i in range(len(Y_Data4)):
        print(i)
        jjji = aac[i]
        yyyi = Y_Data4[i]
        if (abs(yyyi[0] - jjji[0]) < 300):
            k += 1
            # break
        # if abs(yyyi-jjji) < 1:
        a = k / len(Y_Data4)
        acccu = ("%.4f" % a)

    print("                                       circle:", "accracy:", acccu)