from __future__ import print_function
import os
import math
import data_loader
import tensorflow as tf
import tensorflow.keras.applications
import mmd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# Training settings
iteration = 10000
lr = 0.00000003
LEARNING_RATE = 0.01

root_path = "dataset"
src_name = "src"
tgt_name = "dslr"
src_path = 'dataset/raw_pkl/30001_40000.pkl'
tgt_path = 'dataset/raw_pkl/40001_50000.pkl'
checkpoint_path = 'dataset/checkpoint'  # 模型位置
tf.device('/gpu:1')

# 加载数据，赋值给三个变量：源域、目标域、测试，使用方法在data_loader文件内
f_src = data_loader.load_training(src_path)
f_tgt = data_loader.load_training(tgt_path)

# 对四组数据进行变换（src）
src_x_train, src_y_train, src_x_test, src_y_test = data_loader.transform(f_src)
tgt_x_train, tgt_y_train, tgt_x_test, tgt_y_test = data_loader.transform(f_tgt)

sess = tf.Session()
saver = tf.train.import_meta_graph('dataset/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('dataset'))
graph = tf.compat.v1.get_default_graph()

### restore
temp_output = graph.get_tensor_by_name('l2/Relu:0')
tf_x = graph.get_tensor_by_name('input_placeholder_x:0')
tf_y = graph.get_tensor_by_name('ouput_placeholder_y:0')


def train():
    # 将源域与目标域数据放入迭代器
    # src_iter = iter(src_loader)
    # tgt_iter = iter(tgt_train_loader)

    # 【思路】 src_train通过如下模型训练处src_pred 与 真实的 src_label（这个X值下的groundtruth，即要训练模型拿src_train做X，
    # src_label（视觉方法测出）做Y的）拿经过迁移后得到模型根据X输入推出的src_pred（比较接近于目标域的情况）与真实的值的loss作为一个loss。

    # 源域和目标域X距离尽可能小 ---- loss1
    # 源域经过迁移过的模型出的pred要和真实的label距离尽可能小 ---- loss2
    # tgt的域得出的pred是更符合真实label情况的，所以要往tgt方向迁移但是不需要tgt的pred

    # 输入一个x返回output
    # tf_x = tf.placeholder(tf.float32, [None, 6], name="input_placeholder_x")  # input x
    # tf_y = tf.placeholder(tf.float32, [None, 1], name="ouput_placeholder_y")  # input y

    # neural network layers

    # 直接从temp_output得到其他层输出的tensor
    # l1 = tf.layers.dense(tf_x, 512, tf.nn.relu, name="l1")  # hidden layer
    # l2 = tf.layers.dense(l1, 512, tf.nn.relu, name="l2")  # hidden layer
    l3 = tf.layers.dense(temp_output, 512, tf.nn.relu, name="l3_new")  # hidden layer
    l4 = tf.layers.dense(l3, 512, tf.nn.relu, name="l4_new")  # hidden layer
    l5 = tf.layers.dense(l4, 512, tf.nn.relu, name="l5_new")  # hidden layer
    l6 = tf.layers.dense(l5, 512, tf.nn.relu, name="l6_new")  # hidden layer
    l7 = tf.layers.dense(l6, 512, tf.nn.relu, name="l7_new")  # hidden layer
    l8 = tf.layers.dense(l7, 1, name="l8_new")
    # output = tf.layers.dense(l7, 1, name="outputtttttttt_node")                     # output layer
    Uui = tf.add(l8, l8)
    output = tf.subtract(Uui, l8, name="output_node")
    src_loss = tf.losses.mean_squared_error(tf_y, output)  # compute cost

    Saver = tf.train.Saver()  # control training and others
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())  # initialize var in graph
    # tf.reset_default_graph()

    # 将在外部run运行出的经过网络厚的tgt传入，然后和src经过网络厚的值source和target二者做mmd距离
    tf_mmd_loss = tf.placeholder(tf.float32, None, name="mmd_loss_placeholder")
    tf_lambd = tf.placeholder(tf.float32, None, name="lambd_placeholder")

   # loss = src_loss + tf_lambd * tf_mmd_loss
    loss = src_loss + tf_lambd * tf_mmd_loss


    # tf.reset_default_graph()
    # outputNodeName = 'output_node'

    for step in range(5000000):
        # train and net output

        lrt = lr / math.pow((1 + 10 * (step - 1) / (5000000)), 0.5)
        optimizer = tf.train.GradientDescentOptimizer(lrt)
        train_op = optimizer.minimize(loss)

        # 经过Use模型得到source与target

        if step == 0:
            target = sess.run(output,
                                 {tf_x: tgt_x_train, tf_y: tgt_y_train, tf_mmd_loss: 100, tf_lambd: 0})
            _, loss_loss, source = sess.run([train_op, loss, output],
                                            {tf_x: src_x_train, tf_y: src_y_train, tf_mmd_loss: 100, tf_lambd: 0})

        else:
            target = sess.run(output,
                                 {tf_x: tgt_x_train, tf_y: tgt_y_train, tf_mmd_loss: mmd_loss, tf_lambd: lambd})
            _, loss_out, source = sess.run([train_op, loss, output],
                                           {tf_x: src_x_train, tf_y: src_y_train, tf_mmd_loss: mmd_loss,
                                            tf_lambd: lambd})

            # loss


        mmd_loss = mmd.mmd_rbf(source, target)
        # print("mmd_loss:",mmd_loss)
        lambd = 2 / (1 + math.exp(-10 * (step) / 5000000)) - 1
        # print("lambd:",lambd)


        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Input/x", "Input/y", "Output/predictions"])
        if (step > 0):
            # print('loss is: ' + str(loss_out))
            if (step % 5== 0):
                print("step: ", step, "  ", "learning rate: ", lrt)
                pred1 = sess.run(output, {tf_x: tgt_x_test})
                # pred1 =  sess.run(output, feed_dict={tf_x: X_test3})  # (-1, 3)
                print('loss is: ' + str(loss_out))

                # print('prediction is:' + str(pred))
                k = 0
                k1 = 0
                for i in range(len(tgt_y_test)):
                    jjji = pred1[i]
                    yyyi = tgt_y_test[i]
                    if (abs(yyyi[0] - jjji[0]) < 1):
                        # print(Reverse(yyyi[0]),Reverse(jjji[0]))
                        k += 1
                    if (abs(yyyi[0] - jjji[0]) < 5):
                        # print(Reverse(yyyi[0]),Reverse(jjji[0]))
                        k1 += 1
                        # break
                    # if abs(yyyi-jjji) < 1:
                print("pred_loss: ", sess.run(tf.losses.mean_squared_error(pred1, tgt_y_test)))
                print("acc:", k1 / len(tgt_y_test))
                print("------------------------------")
            if (step % 2000 == 0):
                lujing = '/home/zzg/Documents/model.ckpt'
                Saver.save(sess, lujing)
                print("successSave!")


if __name__ == '__main__':
    # 使用train方法训练模型，传入值为外部模型内容
    train()
