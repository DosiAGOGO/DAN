
import datetime as dt
import pickle
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf


dump_path = 'CacheFCN1114.pkl'
XTrain_path = '/home/wdenxw/Train2/TFNN/1109/RegOnly/' + 'XTraincache.pkl'
YTrain_path = '/home/wdenxw/Train2/TFNN/1109/RegOnly' + '/YTraincache.pkl'
XTest_path = '/home/wdenxw/Train2/TFNN/1109/RegOnly/XTestcache.pkl'
YTest_path = '/home/wdenxw/Train2/TFNN/1109/RegOnly/YTestcache.pkl'
# pickle.dump(df,open(dump_path,'wb'))
if os.path.exists(dump_path):
    try:
        dg = pickle.load(open(dump_path, 'rb'))
    except EOFError:
        print("EOF")

X_train, X_test, Y_train, Y_test = train_test_split(dg.iloc[:, 1:7], dg.iloc[:, 0], train_size=0.8)


X_train2 = X_train.as_matrix()
X_train3 = X_train2.tolist()

Y_train2 = Y_train.as_matrix()
rrrYTR = Y_train2.shape[0]
Y_train3 = Y_train2.tolist()

X_test2 = X_test.as_matrix()
X_test3 = X_test2.tolist()

Y_test2 = Y_test.as_matrix().transpose()
rrrYTE = Y_test2.shape[0]
Y_test3 = Y_test2.reshape(rrrYTE, 1)
Y_test4 = Y_test3.tolist()

Y_train3 = Y_train2.reshape(rrrYTR, 1)
Y_train4 = Y_train3.tolist()

pickle.dump(X_train, open(XTrain_path, 'wb'))
pickle.dump(Y_train, open(YTrain_path, 'wb'))
pickle.dump(X_test, open(XTest_path, 'wb'))
pickle.dump(Y_test, open(YTest_path, 'wb'))



# x_pred = [[120,5,85,120,5,85]]


# ����һ��x����output
tf_x = tf.placeholder(tf.float32, [None, 6], name="input_placeholder_x")  # input x
tf_y = tf.placeholder(tf.float32, [None, 1], name="ouput_placeholder_y")  # input y
# neural network layers
l1 = tf.layers.dense(tf_x, 512, tf.nn.relu, name="input_node")  # hidden layer
l2 = tf.layers.dense(l1, 512, tf.nn.relu)  # hidden layer
l3 = tf.layers.dense(l2, 512, tf.nn.relu)  # hidden layer
l4 = tf.layers.dense(l3, 512, tf.nn.relu)  # hidden layer
l5 = tf.layers.dense(l4, 512, tf.nn.relu)  # hidden layer
l6 = tf.layers.dense(l5, 512, tf.nn.relu)  # hidden layer
l7 = tf.layers.dense(l6, 512, tf.nn.relu)  # hidden layer
l8 = tf.layers.dense(l7, 1)
# output = tf.layers.dense(l7, 1, name="outputtttttttt_node")                     # output layer
Uui = tf.add(l8, l8)
output = tf.subtract(Uui, l8, name="output_node")


loss = tf.losses.mean_squared_error(tf_y, output)  # compute cost


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_op = optimizer.minimize(loss)
saveryy = tf.train.Saver()  # control training and others
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # initialize var in graph
# tf.reset_default_graph()


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
# train_op = optimizer.minimize(loss)
for step in range(5000000):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: X_train3, tf_y: Y_train4})
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Input/x", "Input/y", "Output/predictions"])
    if (step % 5000 == 0):
        #now_time = dt.datetime.now().strftime('%F %T')
        #print('Now Time is:' + now_time)
        pred1 = sess.run(output, {tf_x: X_test3})
        # pred1 =  sess.run(output, feed_dict={tf_x: X_test3})  # (-1, 3)
        print('loss is: ' + str(l))
        # print('prediction is:' + str(pred))
        k = 0
        k1 = 0
        for i in range(len(Y_test4)):
            jjji = pred1[i]
            yyyi = Y_test4[i]
            if (abs(yyyi[0] - jjji[0]) < 1):
                # print(Reverse(yyyi[0]),Reverse(jjji[0]))
                k += 1
            if (abs(yyyi[0] - jjji[0]) < 5):
                # print(Reverse(yyyi[0]),Reverse(jjji[0]))
                k1 += 1
                # break
            # if abs(yyyi-jjji) < 1:

        print(k / len(Y_test4))
        print(k1 / len(Y_test4))
    if (step % 10000 == 0):
        lujing = '/home/wdenxw/Train2/TFNN/1109/RegOnly/model.ckpt'
        saveryy.save(sess, lujing)
        print("successSave!")