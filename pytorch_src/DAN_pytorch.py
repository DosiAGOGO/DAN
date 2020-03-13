"""
remove the changing learning-rate, add the acc for judge.
-----------------------------------------
time: 2020-3-7
data:10w_all
fine-tune:false
DA:false

"""

from __future__ import print_function
import torch.nn.functional as F
import mmd
import torch
import os
import math
import pytorch_src.data_loader_pytorch as data_loader
import pytorch_src.DANNet_pytorch as models


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("version 1")
PATH = '/media/zzg/29b7a8df-b813-4851-a749-f9db6d488d0d/zzg/Documents/DAN_raw_0-1w.pth'
# Training settings
iteration = 500000000
lr = 0.00001
l2_decay = 5e-4
momentum = 0.9
no_cuda = False
seed = 8

src_path = '../dataset/raw_pkl/0_10000.pkl'
tgt_path = '../dataset/raw_pkl/40001_50000.pkl'
checkpoint_path = 'dataset/checkpoint'  # 模型位置

'''
预留fine-tune功能，先不实现，先实现模型部分，不fine-tune节约算力。
实现模型部分验证可行性和模型好坏，以及生成一个可以fine-tune的模型。

###########################################################
'''

# GPU
cuda = not no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def train(model):
    """
    # 将源域与目标域数据放入迭代器
    # src_iter = iter(src_loader)
    # tgt_iter = iter(tgt_train_loader)

    # 【思路】 src_train通过如下模型训练处src_pred 与 真实的 src_label（这个X值下的groundtruth，即要训练模型拿src_train做X，
    # src_label（视觉方法测出）做Y的）拿经过迁移后得到模型根据X输入推出的src_pred（比较接近于目标域的情况）与真实的值的loss作为一个loss。

    # 源域和目标域X距离尽可能小 ---- loss1
    # 源域经过迁移过的模型出的pred要和真实的label距离尽可能小 ---- loss2
    # tgt的域得出的pred是更符合真实label情况的，所以要往tgt方向迁移但是不需要tgt的pred

    # neural network layers

    # 直接从temp_output得到其他层输出的tensor
    """
    # 加载数据，赋值给三个变量：源域、目标域、测试，使用方法在data_loader文件内
    f_src = data_loader.load_training(src_path)
    f_tgt = data_loader.load_training(tgt_path)

    # 对四组数据进行变换（src）
    src_data, src_label, src_, src__ = data_loader.transform(f_src)
    tgt_data, tgt_label, tgt_, tgt__ = data_loader.transform(f_tgt)

    correct = 0
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / iteration), 0.75)

        # if (i - 1) % 100 == 0:
        #    print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE / 10)

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data = tgt_label.cuda()

        optimizer.zero_grad()
        # src_pred, mmd_loss = model(src_data, tgt_data)
        src_pred = model(src_data)



        # cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)
        # lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1
        # loss = cls_loss + lambd * mmd_loss
        src_label = src_label.float()

        loss = F.mse_loss(src_pred[:, 0], src_label)
        if i % 1000 == 0:
            print(i, "lr: ", LEARNING_RATE)
            print("  - loss is: ", loss.data)
        loss.backward()
        optimizer.step()


        if i % 5000 == 0:

            print("Successfully saved")
            src_ = src_.cuda()
            Y_test = model(src_)
            Y_test = Y_test[:, 0]

            k = 0

            for i in range(len(src__)):
                ji = src__[i]
                yi = Y_test[i]
                p = abs(yi - ji)
                k = k + p
            k = k/len(src__)

            print("acc:", k)
            print("---------------------------")

            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'acc': k,
            }, PATH)

if __name__ == '__main__':

    model = models.DANNet()
    print(model)
    if cuda:
        model.cuda()
    # 使用train方法训练模型，传入值为外部模型内容
    train(model)

"""
RESULT:
--------------------
epoch:
loss:270
time:2020-3-12-10:15 - 3-13-10:35
acc:0.9

"""



