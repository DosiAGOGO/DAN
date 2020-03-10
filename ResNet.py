import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch
import tensorflow as tf

__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1,groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# block组 三层卷积
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
'''


class log(nn.Module):

    def __init__(self):
        super(log, self).__init__()

        # 输入一个x返回output
        tf_x = tf.placeholder(tf.float32, [None, 6], name="input_placeholder_x")  # input x
        tf_y = tf.placeholder(tf.float32, [None, 1], name="ouput_placeholder_y")  # input y
        # neural network layers
        l1 = tf.layers.dense(tf_x, 512, tf.nn.relu, name="fixed/")  # hidden layer
        l2 = tf.layers.dense(l1, 512, tf.nn.relu, name="fixed/")  # hidden layer
        l3 = tf.layers.dense(l2, 512, tf.nn.relu, name="fixed/")  # hidden layer
        l4 = tf.layers.dense(l3, 512, tf.nn.relu, name="modified/")  # hidden layer
        l5 = tf.layers.dense(l4, 512, tf.nn.relu, name="modified/")  # hidden layer
        l6 = tf.layers.dense(l5, 512, tf.nn.relu, name="modified/")  # hidden layer
        l7 = tf.layers.dense(l6, 512, tf.nn.relu, name="modified/")  # hidden layer
        l8 = tf.layers.dense(l7, 1, name="modified/")
        # output = tf.layers.dense(l7, 1, name="outputtttttttt_node")                     # output layer
        Uui = tf.add(l8, l8)
        output = tf.subtract(Uui, l8, name="output_node")
        #loss = tf.losses.mean_squared_error(tf_y, output)  # compute cost


    # 输入一个x返回一个经过几层后的x
    def forward(self, output):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # #x=self.fc(x)

        '''
        X = None  # input
        yhat = None  # output
        outputNodeName = 'output_node'
        sess = tf.Session()

        # 读取模型
        modelpath = 'E:/AboutPaper/RegOnly/'
        saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        ouputNode = str(outputNodeName) + ':0'
        return output
        '''


class DANNet(nn.Module):
    # 计算源域目标域的距离损失函数的DANNet，使用resnet预训练，继承自Module
    def __init__(self, num_classes=31):
        super(DANNet, self).__init__()
        # DANNet share resnet50 -> resnet
        self.sharedNet = resnet50(True)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        loss = 0
        # 经过ResNet预训练
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            #loss += mmd.mmd_rbf_accelerate(source, target)
            # 损失函数 源域目标域的mmd距离
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        # 对源域做了一个线性变换然后返回回去 做src pred（经过网络后的预测值）
        source = self.cls_fc(source)
        #target = self.cls_fc(target)

        return source, loss

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 走的是ResNet模型的所有层 （加上DAN的一层），[3,4,6,3]是层数（每个block）不是层号。此处不是预训练，是拿ResNet作为模型层的前半部分。
    model = log()
    # 此处才是预训练内容，ResNet的模型参数通过网络下载放上。
    if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        tf.reset_default_graph()

    return model