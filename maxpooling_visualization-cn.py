#!/usr/bin/env python
# coding: utf-8

# ﻿# 最大池化层
# 
# 在此 notebook 中，我们将在 CNN 里添加最大池化层并可视化它的输出结果。 
# 
# 该 CNN 由一个卷积层+激活函数，然后是一个池化层和线性层（以形成期望的输出大小）组成。
# 
# <img src='notebook_ims/CNN_all_layers.png' height=50% width=50% />
# 
# ### 导入图像

# In[ ]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# TODO: Feel free to try out your own images here by changing img_path
# to a file path to another image on your computer!
img_path = 'data/udacity_sdc.png'

# load color image 
bgr_img = cv2.imread(img_path)
# convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255

# plot image
plt.imshow(gray_img, cmap='gray')
plt.show()


# ### 定义和可视化过滤器

# In[ ]:


import numpy as np

## TODO: Feel free to modify the numbers here, to try out another filter!
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

print('Filter shape: ', filter_vals.shape)


# In[ ]:


# Defining four different filters, 
# all of which are linear combinations of the `filter_vals` defined above

# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# For an example, print out the values of filter 1
print('Filter 1: \n', filter_1)


# ### 定义卷积层和池化层
# 
# 你已经知道如何定义卷积层，接下来是：
# * 池化层
# 
# 在下个单元格中，我们将初始化卷积层，使其包含所有已创建的过滤器。然后添加最大池化层（请参阅[此文档](http://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html），卷积核大小是 2x2，所以在这一步之后，图像分辨率变小了！
# 
# 最大池化层会减小输入的 x-y 大小，并仅保留最活跃的像素值。下面是一个示例 2x2 池化核，步长为 2，应用到了一小批灰阶像素值，使这批数据的大小缩小到 1/4。只有 2x2 中的最大像素值保留在新的池化输出中。
# 
# <img src='notebook_ims/maxpooling_ex.png' height=50% width=50% />

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
    
# define a neural network with a convolutional layer with four filters
# AND a pooling layer of size (2, 2)
class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # defines the convolutional layer, assumes there are 4 grayscale filters
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        # define a pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        
        # applies pooling layer
        pooled_x = self.pool(activated_x)
        
        # returns all layers
        return conv_x, activated_x, pooled_x
    
# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# print out the layer in the network
print(model)


# ### 可视化每个过滤器的输出
# 
# 首先，我们将定义一个辅助函数 `viz_layer`，它接受特定的层级和过滤器数量（可选参数），并在图像穿过之后，显示该层级的输出。

# In[ ]:


# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters= 4):
    fig = plt.figure(figsize=(20, 20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1)
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))


# 我们看看在应用 ReLu 激活函数之前和之后卷积层的输出。
# 
# #### ReLU 激活
# 
# ReLU 函数将所有负像素值变成 0（黑色）。对于输入像素值 `x`，请查看下图中的方程。 
# 
# <img src='notebook_ims/relu_ex.png' height=50% width=50% />

# In[ ]:


# plot original image
plt.imshow(gray_img, cmap='gray')

# visualize all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))

    
# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get all the layers 
conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)

# visualize the output of the activated conv layer
viz_layer(activated_layer)


# ### 可视化池化层的输出
# 
# 然后，查看池化层的输出。池化层将上图中的特征图当做输入，并以特定的池化系数降低这些特征图的维度，仅用给定核区域的最大值（最亮像素）构建新的更小图像。
# 
# 请查看 x,y 轴上的值，看看图像的大小变化了多少。
# 
# 
# 
# 
# ```python
# # visualize the output of the pooling layer
# viz_layer(pooled_layer)
# ```
