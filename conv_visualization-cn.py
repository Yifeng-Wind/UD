#!/usr/bin/env python
# coding: utf-8

# # 卷积层
# 
# 在此 notebook 中，我们将可视化卷积层的四个过滤输出（即激活图）。 
# 
# 在此示例中，我们将通过初始化卷积层的**权重**，定义应用到输入图像的四个过滤器，但是训练过的 CNN 将学习这些权重的值。
# 
# <img src='notebook_ims/conv_layer.gif' height=60% width=60% />
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


# In[ ]:


# visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')


# ## 定义卷积层 
# 
# [该文](http://pytorch.org/docs/stable/nn.html)介绍了构成任何神经网络的各种层级。对于卷积神经网络，我们首先会定义：
# * 卷积层
# 
# 初始化一个卷积层，使其包含你所创建的所有过滤器。注意，你不需要训练此网络；你只是初始化卷积层的权重，当对此网络进行一次前向传播后，你就可以可视化发生的情况了。
# 
# 
# #### `__init__` 和 `forward`
# 要在 PyTorch 中定义神经网络，你需要在函数 `__init__` 中定义模型层级，并在函数 `forward` 中定义将这些初始化的层级应用到输入 (`x`) 上的网络前向行为。在 PyTorch 中，我们将所有输入转换为张量数据类型，它和 Python 中的列表数据类型相似。 
# 
# 下面我定义了一个叫做 `Net` 的类结构，该结构具有一个包含四个 3x3 灰阶过滤器的卷积层。

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
    
# define a neural network with a single convolutional layer with four filters
class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # defines the convolutional layer, assumes there are 4 grayscale filters
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        
        # returns both layers
        return conv_x, activated_x
    
# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# print out the layer in the network
print(model)


# ### 可视化每个过滤器的输出
# 
# 首先，我们将定义一个辅助函数 `viz_layer`，它接受特定的层级和过滤器数量（可选参数），并在图像穿过之后，显示该层级的输出。

# In[1]:


# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters= 4):
    fig = plt.figure(figsize=(20, 20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))


# 我们看看在应用 ReLu 激活函数之前和之后卷积层的输出。

# In[2]:


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

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

# visualize the output of a conv layer
viz_layer(conv_layer)


# #### ReLU 激活
# 
# 在此模型中，我们使用了缩放卷积层输出的激活函数。为此，我们使用了 ReLU 函数，此函数直接将所有负像素值变成 0（黑色）。对于输入像素值 `x`，请查看下图中的方程。 
# 
# <img src='notebook_ims/relu_ex.png' height=50% width=50% />

# In[ ]:


# after a ReLu is applied
# visualize the output of an activated conv layer
viz_layer(activated_layer)


# ```python
# 
# ```
