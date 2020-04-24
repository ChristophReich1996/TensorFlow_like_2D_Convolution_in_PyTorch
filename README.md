# TensorFlow like 2D Convolution in PyTorch

When using a 2D convolution in [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D), only the number of filters has to be specified. In [PyTorch](https://pytorch.org/docs/stable/nn.html#convolution-layers), however, the number of input channels and the number of output channels has to be defined. To overcome this translation problem, I implemented a Pytorch module which determines the number of input channels when the forward method is called the for the first time.
