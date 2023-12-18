from typing import List, Callable

import torch
import cv2 as cv
import numpy as np


class GradCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = model.cuda()
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers)

    @staticmethod
    def get_loss(output, target):
        loss = output  # 直接将预测值作为Loss回传，本文展示的是语义分割的结果
        return loss

    @staticmethod
    def get_cam_weights(grads):
        # GAP全局平均池化，得到大小为[B,C,1,1]
        # 因为我们输入一张图，所以B=1，C为特征层的通道数
        return np.mean(grads, axis=(3, 4), keepdims=True)

    @staticmethod
    def get_target_width_height(input_tensor):
        # 获取原图的高和宽
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def get_cam_image(self, activations, grads):
        # 将梯度图进行全局平均池化，weights大小为[1, C, 1, 1]，在通道上具有不同权重分布
        weights = self.get_cam_weights(grads)  # 对梯度图进行全局平均池化
        print(f'weight.shape:{weights.shape}')
        num = int(weights.shape[2]/2)
        weighted_activations = weights[:, :, num, :, :] * activations[:, :, num, :, :]  # 和原特征层加权乘
        cam = weighted_activations.sum(axis=1)  # 在C维度上求和，得到大小为(1,h,w)
        return cam

    @staticmethod
    def scale_cam_img(cam, target_size=None):
        # 将cam缩放到与原始图像相同的大小，并将其值缩放到[0,1]之间
        result = []
        for img in cam:  # 因为传入的目标层（target_layers）可能为复数，所以一层一层看
            img = img - np.min(img)  # 减去最小值
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv.resize(img, target_size)
                # 注意：cv2.resize(src, (width, height))，width在height前
            result.append(img)
        result = np.float32(result)
        return result

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy() for a in
                            self.activations_and_grads.activations]
        grads_list = [a.cpu().data.numpy() for a in
                      self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # 一张一张特征图和梯度对应着处理
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # ReLU
            scaled = self.scale_cam_img(cam, target_size)
            # 将CAM图缩放到原图大小，然后与原图叠加，这考虑到特征图可能小于或大于原图情况
            cam_per_target_layer.append(scaled[:, None, :])
            # 在None标注的位置加入一个维度，相当于scaled.unsqueeze(1)，此时scaled大小为
            # [1,1,H,W]
        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_layer):
        cam_per_layer = np.concatenate(cam_per_layer, axis=1)
        # 在Channel维度进行堆叠，并没有做相加的处理
        cam_per_layer = np.maximum(cam_per_layer, 0)
        # 当cam_per_layer任意位置值小于0，则置为0
        result = np.mean(cam_per_layer, axis=1)
        # 在channels维度求平均，压缩这个维度，该维度返回为1
        # 也就是说如果最开始输入的是多层网络结构时，经过该方法会将这些网络结构
        # 在Channels维度上压缩，使之最后成为一张图
        return self.scale_cam_img(result)

    def __call__(self, input_tensor, target):  # __init__()后自动调用__call__()方法
        # 这里的target就是目标的gt（双边缘）
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        # 正向传播的输出结果，创建ActivationsAndGradients类后调用__call__()方法，执行self.model(x)
        # 注意这里的output未经softmax，所以如果网络结构中最后的ouput不能经历激活函数
        output = self.activations_and_grads(input_tensor)[0]
        _output = output.detach().cpu()
        _output = _output.squeeze(0).squeeze(0)

        self.model.zero_grad()

        loss = self.get_loss(output, target)
        loss.backward(torch.ones_like(output), retain_graph=True)
        # 将输出结果作为Loss回传，记录回传的梯度图，
        # 梯度最大的说明在该层特征在预测过程中起到的作用最大，
        # 预测的部分展示出来就是整个网络预测时的注意力

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        # 计算每一层指定的网络结构中的cam图
        return self.aggregate_multi_layers(cam_per_layer)
        # 将指定的层结构中所有得到的cam图堆叠并压缩为一张图

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class ActivationsAndGradients:
    # 自动调用__call__()函数，获取正向传播的特征层A和反向传播的梯度A'
    def __init__(self, model, target_layers):

        # 传入模型参数，申明特征层的存储空间（self.activations）
        # 和回传梯度的存储空间（self.gradients）
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        self.reshape_transform = None

        # 注意，上文指明目标网络层是是用列表存储的（target_layers = [model.down4]）
        # 源码设计的可以得到多层cam图
        # 这里注册了一个前向传播的钩子函数“register_forward_hook()”，其作用是
        # 在不改变网络结构的情况下获取某一层的输出，也就是获取正向传播的特征层
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation
                )
            )

        # hasattr(object,name)返回值:如果对象有该属性返回True,否则返回False
        # 其作用是判断当前环境中是否存在该函数（解决版本不匹配的问题）
        if hasattr(target_layer, 'register_full_backward_hook'):
            self.handles.append(
                target_layer.register_full_backward_hook(self.save_gradient))
        else:
            # 注册反向传播的钩子函数“register_backward_hook”，用于存储反向传播过程中梯度图
            self.handles.append(
                target_layer.register_backward_hook(self.save_gradient))

    # 官方API文档对于register_forward_hook()函数有着类似的用法，
    # self.activations中存储了正向传播过程中的特征层
    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    # 与上述类似，只不过save_gradient()存储梯度信息，值得注意的是self.gradients的存储顺序
    def save_gradient(self, model, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients
        # 反向传播的梯度A’放在最前，目的是与特征层顺序一致

    def __call__(self, x):
        # 自动调用，会self.model(x)开始正向传播，注意此时并没有反向传播的操作
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
            # handle要及时移除掉，不然会占用过多内存
