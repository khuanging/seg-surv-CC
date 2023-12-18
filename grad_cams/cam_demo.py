import os
import cv2
import SimpleITK as sitk
import argparse
import torchvision
import scipy.ndimage as ndimage
from torchsummary import summary
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import monai.transforms as transforms
from models.model_cotr.ResTranUnet import ResTranUnet
from grad_cams.util import show_cam_on_image
from grad_cams.Grad_cam import GradCAM

def read_Nifit(dicts):
    image = sitk.ReadImage(dicts['image'])
    mask = sitk.ReadImage(dicts['mask'])
    np_image = sitk.GetArrayFromImage(image)
    np_mask = sitk.GetArrayFromImage(mask)
    np_image = np_image.astype('float32')
    # np_image = np.expand_dims(np_image, axis=0)
    np_mask = np_mask.astype('uint8')

    dicts = {"image": np_image, "mask": np_mask}
    return dicts

def main():
    grad_model = torch.nn.DataParallel(net).cuda()
    grad_model.load_state_dict(torch.load(opt.model_path))
    grad_model.eval()
    target_layer1 = [net.backbone.layer1]

    data_transform = transforms.Compose([
        transforms.ToTensord(keys=["image", "mask"]),
        transforms.AddChanneld(keys=["image", "mask"]),
        transforms.NormalizeIntensityd(keys=["image"])
    ])
    # load image
    lavac_path = os.path.join(opt.data_path, "T2.nii")
    mask_path = os.path.join(opt.data_path, "T2_mask.nii")
    data = {"image": lavac_path, "mask": mask_path}
    data_dict = read_Nifit(data)
    img = data_dict["image"]

    # data preprocess
    data_dict = data_transform(data_dict)
    img_array, mask_array = data_dict["image"], data_dict["mask"]
    img_tensor = torch.tensor(img_array).unsqueeze(0)
    mask_tensor = torch.tensor(mask_array).unsqueeze(0)
    print(f'0.模型的输入大小：{img_tensor.shape}')  # ([1, 1, 32, 128, 128])
    mr_tensor = img_tensor.cuda()
    cam1 = GradCAM(model=grad_model, target_layers=target_layer1, use_cuda=True)

    grayscale_cam1 = cam1(input_tensor=mr_tensor, target=mask_array)
    grayscale_cam = [grayscale_cam1]

    show_cam_on_image(opt, grayscale_cam, img)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument('--basemodel', type=str, default='')
    parser.add_argument('--cli_path', type=str,
                        default='')
    parser.add_argument('--modal', type=str,
                        default='')
    parser.add_argument('--model_path', type=str,
                        default=f'model.pth')
    parser.add_argument('--data_path', type=str,
                        default=f'')
    opt = parser.parse_args()
    net = ResTranUnet(img_size=(32, 128, 128), num_classes=2)

    main()
