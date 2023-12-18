import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from datetime import datetime
from skimage.transform import resize

def show_cam_on_image(args, cam, mr_array):
    heatmaps, axial_grad_cmaps, axial_overlays = [], [], []
    axial_slice_count = 16
    coronal_slice_count = 64
    sagittal_slice_count = 64
    # 原图
    axial_mr_img = np.squeeze(mr_array[axial_slice_count, :, :])
    for i in range(len(cam)):
        capi = resize(cam[i], (32, 128, 128))
        print(f'capi:{capi.shape}')
        capi = np.maximum(capi, 0)
        heatmap = (capi - capi.min()) / (capi.max() - capi.min())
        # heatmaps.append(heatmap)
        print(heatmap.shape)
        # np.squeeze删除shape中为1的维度
        # axial_mr_img 轴向面
        axial_grad_cmap = np.squeeze(heatmap[axial_slice_count * 1, :, :])
        axial_grad_cmaps.append(axial_grad_cmap)
        # # sagittal_mr_img 矢状面
        # sagittal_mr_img = np.squeeze(mr_array[:, :, sagittal_slice_count])
        # sagittal_grad_cmap_img = np.squeeze(heatmap[:, :, sagittal_slice_count * 1])
        # print(f'sagittal_mr_img.shape:{sagittal_mr_img.shape}, sagittal_grad_cmap_img.shape:{sagittal_grad_cmap_img.shape}')
        # # coronal_mr_img 冠状面
        # coronal_mr_img = np.squeeze(mr_array[:, coronal_slice_count, :])
        # coronal_grad_cmap_img = np.squeeze(heatmap[:, coronal_slice_count * 1, :])

        # 放大十倍以使权重图更流畅
        # axial_mr_img = ndimage.zoom(axial_mr_img, (10, 10), order=3)
        # 将权重图与原始图像叠加
        axial_overlay = cv2.addWeighted(axial_mr_img, 0.1, axial_grad_cmap, 0.8, 0)
        axial_overlays.append(axial_overlay)

        # # sagittal_mr_img = ndimage.zoom(sagittal_mr_img, (10, 10), order=3)
        # sagittal_overlay = cv2.addWeighted(sagittal_mr_img, 0.1, sagittal_grad_cmap_img, 0.8, 0)

        # # coronal_mr_img = ndimage.zoom(coronal_mr_img, (10, 10), order=3)
        # coronal_overlay = cv2.addWeighted(coronal_mr_img, 0.1, coronal_grad_cmap_img, 0.8, 0)

    fig = plt.figure(figsize=(16, 4), dpi=300, facecolor="w")

    # 4*6 的网格；划分成4行6列（0,0）位置开始，跨度为 2 行 2 列
    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((1, 4), (0, 2), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((1, 4), (0, 3), colspan=1, rowspan=1)

    ax1.imshow(axial_mr_img, cmap='gray')
    ax2.imshow(axial_overlays[0], cmap='jet')
    ax3.imshow(axial_overlays[1], cmap='jet')

    pl4 = ax4.imshow(axial_overlays[2], cmap='jet')

    fig.subplots_adjust(bottom=0.03, top=0.99, left=0.03, right=0.95)

    # colorbar 左 下 宽 高
    # 对应 l,b,w,h；设置colorbar位置；
    position = fig.add_axes([0.96, 0.19, 0.015, 0.6])
    fig.colorbar(pl4, ax=ax4, cax=position)
    # 设置全局字体
    plt.rc('font', family='Arial')
    # plt.colorbar(ax3, shrink=0.5)  # color bar if need
    plt.savefig(f'./grad_cams/image_result/{args.basemodel}_{args.modal}_{mr}.png', dpi=300, transparent=False)
    plt.show()