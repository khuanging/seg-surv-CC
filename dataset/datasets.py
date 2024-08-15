import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import monai.transforms as transforms
from abc import ABC
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Tuple, Union


def get_base_transform():
    base_transform = [
        transforms.ToTensord(keys=["image", "mask"]),
        transforms.AddChanneld(keys=["image", "mask"]),
        transforms.NormalizeIntensityd(keys=["image"])
    ]
    return base_transform


def get_train_transform():
    train_transform = [
        transforms.RandBiasFieldd(keys="image", prob=0.3, coeff_range=(0.2, 0.3)),
        # transforms.GibbsNoised(keys="image", alpha=0.3),
        # transforms.RandAdjustContrastd(keys="image", prob=0.3, gamma=(1.5, 2)),
        # transforms.RandAxisFlipd(keys=["image", "mask"], prob=0.3),
        transforms.RandAffined(keys=["image", "mask"], prob=0.3),
        transforms.RandZoomd(keys=["image", "mask"], prob=0.3, mode=["area", "nearest"]),
    ]
    return train_transform


class MRDataset(Dataset, ABC):
    def __init__(self, data_path: str, cli_path: str, modal: str, id: list, train: bool):
        super(MRDataset, self).__init__()

        self.data_path = data_path
        self.modal = modal
        self.all_cli_data = pd.read_excel(cli_path)
        self.cli_data = self.all_cli_data.loc[self.all_cli_data['ID'].isin(id)]
        self.base_transform = get_base_transform()
        self.train_transform = transforms.Compose(get_base_transform() + get_train_transform())

        self.val_transform = transforms.Compose(get_base_transform())
        self.train = train
        self.id = self.cli_data['ID'].values.tolist()

    def __getitem__(self, index: int):
        id_path = os.path.join(self.data_path, str(self.id[index]))
        if self.modal == "t1":
            id_file, mask_file = "t1.nii", "t1_mask.nii"
        else:
            id_file, mask_file = "t2.nii", "t2_mask.nii"
        image_path = os.path.join(id_path, id_file)
        mask_path = os.path.join(id_path, mask_file)
        data_path = {"image": image_path, "mask": mask_path}
        data_dict = self.__read_Nifit__(data_path)

        if self.train:
            data_dict = self.train_transform(data_dict)
        else:
            data_dict = self.val_transform(data_dict)

        return data_dict["image"], data_dict["mask"]

    def __len__(self):
        return len(self.id)

    @staticmethod
    def __read_Nifit__(dicts):
        image = sitk.ReadImage(dicts['image'])
        mask = sitk.ReadImage(dicts['mask'])
        np_image = sitk.GetArrayFromImage(image)
        np_mask = sitk.GetArrayFromImage(mask)
        np_image = np_image.astype('float32')
        np_mask = np_mask.astype('uint8')

        dicts = {"image": np_image, "mask": np_mask}
        return dicts

    TensorOrArray = Union[torch.Tensor, np.ndarray]


if __name__ == '__main__':
    pass
