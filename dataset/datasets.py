import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import monai.transforms as transforms
from abc import ABC
from math import ceil, sqrt
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")


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
    def __init__(self, data_path: str, cli_path: str, modal: str, mr: list, train: bool, use_cli: bool, num_time_bins: int=15):
        super(MRDataset, self).__init__()

        self.data_path = data_path
        # self.mode = mode
        self.num_bins = num_time_bins
        self.modal = modal
        self.all_cli_data = pd.read_excel(cli_path)
        self.cli_data = self.all_cli_data.loc[self.all_cli_data['MR'].isin(mr)]
        self.base_transform = get_base_transform()
        self.train_transform = transforms.Compose(get_base_transform() + get_train_transform())

        self.val_transform = transforms.Compose(get_base_transform())
        self.train = train
        self.use_cli = use_cli
        # if train :
        #     self.cli_data = shuffle(self.cli_data, random_state=42)
        if use_cli is not None:
            self.times, self.events, self.clinical_data, self.cli = self.make_cli_data()
        self.mr = self.cli_data['MR'].values.tolist()

    # encoding clinical data
    def make_cli_data(self):
        clinical_data = self.cli_data
        times = np.array(clinical_data['time'].values.tolist(), dtype=np.float32)
        events = np.array(clinical_data['event'].values.tolist(), dtype=np.int32)
        try:  # training data
            clin_var_data = clinical_data.drop(['MR', 'time', 'event'], axis=1)
        except:  # test data
            clin_var_data = clinical_data.drop(['MR'], axis=1)
        # clin_var_data to one-hot encoder
        clin_var_data['age'] = scale(clin_var_data.loc[:, 'age'])
        cli_encode = pd.get_dummies(clin_var_data.loc[:, clin_var_data.columns != 'age'], drop_first=True)
        clinical_data = pd.concat([clinical_data['time'], clinical_data['event'], clin_var_data['age'], cli_encode],
                                  axis=1)
        cli = pd.concat([clin_var_data['age'], cli_encode], axis=1).values

        return times, events, clinical_data, cli

    def __getitem__(self, index: int):
        mr_path = os.path.join(self.data_path, str(self.mr[index]))
        if self.modal == "lavac":
            mr_file, mask_file = "T1+C.nii", "T1+C_mask.nii"
        else:
            mr_file, mask_file = "T2.nii", "T2_mask.nii"
        lavac_path = os.path.join(mr_path, mr_file)
        mask_path = os.path.join(mr_path, mask_file)
        data_path = {"image": lavac_path, "mask": mask_path}
        data_dict = self.__read_Nifit__(data_path)

        if self.train:
            data_dict = self.train_transform(data_dict)
        else:
            data_dict = self.val_transform(data_dict)
        # read clinical data
        if self.use_cli == "surv":
            cli, times, events = self.cli[index], self.times[index], self.events[index]
            return (data_dict["image"], cli), data_dict["mask"], events, times, self.mr[index]
        elif self.use_cli == "deep_surv":
            return data_dict["image"], self.cli[index], self.times[index], self.events[index], self.mr[index]
        else:
            return data_dict["image"], data_dict["mask"], self.mr[index]

    def __len__(self):
        return len(self.mr)

    @staticmethod
    def __read_Nifit__(dicts):
        image = sitk.ReadImage(dicts['image'])
        mask = sitk.ReadImage(dicts['mask'])
        np_image = sitk.GetArrayFromImage(image)
        np_mask = sitk.GetArrayFromImage(mask)
        np_image = np_image.astype('float32')
        # np_image = np.expand_dims(np_image, axis=0)
        np_mask = np_mask.astype('uint8')

        dicts = {"image": np_image, "mask": np_mask}
        return dicts

    TensorOrArray = Union[torch.Tensor, np.ndarray]

    def make_time_bins(self, times: TensorOrArray,
                       num_bins: Optional[int] = None,
                       use_quantiles: bool = False,
                       event: Optional[TensorOrArray] = None) -> torch.Tensor:
        """Creates the bins for survival time discretisation.

        By default, sqrt(num_observation) bins corresponding to the quantiles of
        the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

        Parameters
        ----------
        times
            Array or tensor of survival times.
        num_bins
            The number of bins to use. If None (default), sqrt(num_observations)
            bins will be used.
        use_quantiles
            If True, the bin edges will correspond to quantiles of `times`
            (default). Otherwise, generates equally-spaced bins.
        event
            Array or tensor of event indicators. If specified, only samples where
            event == 1 will be used to determine the time bins.

        Returns
        -------
        torch.Tensor
            Tensor of bin edges.
        """
        # TODO this should handle arrays and (CUDA) tensors
        if event is not None:
            times = times[event == 1]
        if num_bins is None:
            num_bins = ceil(sqrt(len(times)))
        if use_quantiles:
            # NOTE we should switch to using torch.quantile once it becomes
            bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
        else:
            # print(f'time.min:{times.min()}, {times.max()}, {num_bins}')
            bins = np.linspace(times.min(), times.max(), num_bins)
        bins = torch.tensor(bins, dtype=torch.float)
        return bins

    def encode_survival(self, time: Union[float, int, TensorOrArray],
                        event: Union[int, bool, TensorOrArray],
                        bins: TensorOrArray) -> torch.Tensor:
        """Encodes survival time and event indicator in the format
        required for MTLR training.

        For uncensored instances, one-hot encoding of binned survival time
        is generated. Censoring is handled differently, with all possible
        values for event time encoded as 1s. For example, if 5 time bins are used,
        an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
        instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
        'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

        Parameters
        ----------
        time
            Time of event or censoring.
        event
            Event indicator (0 = censored).
        bins
            Bins used for time axis discretisation.
        Returns
        -------
        torch.Tensor
            Encoded survival times.
        """
        # TODO this should handle arrays and (CUDA) tensors
        if not isinstance(time, (float, int, np.ndarray)):
            time = np.atleast_1d(time)
            time = torch.tensor(time)
        if not isinstance(event, (int, bool, np.ndarray)):
            event = np.atleast_1d(event)
            event = torch.tensor(event)

        if isinstance(bins, np.ndarray):
            bins = torch.tensor(bins)

        try:
            device = bins.device
        except AttributeError:
            device = "cpu"

        time = np.clip(time, 0, bins.max())
        # add extra bin [max_time, inf) at the end
        y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                        dtype=torch.float,
                        device=device)

        # so we need to set it to True
        bin_idxs = torch.bucketize(time, bins, right=True)
        for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
            if e == 1:
                y[i, bin_idx] = 1
            else:
                y[i, bin_idx:] = 1
        return y.squeeze()


if __name__ == '__main__':
    pass
