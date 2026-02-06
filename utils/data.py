import os
import os.path as osp

import numpy as np
import PIL.Image as PImage
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.folder import IMG_EXTENSIONS, DatasetFolder
from torchvision.transforms import InterpolationMode, transforms


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments_gt = data["moments_gt"]
            moments_x = data["moments_x"]
        else:
            moments_gt = data["moments_gt_flip"]
            moments_x = data["moments_x_flip"]

        return moments_gt, moments_x, target




def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class CenterCropTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.img_size)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def build_dataset(
    data_path: str, 
    final_reso: int,
    hflip=False, 
    use_cached=False,
):
    train_aug, val_aug = [

        CenterCropTransform(final_reso),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ], [
        CenterCropTransform(final_reso),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    if hflip: 
        train_aug.insert(1, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)

    # build dataset
    if use_cached:
        print(f"######### using cached dataset #######")

        if final_reso == 256:
            train_set = CachedFolder(osp.join(data_path, "train_cache_mvar"))
            val_set = CachedFolder(osp.join(data_path, "val_cache_mvar"))
        elif final_reso == 512:
            train_set = CachedFolder(osp.join(data_path, "train_cache_mvar_512"))
            val_set = CachedFolder(osp.join(data_path, "val_cache_mvar_512"))
        else:
            raise ValueError(f"final_reso={final_reso}")
    else:
        print(f"########### using image dataset #########")
        train_set = DatasetFolder(
            root=osp.join(data_path, 'train'), 
            loader=pil_loader, 
            extensions=IMG_EXTENSIONS, 
            transform=train_aug)
        val_set = DatasetFolder(
            root=osp.join(data_path, 'val'), 
            loader=pil_loader, 
            extensions=IMG_EXTENSIONS, 
            transform=val_aug)
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')

    return num_classes, train_set, val_set


class LabelOnlyDataset(Dataset):
    def __init__(self, label_list):
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        return self.label_list[idx]


def build_dataset_for_eval(
    num_classes: int,
    num_samples_per_class: int,
):
    label_list = [i for i in range(num_classes) for _ in range(num_samples_per_class)]

    val_set = LabelOnlyDataset(label_list)

    print(f"[Eval Dataset] {len(val_set)=}, {num_classes=}")

    return num_classes, val_set



def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
