# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import io
import numpy as np
from zipfile import ZipFile

from PIL import Image
from torch.utils.data import Dataset

class ZipImageNetDataset(datasets.VisionDataset):
    def __init__(self,
                 dataroot: str,
                 train: bool = True,
                 transforms=None,
                 transform=None,
                 target_transform=None):
        super().__init__(root=dataroot,
                         transforms=transforms,
                         transform=transform,
                         target_transform=target_transform)
        self.zfpath = os.path.join(
            dataroot,
            f"{'train' if train else 'val'}_blurred.zip",
        )

        # Images are structured in directories based on class
        with open(os.path.join(dataroot, "map_clsloc.txt")) as f:
            def parse_row(row: str) -> tuple[str, int]:
                classname, classnum, _ = row.split()
                return classname, (int(classnum) - 1)
            self.classes: dict[str, int] = dict(parse_row(row) for row in f)

        # Avoid reusing the file handle created here, for known issue with multi-worker:
        # https://discuss.pytorch.org/t/dataloader-with-zipfile-failed/42795
        self.zf = None
        with ZipFile(self.zfpath) as zf:
            self.imglist: list[str] = [
                path for path in zf.namelist()
                if path.endswith(".jpg")
            ]
            self.image0 = Image.open(io.BytesIO(zf.read(self.imglist[0])))
            self.label0 = self.get_label(self.imglist[0])


    def get_label(self, path: str) -> int:
        if not path.endswith(".jpg"):
            raise ValueError(f"Expected path to image, got {path}")
        classname: str = path.split("/")[-2]
        return self.classes[classname]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx: int):
        if self.zf is None:
            self.zf=ZipFile(self.zfpath)
        imgpath = self.imglist[idx]
        img = self.image0  # Image.open(io.BytesIO(self.zf.read(imgpath)))
        label = self.label0  # self.get_label(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


class INatDataset(ImageFolder):
    def __init__(self,
                 root,
                 train=True,
                 year=2018,
                 transform=None,
                 target_transform=None,
                 category='name',
                 loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_imagenetv2(args):
    transform = build_transform(False, args)
    root = os.path.join(args.data_path, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 1000
    return dataset, nb_classes

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.zip_dataloader and not (args.data_set == 'IMNET'): raise NotImplementedError()

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if args.zip_dataloader:
            dataset = ZipImageNetDataset(args.data_path,
                                         train=is_train,
                                         transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
