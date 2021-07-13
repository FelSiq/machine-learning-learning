import typing as t
import os
import collections

import torchvision
import torch
import tqdm.auto
import numpy as np
import PIL
import matplotlib.pyplot as plt
import bpemb


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        img,
        descs,
        batch_size: int,
        shuffle: bool = True,
        resample_descs: bool = True,
        transforms: t.Optional[torchvision.transforms.Compose] = None,
    ):
        assert len(img) == len(descs)
        assert int(batch_size) > 0

        super(IterableDataset, self).__init__()

        self.img = img
        self.descs = descs
        self.shuffle = bool(shuffle)
        self.resample_descs = bool(resample_descs)
        self.chosen_desc_inds = None
        self.chosen_descs = None
        self.transforms = transforms

        self.start = 0
        self.batch_size = int(batch_size)

    def __len__(self):
        return int(np.ceil(len(self.img) / self.batch_size))

    def __next__(self):
        if self.start >= len(self.chosen_descs):
            raise StopIteration

        end = self.start + self.batch_size

        batch_img = self.img[self.start : end]
        batch_descs = self.chosen_descs[self.start : end]

        if self.transforms is not None:
            batch_img = self.transforms(batch_img)

        self.start = end

        return (batch_img, batch_descs)

    def __iter__(self):
        n = len(self.img)

        if self.shuffle:
            inds = np.arange(n)
            np.random.shuffle(inds)
            self.img = self.img[inds, ...]
            self.descs = [self.descs[i] for i in inds]

        if self.chosen_desc_inds is None or self.resample_descs:
            self.chosen_desc_inds = np.random.randint(5, size=n)

        self.chosen_descs = []

        for i, chosen_desc_ind in enumerate(self.chosen_desc_inds):
            self.chosen_descs.append(self.descs[i][chosen_desc_ind])

        self.start = 0

        return self


def gather_data_from_disc(
    imgs_uri: str = "./images",
    desc_uri: str = "./descriptions",
    img_shape: t.Tuple[int, int] = (32, 32),
    mean: t.Tuple[int, int, int] = (0.485, 0.456, 0.406),
    std: t.Tuple[int, int, int] = (0.229, 0.224, 0.225),
):
    script_path = os.path.dirname(os.path.realpath(__file__))

    imgs_uri = os.path.join(script_path, imgs_uri)
    desc_uri = os.path.join(script_path, desc_uri)

    n = len(os.listdir(imgs_uri))

    imgs = np.empty((n, *img_shape, 3), dtype=np.float32)
    descriptions = []  # type: t.List[t.List[str]]

    for i in tqdm.auto.tqdm(range(n)):
        name_img = os.path.join(imgs_uri, f"{i}.jpg")
        name_desc = os.path.join(desc_uri, f"{i}.txt")

        img = PIL.Image.open(name_img)
        img = img.resize(img_shape)
        img = np.asfarray(img)

        with open(name_desc, "r") as f:
            desc = f.read().lower().splitlines()

        imgs[i, ...] = img
        descriptions.append(desc)

    imgs = (imgs / 255.0 - mean) / std
    imgs = np.moveaxis(imgs, -1, 1)
    imgs = imgs.astype(np.float32, copy=False)

    return imgs, descriptions


def word_to_int_(img_descriptions, codec):
    for i, descriptions in tqdm.auto.tqdm(enumerate(img_descriptions)):
        descriptions = codec.encode_ids_with_bos_eos(descriptions)
        img_descriptions[i] = [
            torch.tensor(desc, dtype=torch.long) for desc in descriptions
        ]


def get_data(
    batch_size_train: int = 32,
    batch_size_eval: int = 32,
    eval_size: int = 64,
    vs: int = 3000,
    dim: int = 50,
    img_shape: t.Tuple[int, int] = (32, 32),
    transforms: t.Optional[torchvision.transforms.Compose] = None,
):
    imgs, descriptions = gather_data_from_disc(img_shape=img_shape)
    codec = bpemb.BPEmb(lang="en", vs=vs, dim=dim, add_pad_emb=True)
    word_to_int_(descriptions, codec)

    imgs = torch.from_numpy(imgs)

    m = len(descriptions)

    imgs_eval, imgs_train = imgs[:eval_size], imgs[eval_size:m]
    desc_eval, desc_train = descriptions[:eval_size], descriptions[eval_size:]

    dataloader_train = IterableDataset(
        imgs_train, desc_train, batch_size=batch_size_train
    )
    dataloader_eval = IterableDataset(
        imgs_eval,
        desc_eval,
        batch_size=batch_size_eval,
        shuffle=False,
        resample_descs=False,
        transforms=transforms,
    )

    return dataloader_train, dataloader_eval, codec


if __name__ == "__main__":
    prepare_data()
