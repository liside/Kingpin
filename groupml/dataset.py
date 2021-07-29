from torch.utils.data import Dataset
import bisect
import pandas as pd
from enum import Enum
from hdfs import InsecureClient
import torch
import ray
import os
from filelock import FileLock
from io import StringIO, BytesIO
import sys
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
import socket
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)
import gc
from ray.util.sgd.utils import check_for_failure
import boto3


class DataType(Enum):
    CSV = 1
    IMAGE = 2


class Storage(Enum):
    FS = 1
    HDFS = 2
    S3 = 3

AWS_SECRET_ACCESS_KEY="AWS_SECRET_ACCESS_KEY"
AWS_ACCESS_KEY_ID="AWS_ACCESS_KEY_ID"

def load_one_file(path, dtype=DataType.CSV, storage=Storage.FS, cache=False, client=None, tmp_path="/mnt2/tmp"):
    if dtype == DataType.CSV:
        if cache and os.path.exists(tmp_path + path):
            with FileLock(tmp_path + "/" + path.replace("\\", "") + ".lock"):
                try:
                    with open(tmp_path + path) as reader:
                        return pd.read_csv(reader, header=None, delimiter=",").to_numpy(dtype=np.float32)
                except:
                    os.remove(tmp_path + path)
        if storage == Storage.HDFS:
            if client is None:
                client = InsecureClient('http://namenode:9870')
            with client.read(path, encoding='utf-8') as reader:
                content = reader.read()
                data = pd.read_csv(StringIO(content), header=None, delimiter=",").to_numpy(dtype=np.float32)
                if cache:
                    with FileLock(tmp_path + "/" + path.replace("/", "") + ".lock"):
                        if not os.path.exists(tmp_path + path):
                            touch(tmp_path + path)
                            with open(tmp_path + path, "w+") as writer:
                                writer.write(content)
                return data
        elif storage == Storage.FS:
            with open(path) as reader:
                return pd.read_csv(reader, header=None, delimiter=",").to_numpy(dtype=np.float32)
    elif dtype == DataType.IMAGE:
        if storage == Storage.HDFS:
            if cache and os.path.exists(tmp_path + path):
                with FileLock(tmp_path + "/" + path.replace("\\", "") + ".lock"):
                    try:
                        with open(tmp_path + path, mode="rb", encoding="ISO-8859-1") as reader:
                            return BytesIO(reader.read())
                    except:
                        os.remove(tmp_path + path)
            if client is None:
                client = InsecureClient('http://namenode:9870')
            with client.read(path, encoding="ISO-8859-1") as reader:
                content = reader.read()
                if cache:
                    with FileLock(tmp_path + "/" + path.replace("/", "") + ".lock"):
                        if not os.path.exists(tmp_path + path):
                            touch(tmp_path + path)
                            with open(tmp_path + path, "w+") as writer:
                                writer.write(content)
                return BytesIO(bytes(content, encoding="ISO-8859-1"))
        elif storage == Storage.FS:
            with open(path, mode="rb") as reader:
                return BytesIO(reader.read())
        elif storage == Storage.S3:
            path = path[1:].split("/", 1)
            bucket_name, file_path = path[0], path[1]
            s3 = boto3.resource('s3',
                                aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key= AWS_SECRET_ACCESS_KEY,
                                region_name='us-west-2')
            bucket = s3.Bucket(bucket_name)
            object = bucket.Object(file_path)

            file_stream = BytesIO()
            object.download_fileobj(file_stream)

            return file_stream

@ray.remote
def load_file(path):
    client = InsecureClient('http://namenode:9870')
    with client.read(path, encoding='utf-8') as reader:
        content = reader.read()
        data = pd.read_csv(StringIO(content), header=None, delimiter=",").to_numpy(dtype=np.float16)

        return data

@ray.remote(num_cpus=1)
def remote_read_one_file(path, dtype=DataType.CSV, storage=Storage.FS, cache=False, client=None, tmp_path="/mnt2/tmp"):
    if dtype == DataType.CSV:
        if cache and os.path.exists(tmp_path + path):
            with FileLock(tmp_path + "/" + path.replace("\\", "") + ".lock"):
                try:
                    with open(tmp_path + path) as reader:
                        return pd.read_csv(reader, header=None, delimiter=",").to_numpy(dtype=np.float32)
                except:
                    os.remove(tmp_path + path)
        if storage == Storage.HDFS:
            if client is None:
                client = InsecureClient('http://namenode:9870')
            with client.read(path, encoding='utf-8') as reader:
                content = reader.read()
                data = pd.read_csv(StringIO(content), header=None, delimiter=",").to_numpy(dtype=np.float32)
                if cache:
                    with FileLock(tmp_path + "/" + path.replace("/", "") + ".lock"):
                        if not os.path.exists(tmp_path + path):
                            touch(tmp_path + path)
                            with open(tmp_path + path, "w+") as writer:
                                writer.write(content)
                return data
        elif storage == Storage.FS:
            with open(path) as reader:
                return pd.read_csv(reader, header=None, delimiter=",").to_numpy(dtype=np.float32)
    elif dtype == DataType.IMAGE:
        if storage == Storage.HDFS:
            if cache and os.path.exists(tmp_path + path):
                with FileLock(tmp_path + "/" + path.replace("\\", "") + ".lock"):
                    try:
                        with open(tmp_path + path, mode="rb") as reader:
                            return BytesIO(reader.read())
                    except:
                        os.remove(tmp_path + path)
            if client is None:
                client = InsecureClient('http://namenode:9870')
            with client.read(path, encoding="ISO-8859-1") as reader:
                content = reader.read()
                if cache:
                    with FileLock(tmp_path + "/" + path.replace("/", "") + ".lock"):
                        if not os.path.exists(tmp_path + path):
                            touch(tmp_path + path)
                            with open(tmp_path + path, "w+") as writer:
                                writer.write(content)
                return BytesIO(bytes(content, encoding="ISO-8859-1"))
        elif storage == Storage.FS:
            with open(path, mode="rb") as reader:
                return BytesIO(reader.read())
        elif storage == Storage.S3:
            path = path[1:].split("/", 1)
            bucket_name, file_path = path[0], path[1]
            s3 = boto3.resource('s3',
                                aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key= AWS_SECRET_ACCESS_KEY,
                                region_name='us-west-2')
            bucket = s3.Bucket(bucket_name)
            object = bucket.Object(file_path)

            file_stream = BytesIO()
            object.download_fileobj(file_stream)

            return file_stream


class DummyDataset(Dataset):
    def __init__(self, length):
        self._len = length

    def __getitem__(self, idx):
        return np.ones(1)

    def __len__(self):
        return self._len


class CSVDataset(Dataset):
    def __init__(self, path, length, eager=True, cache=False, remote=None, loader=None,
                 storage=Storage.FS, first_col_target=True, client=None, indexes=None,
                 whole_data=None, whole_target=None, retry_times=3, pg=None):
        self._data = None
        self._target = None
        self._remote = remote
        self._path = path
        self._len = length
        self._first_col_target = first_col_target
        self._storage = storage
        self._cache = cache
        self._loader = loader
        self._client = client
        self._indexes = indexes
        self._whole_data = whole_data
        self._whole_target = whole_target
        self._retry_times = retry_times
        self._pg = pg
        if eager:
            self.load_csv()

    def __getitem__(self, idx):
        if self._data is None:
            self.load_csv()
        if self._first_col_target:
            return self._data[idx], self._target[idx]
        else:
            return self._data[idx]

    def __len__(self):
        return self._len

    def load_csv(self):
        remote_ref = None
        if self._remote:
            count = 0
            while True:
                try:
                    remote_ref = remote_read_one_file.options(placement_group=self._pg,
                                                              num_cpus=1, resources={socket.gethostname(): 1})\
                        .remote(self._path, DataType.CSV, self._storage, self._cache)
                    data = ray.get(remote_ref)
                    break
                except:
                    e = sys.exc_info()[0]
                    print("failed", count, "times with error", e)
                    count += 1
                    if count >= self._retry_times:
                        raise
        elif self._loader is not None:
            data = ray.get(self._loader)
        else:
            count = 0
            while True:
                try:
                    data = load_one_file(self._path, dtype=DataType.CSV, storage=self._storage, client=self._client,
                                         cache=self._cache)
                    break
                except:
                    e = sys.exc_info()[0]
                    print("failed", count, "times with error", e)
                    count += 1
                    if count >= self._retry_times:
                        raise

        if self._indexes is not None:
            if self._first_col_target:
                self._whole_data[self._indexes[0]:self._indexes[1]] = torch.from_numpy(data[:, 1:]).float()
                self._whole_target[self._indexes[0]:self._indexes[1]] = torch.from_numpy(data[:, 0]).long()
            else:
                self._whole_data[self._indexes[0]:self._indexes[1]] = torch.from_numpy(data).float()
        else:
            if self._first_col_target:
                self._data = torch.from_numpy(data[:, 1:]).float()
                self._target = torch.from_numpy(data[:,0]).long()
            else:
                self._data = torch.from_numpy(data)

        del data
        if remote_ref is not None:
            ray.internal.free(remote_ref)
        # gc.collect()

    def get_data(self):
        if self._data is None:
            self.load_csv()
        return self._data.numpy(), self._target.numpy()

    def clear(self):
        self._data = None
        self._target = None
        gc.collect()


def _transform(image, mask):
    # Random rotations to improve rotations invariance
    angle = transforms.RandomRotation.get_params([-15, 15])
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)

    # # Resize
    # resize = transforms.Resize((256, 128))
    # image = resize(image)
    # mask = resize(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
         image, output_size=(512, 512))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Convert to numpy array
    image = TF.to_tensor(image).numpy()
    mask = np.asarray(mask)
    return image, mask


def load_image(params):
    path, storage, cache, train, transform, remote = params['path'], params['storage'], params['cache'], \
                                                     params['train'], params['transform'], params['remote']
    img_name = path
    # TODO: fix this hardcoded path
    label_name = path \
        .replace("images", "labels") \
        .replace("leftImg8bit", "gtCoarse_labelIds")

    if remote:
        remote_func = remote_read_one_file.options(num_cpus=1, resources={socket.gethostname(): 1})
        remote_ref = remote_func.remote(img_name, DataType.IMAGE, storage, cache)
        data = ray.get(remote_ref)
        img = Image.open(data).convert("RGB")
        del data
        if remote_ref is not None:
            ray.internal.free(remote_ref)

        remote_ref = remote_func.remote(label_name, DataType.IMAGE, storage, cache)
        data = ray.get(remote_ref)
        label = Image.open(data).convert("L")
        del data
        if remote_ref is not None:
            ray.internal.free(remote_ref)
    else:
        img = Image.open(load_one_file(img_name, DataType.IMAGE, storage, cache=cache)).convert("RGB")
        label = Image.open(load_one_file(label_name, DataType.IMAGE, storage, cache=cache)).convert("L")

    # perform transformations
    if train:
        img, label = transform(img, label)
        img = img[::-1, :, :]  # switch to BGR
    else:
        img = np.asarray(img)
        label = np.asarray(label)

        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.

    img_tensor = torch.from_numpy(img.copy()).float()
    label_tensor = torch.from_numpy(label.copy()).long()
    return img_tensor, label_tensor


class ImageDataset(Dataset):
    def __init__(self, paths, means, transform, storage=Storage.FS, train=False, cache=False, remote=False,
                 eager=False):
        self.paths = paths
        self.means = means
        self.train = train
        self._parsed = {}
        self._transform = transform
        self._storage = storage
        self._cache = cache
        self._remote = remote
        self._client = None
        if storage == Storage.HDFS:
            self._client = InsecureClient('http://namenode:9870')
        if eager:
            self._load()

    def _load(self):
        params = [{
            "path": self.paths[i]["file_path"],
            "storage": self._storage,
            "cache": self._cache,
            "train": self.train,
            "transform": self._transform,
            "remote": self._remote
        } for i in range(len(self.paths))]
        with ThreadPoolExecutor(max_workers=20) as executor:
            self._parsed = {i: res for i, res in enumerate(executor.map(load_image, params))}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx in self._parsed:
            return self._parsed[idx]

        img_name = self.paths[idx]["file_path"]
        # TODO: fix this hardcoded path
        label_name = self.paths[idx]["file_path"]\
            .replace("images", "labels")\
            .replace("leftImg8bit", "gtCoarse_labelIds")
        img = Image.open(load_one_file(img_name, DataType.IMAGE, self._storage, client=self._client,
                                       cache=self._cache)).convert("RGB")
        label = Image.open(load_one_file(label_name, DataType.IMAGE, self._storage, client=self._client,
                                         cache=self._cache)).convert("L")

        # perform transformations
        if self.train:
            img, label = self._transform(img, label)
            img = img[::-1, :, :]  # switch to BGR
        else:
            img = np.asarray(img)
            label = np.asarray(label)

            img = img[:, :, ::-1]  # switch to BGR
            img = np.transpose(img, (2, 0, 1)) / 255.

        # reduce mean
        # img[0] -= self.means[0]
        # img[1] -= self.means[1]
        # img[2] -= self.means[2]

        # convert to tensor
        img_tensor = torch.from_numpy(img.copy()).float()
        label_tensor = torch.from_numpy(label.copy()).long()

        del img
        img = None
        del label
        label = None
        # cache images
        self._parsed[idx] = (img_tensor, label_tensor)

        return img_tensor, label_tensor


def _init_dataset(params):
    return CSVDataset(**params)


def touch(path):
    os.makedirs(path + "tmp", exist_ok=True)
    with open(path, 'a'):
        os.utime(path, None)


class ParallelDataset(Dataset):
    def __init__(self, train_files, data_type=DataType.CSV, dimension=(100,),
                 storage=Storage.FS, num_jobs=1, eager=True, cache=False, train=False, remote=False, aggregate=False,
                 placement=None):
        """

        :param train_files:
        :param data_type:
        :param num_jobs:
        :param eager:
        :param cache:
        """
        super(ParallelDataset, self).__init__()
        self.num_jobs = num_jobs
        self.eager = eager
        self.cache = cache
        self.data_type = data_type
        self.train_files = train_files
        self.data_executor = None
        self.lgb_dataset = None
        self.data = None
        self.target = None
        if data_type == DataType.CSV:
            self.cumulative_sizes = self.cumulative_sum([train_file["num_examples"] for train_file in train_files])
            if len(self.cumulative_sizes) == 0:
                self._len = 0
                return
            else:
                self._len = self.cumulative_sizes[-1]
            self.csv_files = [
                {"path": train_file["file_path"],
                 "length": train_file["num_examples"],
                 "eager": eager,
                 "cache": cache,
                 "remote": remote,
                 "loader": train_file["loader"] if "loader" in train_file else None,
                 "storage": storage} for train_file in train_files
            ]
            self.datasets = {}
            if eager:
                print("===============start to load data eagerly================")
                if self.data_executor is None:
                    self.data_executor = ThreadPoolExecutor(max_workers=num_jobs)
                pg = placement
                if placement is None:
                    pg = placement_group([{socket.gethostname(): num_jobs, "CPU": num_jobs}], strategy="STRICT_PACK")
                if aggregate:
                    self.data = torch.empty([self.cumulative_sizes[-1]] + list(dimension), dtype=torch.float)
                    self.target = torch.empty((self.cumulative_sizes[-1],), dtype=torch.long)
                    ray.get(pg.ready())
                    for i, file in enumerate(self.csv_files):
                        file["pg"] = pg
                        file["whole_data"] = self.data
                        file["whole_target"] = self.target
                        file["indexes"] = (self.cumulative_sizes[i - 1] if i > 0 else 0, self.cumulative_sizes[i])
                self.datasets = {i: res for i, res in enumerate(self.data_executor.map(_init_dataset,
                                                                                           self.csv_files))}
                if placement is None:
                    remove_placement_group(pg)
                print("===============finish loading data eagerly================")
            else:
                if storage == Storage.HDFS:
                    client = InsecureClient('http://namenode:9870')
                    for file in self.csv_files:
                        file["client"] = client
                for i, file in enumerate(self.csv_files):
                    self.datasets[i] = _init_dataset(file)

        elif data_type == DataType.IMAGE:
            self._len = len(train_files)
            # TODO: do not hardcode
            means = np.array([103.939, 116.779, 123.68]) / 255
            self.dataset = ImageDataset(train_files, means, _transform, storage, train, cache, eager, remote)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.data_type == DataType.CSV:
            if self.data is not None and self.target is not None:
                return self.data[idx], self.target[idx]

            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
                self.datasets[dataset_idx - 1].clear()
            return self.datasets[dataset_idx][sample_idx]
        elif self.data_type == DataType.IMAGE:
            return self.dataset[idx]

    @staticmethod
    def cumulative_sum(sequence):
        r, s = [], 0
        for l in sequence:
            # l = len(e)
            r.append(l + s)
            s += l
        return r

    def to_lightgbm(self):
        if self._len == 0:
            return None
        if self.lgb_dataset is not None:
            return self.lgb_dataset
        if self.data_type == DataType.CSV:
            self.lgb_dataset = lgb.Dataset(self.data.numpy(), self.target.numpy(), free_raw_data=False,
                                           params={"pre_partition": True}).construct()
            return self.lgb_dataset

    def to_numpy(self):
        return self.data, self.target


class SimpleDataset(Dataset):
    def __init__(self, tensors, cumulative_sizes):
        self.data = tensors
        self.cumulative_sizes = cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]

        data = torch.from_numpy(self.data[dataset_idx][sample_idx, :]).float()
        return data[1:], data[0].long()


def load_data_to_object_store(train_files, data_type=DataType.CSV, storage=Storage.HDFS):
    if data_type == DataType.CSV:
        cumulative_sizes = ParallelDataset.cumulative_sum([train_file["num_examples"] for train_file in train_files])

        remote_read = load_file.options(num_cpus=1, resources={socket.gethostname(): 1})
        references = []
        for train_file in train_files:
            references.append(remote_read.remote(train_file["file_path"]))

        print("====================start loading================================")
        with ThreadPoolExecutor(max_workers=20) as pool:
            data = list(pool.map(lambda file: ray.get(remote_read.remote(file["file_path"])), train_files))
        print("====================finish loading================================")

        return ray.put(data), cumulative_sizes
