#Copyright 2020 Side Li, Arun Kumar
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import torch

from groupml.models import UnetGenerator
from groupml.dataset import Storage, DataType
from groupml.api import run_group_learning
from groupml.operators import Workload


def unet_creator():
    return UnetGenerator(3, 34, 8)


def adam_creator(model, lr=1e-5, l2=0):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)


groups = [
    'erfurt', 'augsburg', 'hamburg', 'hanover', 'oberhausen', 'weimar', 'mannheim', 'bayreuth', 'nuremberg'
]

run_group_learning(
    metadata_path="hdfs:///cityscape/metadata.json",
    groups=groups,
    search_space={
        "lr": [1e-2, 1e-3, 1e-4, 1e-5],
        "l2": [0, 1e-3, 1e-2]
    },
    num_workers=4,
    num_cpus_per_worker=20,
    model_creator=unet_creator,
    model_params={
       "n_class":  34
    },
    optim_creator=adam_creator,
    optim_params={
    },
    data_params={
        "storage": Storage.HDFS,
        "eager": True,
        "remote": True,
        "cache": False,
        "data_type": DataType.IMAGE
    },
    loader_params={
        "batch_size": 16,
    },
    storage=Storage.HDFS,
    workload=Workload.SEQUENTIAL,
    verbose=True,
    log_path="/gl_results/",
    use_gpu=True
)

