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


from collections import defaultdict
from enum import Enum
import torch


class Keys(Enum):
    MODEL = 1
    MODEL_PARAMS = 2
    OPTIM = 3
    OPTIM_PARAMS = 4
    DATA_PARAMS = 5
    SEARCH_SPACE = 6
    LOADER_PARAMS = 7
    TRAIN_PARAMS = 8


class Workload(Enum):
    ALGEBRAIC = 1
    SEQUENTIAL = 2
    SAMPLING = 3
    HYBRID = 4


class Configuration(object):
    def __init__(self, json_input=None):
        self.configs = defaultdict(dict)
        self.workload = Workload.ALGEBRAIC
        if json_input is not None:
            self.configs = json_input.copy()

    def register_group(self, group, model, model_params, optim, optim_params, data_params, loader_params,
                       search_space=None, train_params=None):
        self.add_model(group, model)
        self.add_model_params(group, model_params)
        self.add_optimizer(group, optim)
        self.add_optimizer_params(group, optim_params)
        self.add_search_space(group, search_space)
        self.add_data_params(group, data_params)
        self.add_loader_params(group, loader_params)
        self.add_train_params(group, train_params)

    def register_groups(self, groups, model, model_params, optim, optim_params, data_params, loader_params,
                        search_space=None, train_params=None, criterion=None):
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        else:
            self.criterion = criterion(reduction='sum')
        for group in groups:
            self.register_group(group, model, model_params, optim, optim_params, data_params, loader_params,
                                search_space, train_params)

    def get_group_config(self, group):
        return self.configs[group]

    def get_new_model(self, group, add_params=None):
        if add_params is None:
            add_params = {}
        return self.get_model(group)(**{**self.get_model_params(group), **add_params})

    def get_new_optimizer(self, group, model, add_params=None):
        if add_params is None:
            add_params = {}
        return self.get_optimizer(group)(model, **{**self.get_optimizer_params(group), **add_params})

    def add_model(self, group, model_creator):
        self.configs[group][Keys.MODEL] = model_creator

    def get_model(self, group):
        return self.configs[group][Keys.MODEL]

    def add_model_params(self, group, model_params):
        self.configs[group][Keys.MODEL_PARAMS] = model_params

    def get_model_params(self, group):
        return self.configs[group][Keys.MODEL_PARAMS]

    def add_optimizer(self, group, optim_creator):
        self.configs[group][Keys.OPTIM] = optim_creator

    def get_optimizer(self, group):
        return self.configs[group][Keys.OPTIM]

    def add_optimizer_params(self, group, optim_params):
        self.configs[group][Keys.OPTIM_PARAMS] = optim_params

    def get_optimizer_params(self, group):
        return self.configs[group][Keys.OPTIM_PARAMS]

    def add_data_params(self, group, data_params):
        self.configs[group][Keys.DATA_PARAMS] = data_params

    def get_data_params(self, group):
        return self.configs[group][Keys.DATA_PARAMS]

    def add_loader_params(self, group, loader_params):
        self.configs[group][Keys.LOADER_PARAMS] = loader_params

    def get_loader_params(self, group):
        return self.configs[group][Keys.LOADER_PARAMS]

    def add_train_params(self, group, train_params):
        self.configs[group][Keys.TRAIN_PARAMS] = train_params

    def get_train_params(self, group):
        return self.configs[group][Keys.TRAIN_PARAMS]

    def add_search_space(self, group, search_space):
        self.configs[group][Keys.SEARCH_SPACE] = search_space

    def get_search_space(self, group):
        return self.configs[group][Keys.SEARCH_SPACE]

    def get_group_names(self):
        return list(self.configs.keys())

    def set_workload_type(self, workload):
        self.workload = workload

    def get_workload_type(self):
        return self.workload

    def set_criterion(self, criterion):
        self.criterion = criterion

    def get_criterion(self):
        return self.criterion
