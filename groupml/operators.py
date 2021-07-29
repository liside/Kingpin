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


from ray.util.sgd.torch import TrainingOperator
import psutil
import time
import datetime
from collections import defaultdict
import threading
import os
import gc
import json
from concurrent.futures import ThreadPoolExecutor
import socket
import sys


import GPUtil
import ray
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from hdfs import InsecureClient
import lightgbm as lgb
import numpy as np
from ray.util.sgd.torch import BaseTorchTrainable
from ray.util.sgd.utils import check_for_failure
from ray.util.placement_group import (
    placement_group,
    remove_placement_group
)

from .sampler import DataParallelSampler
from .configuration import Workload
from .dataset import ParallelDataset, load_one_file, Storage, DataType, touch, DummyDataset
from .utils import init_logging, safe_log_in_dict


@ray.remote(num_cpus=1)
class Counter(object):
    def __init__(self):
        self.counter = 0

    def increment(self):
        self.counter += 1
        return self.counter


@ray.remote(num_cpus=1, resources={"MetricResource": 1})
class MetricsOperator(object):
    def __init__(self, path, gpu=False):
        self._metrics = list()
        self._path = path + "resource.log"
        self._timestamp_path = path + "timestamp.log"
        self._gpu = gpu
        if not os.path.exists(path):
            touch(self._path)
        if not os.path.exists(self._timestamp_path):
            touch(self._timestamp_path)
        self.init_metrics()

    def init_metrics(self):
        self.base_metrics = {"network": {"sent": 0, "recv": 0}}
        self.base_metrics = self.sample_metrics()
        with open(self._path, 'w'):
            os.utime(self._path, None)
        with open(self._timestamp_path, "w"):
            os.utime(self._path, None)

    def start_collect_metrics(self):
        self._running = True
        self._metrics = list()
        self._metrics_thread = threading.Thread(target=self.background_sampling, args=(), daemon=True)
        self._metrics_thread.start()

    def background_sampling(self):
        while self._running:
            time.sleep(1)
            self._metrics.append(self.sample_metrics())

    def finish_collect_metrics(self):
        self._running = False
        self._metrics_thread.join()

    def sample_metrics(self, unit="MB"):
        weight = 1
        if unit == "MB":
            weight = 1024 * 1024
        elif unit == "GB":
            weight = 1024 * 1024 * 1024
        gpu = 0
        if self._gpu:
            gpu = [g.load for g in GPUtil.getGPUs()][0]
        network_stat = psutil.net_io_counters()
        result = {
            "time": str(datetime.datetime.utcnow()),
            "cpu": psutil.cpu_percent(interval=1),
            "mem": (psutil.virtual_memory().total - psutil.virtual_memory().available) / weight,
            "disk": psutil.disk_usage("/mnt2").used / weight,
            "network": {
                "sent": network_stat.bytes_sent / weight - self.base_metrics["network"]["sent"],
                "recv": network_stat.bytes_recv / weight - self.base_metrics["network"]["recv"]
            }
        }
        if self._gpu:
            result["gpu"] = max([g.load for g in GPUtil.getGPUs()][0], gpu)

        with open(self._path, "a+") as f:
            f.write(json.dumps(result) + "\n")

        return result

    def get_metrics(self):
        return self._metrics


def apply_to_all(trainer, func_name, params):
    """
    Apply the specified member function in the training operator.
    """
    remote_worker_results = []
    for i, worker in enumerate(trainer.worker_group.remote_workers):
        result = worker.apply_operator.remote(lambda op: getattr(op, func_name)(**params))
        remote_worker_results.append(result)

    success = check_for_failure(remote_worker_results)
    if success:
        return success, ray.get(remote_worker_results)
    return success, None


def as_trainable(trainer, override_tune_step=None):
    class TorchTrainable(BaseTorchTrainable):
        def step(self):
            if override_tune_step is not None:
                output = override_tune_step(
                    self._trainer, {"iteration": self.training_iteration})
                return output
            else:
                return super(TorchTrainable, self).step()

        def _create_trainer(self, tune_config):
            """Overrides the provided config with Tune config."""
            existing_config = trainer.config.copy()
            existing_config.update(tune_config)
            apply_to_all(trainer, "setup", {"config": existing_config, "init": False})
            return trainer

        def cleanup(self):
            # Handle cleaning by ourselves
            print("---------------------------cleanup called-------------------------------------")
            return
    return TorchTrainable


class BaseOperator(TrainingOperator):
    def setup(self, config):
        torch.set_num_threads(config["num_cpus"])
        return

    def init_metrics(self):
        self.base_metrics = {"network": {"sent": 0, "recv": 0}}
        self.base_metrics = self.sample_metrics()

    def start_collect_metrics(self):
        self._running = True
        self._metrics = list()
        self._metrics_thread = threading.Thread(target=self.background_sampling, args=(), daemon=True)
        self._metrics_thread.start()

    def background_sampling(self):
        while self._running:
            time.sleep(1)
            self._metrics.append(self.sample_metrics())

    def finish_collect_metrics(self):
        self._running = False
        self._metrics_thread.join()

    def sample_metrics(self, unit="MB"):
        weight = 1
        if unit == "MB":
            weight = 1024 * 1024
        elif unit == "GB":
            weight = 1024 * 1024 * 1024
        gpu = 0
        if self._use_gpu:
            gpu = [g.load for g in GPUtil.getGPUs()][0]
        network_stat = psutil.net_io_counters()
        result = {
            "time": str(datetime.datetime.utcnow()),
            "cpu": psutil.cpu_percent(interval=1),
            "mem": psutil.virtual_memory().used / weight,
            "disk": psutil.disk_usage("/mnt2").used,
            "network": {
                "sent": network_stat.bytes_sent / weight - self.base_metrics["network"]["sent"],
                "recv": network_stat.bytes_recv / weight - self.base_metrics["network"]["recv"]
            }
        }
        if self._use_gpu:
            result["gpu"] = max([g.load for g in GPUtil.getGPUs()][0], gpu)
        return result

    def get_metrics(self):
        return self._metrics

    def collective_loss(self, loss, gpu=False):
        tensor = torch.tensor(loss)
        if gpu:
            tensor = tensor.cuda()
        dist.all_reduce(tensor)

        return tensor

    def collective_max(self, value):
        world_size = dist.get_world_size()
        tensor_list = [torch.tensor(0) for _ in range(world_size)]
        tensor = torch.tensor(value)
        dist.all_gather(tensor_list, tensor)

        return max(tensor_list).item()

    def collective_avg_grad(self, model, loss):
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.zeros(param.size())
            # print(param.grad.size())
            dist.all_reduce(param.grad.data)

        tensor = torch.tensor(loss)
        dist.all_reduce(tensor)

        return tensor

    def force_exit(self):
        sys.exit()


def _merge_grad(model_a, grads):
    for p1, grad in zip(model_a.parameters(), grads):
        if p1.grad is None:
            p1.grad = grad
        else:
            p1.grad += grad


def _merge_model(model_a, model_b):
    for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
        if p1.grad is None:
            p1.grad = p2.grad
        else:
            p1.grad += p2.grad


class GLOperator(BaseOperator):
    def setup(self, config):
        init_logging(config["log_path"])
        torch.set_num_threads(config["num_cpus"])
        metadata = config.get("metadata")
        self.model_config = config.get("model_config")
        self.rank = dist.get_rank()
        self._lbfgs = self.model_config.get_workload_type() == Workload.ALGEBRAIC
        self.models = metadata[self.rank].get("models")
        self.optimizers = metadata[self.rank].get("optimizers")
        self.params = metadata[self.rank].get("params")
        self.distribution = metadata[self.rank].get("distribution")
        self.custom_val = config.get("custom_val")
        self.train_data = {}
        self.val_data = {}
        self.train = {}
        self.val = {}
        self.num_examples = {}
        self.num_val_examples = {}
        self.num_cpus = config["num_cpus"]
        self.num_epochs = 0
        self.gpu_models = {}
        self.gpu_optimizers = {}
        safe_log_in_dict(location=self.rank, epoch="", group="", timing="start",
                         action="load data", params="", result="")

        for group in metadata[self.rank]["groups"]:
            print("initializing....", group["name"])
            name = group["name"]
            self.num_examples[name] = group["total_examples"]
            train_dataset = ParallelDataset(group["train_files"],
                                            **{**self.model_config.get_data_params(name), **{"train": True}})
            self.num_val_examples[name] = group["total_val_examples"]
            val_dataset = ParallelDataset(group["val_files"],
                                          **self.model_config.get_data_params(name))
            self.train_data[name] = train_dataset
            self.val_data[name] = val_dataset

            self.train[name] = DataLoader(train_dataset, **self.model_config.get_loader_params(name))
            if self._use_gpu:
                # TODO: fix this hardcoding later
                loader_params = self.model_config.get_loader_params(name)
                loader_params["batch_size"] = 4
                self.val[name] = DataLoader(val_dataset, **loader_params)
            else:
                self.val[name] = DataLoader(val_dataset, **self.model_config.get_loader_params(name))
        self.criterion = self.model_config.get_criterion()

        safe_log_in_dict(location=self.rank, epoch="", group="", timing="finish",
                         action="load data", params="", result="")

        self.register_data(
            train_loader=DataLoader(DummyDataset(10)), validation_loader=DataLoader(DummyDataset(10))
        )
        self.init_metrics()

    def init_data(self, metadata):
        for group in metadata[self.rank]["groups"]:
            name = group["name"]
            self.num_examples[name] = group["total_examples"]
            self.train[name] = DataLoader(ParallelDataset(group["train_files"],
                                                          **self.model_config.get_data_params(name)),
                                          **self.model_config.get_loader_params(name))
            self.val[name] = DataLoader(ParallelDataset(group["val_files"],
                                                        **self.model_config.get_data_params(name)),
                                        **self.model_config.get_loader_params(name))

    def train_epoch(self, iterator, info):
        learn_result = defaultdict(dict)
        for group in self.models:
            learn_result[group] = defaultdict(dict)
            for param_config in self.models[group]:
                learn_result[group][param_config]["loss"] = info[group][param_config]["loss"]
        self.refresh_state(info)

        resource_metrics = self.get_metrics()

        return {
            "training": learn_result,
            "resource": resource_metrics
        }

    def validate(self, val_iterator, info):
        result = defaultdict(dict)
        for group in self.models:
            for config in self.models[group]:
                print("start validating", group, config)
                if self.models[group][config] in self.gpu_models and \
                        self.gpu_models[(group, config)][0] == self.models[group][config]:
                    model = self.gpu_models[(group, config)][1]
                else:
                    model = ray.get(self.models[group][config])
                val_loss = 0.0
                correct_count = 0

                if self.model_config.get_workload_type() == Workload.SAMPLING and self.val_data[group].data is not None:
                    inputs = self.val_data[group].data
                    targets = self.val_data[group].target
                    outputs = torch.from_numpy(model.predict(inputs.numpy()))
                    outputs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, targets.double())

                    val_loss += loss.item()
                    preds = (outputs > 0.5).long()
                    correct_count += (preds == targets).double().sum().item()
                else:
                    if self._use_gpu and not next(model.parameters()).is_cuda:
                        model = model.cuda()
                    if self.model_config.get_workload_type() != Workload.SAMPLING:
                        model.eval()

                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(self.val[group]):
                                batch_examples = np.prod(list(targets.size()[1:]))
                                if self._use_gpu:
                                    inputs = inputs.to("cuda")
                                    targets = targets.to("cuda")
                                outputs = model(inputs)
                                loss = self.criterion(outputs, targets)
                                val_loss += loss.item() / batch_examples
                                if len(targets) > 2:
                                    outputs = F.softmax(outputs, dim=1)
                                preds = torch.argmax(outputs, 1)
                                correct_count += (preds == targets).double().sum().item() / batch_examples
                                # if self._use_gpu:
                                #     del inputs
                                #     del targets
                                #     torch.cuda.empty_cache()
                result[group][config] = {
                    "val_loss": val_loss / self.num_val_examples[group],
                    "val_accuracy": correct_count / self.num_val_examples[group]
                }
                print("finish validating", group, config)

        torch.cuda.empty_cache()
        self.num_epochs += 1
        return result

    def calculate_grad(self):
        result = defaultdict(dict)
        for group in self.models:
            for config in self.models[group]:
                print("start working on", group, config)
                safe_log_in_dict(location=self.rank, epoch=self.num_epochs, group=group,
                                 timing="start", action="calculate_grad", params=config,
                                 result="")
                model = ray.get(self.models[group][config])
                optimizer = self.load_optimizer(self.optimizers[group][config], group, model)
                model.train()
                local_loss = 0.0
                for batch_idx, (inputs, targets) in enumerate(self.train[group]):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    local_loss += loss.item()

                grads = [para.grad for para in model.parameters()]
                result[group][config] = {
                    "grad": ray.put(grads),
                    "loss": local_loss
                }
                print("finish working on", group, config)
                safe_log_in_dict(location=self.rank, epoch=self.num_epochs, group=group,
                                 timing="finish", action="calculate_grad", params=config,
                                 result="")
                gc.collect(1)
        return result

    def aggregate_update(self, states):
        safe_log_in_dict(location=self.rank, epoch=self.num_epochs, group="",
                         timing="start", action="aggregate_update", params="",
                         result="")
        result = defaultdict(dict)
        states = states[self.rank]
        for group, state in states.items():
            result[group] = defaultdict(dict)
            for param_config in state:
                model = ray.get(self.models[group][param_config])
                optimizer = self.load_optimizer(self.optimizers[group][param_config], group, model)
                model.train()
                optimizer.zero_grad()
                loss = 0.0

                for i in range(len(state[param_config]['grads'])):
                    _merge_grad(model, ray.get(state[param_config]['grads'][i]))

                if hasattr(model, "l1") and model.l1 != 0:
                    for param in model.parameters():
                        loss += model.l1 * torch.norm(param, 1)
                    # loss = loss.item()
                loss = sum(state[param_config]['losses']) / self.num_examples[group]

                if self._lbfgs:
                    def closure():
                        return loss

                    optimizer.step(closure)
                else:
                    optimizer.step()

                result[group][param_config]['model'] = ray.put(model)
                result[group][param_config]['optimizer'] = ray.put(optimizer)
                result[group][param_config]['loss'] = loss
        safe_log_in_dict(location=self.rank, epoch=self.num_epochs, group="",
                         timing="finish", action="aggregate_update", params="",
                         result="")
        return result

    def train_one_config(self, runnables, states):
        safe_log_in_dict(location=self.rank, epoch=self.num_epochs, group="",
                         timing="start", action="train_one_config", params="",
                         result="")
        result = defaultdict(dict)
        for group in self.models:
            for config in runnables[self.rank][group]:
            # config = runnables[self.rank][group]
                t1 = time.time()
                print("start working on", group, config)
                if states[group]["models"][config] in self.gpu_models and \
                    self.gpu_models[(group, config)][0] == states[group]["models"][config]:
                    model = self.gpu_models[(group, config)][1]
                else:
                    model = ray.get(states[group]["models"][config])

                if self._use_gpu and not next(model.parameters()).is_cuda:
                    model = model.cuda()

                if states[group]["optimizers"][config] in self.gpu_optimizers and \
                    self.optimizers[(group, config)][0] == states[group]["optimizers"][config]:
                    optimizer = self.gpu_optimizers[(group, config)][1]
                else:
                    optimizer = self.load_optimizer(states[group]["optimizers"][config], group, model)

                model.train()
                print("init one model took", time.time() - t1)
                local_loss = 0.0
                for batch_idx, (inputs, targets) in enumerate(self.train[group]):
                    optimizer.zero_grad()
                    batch_examples = np.prod(list(targets.size()[1:]))
                    if self._use_gpu:
                        inputs = inputs.to("cuda")
                        targets = targets.to("cuda")
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    local_loss += loss.item() / self.num_examples[group] / batch_examples
                    # if self._use_gpu:
                    #     del inputs
                    #     del targets
                    #     torch.cuda.empty_cache()
                t2 = time.time()
                if states[group]["models"][config] in self.gpu_models and \
                    self.gpu_models[(group, config)][0] == states[group]["models"][config]:
                    model_id = states[group]["models"][config]
                    optim_id = states[group]["optimizers"][config]
                else:
                    model_id = ray.put(model.to("cpu") if self._use_gpu else model)
                    optim_id = ray.put(optimizer)

                result[group][config] = {
                    "model": model_id,
                    "optimizer": optim_id,
                    "loss": local_loss
                }

                self.gpu_models[(group, config)] = (model_id, model)
                self.gpu_optimizers[(group, config)] = (optim_id, optimizer)
                print("serialize took", time.time() - t2)
                print("finish working on", group, config)
            # gc.collect(1)
        safe_log_in_dict(location=self.rank, epoch=self.num_epochs, group="",
                         timing="finish", action="train_one_config", params="",
                         result="")
        return result

    def refresh_state(self, states):
        for group in self.models:
            for param_config in self.models[group]:
                self.models[group][param_config] = states[group][param_config]['model']
                if "optimizer" in states[group][param_config]:
                    self.optimizers[group][param_config] = states[group][param_config]['optimizer']

    def get_state(self):
        states = {}
        for group in self.models:
            states[str(group)] = {
                "model": self.models[group],
                "optimizer": self.optimizers[group]
            }

        return states

    def load_optimizer(self, op_id, group, model):
        # optimizer = self.model_config.get_optimizer(model, self.model_config)
        # optimizer.load_state_dict(ray.get(op_id).state_dict())
        optim_param_names = set(self.model_config.get_optimizer(group).__code__.co_varnames)
        params = {}
        for name in self.params:
            if name in optim_param_names:
                params[name] = self.params[name]

        optimizer = self.model_config.get_new_optimizer(group, model, params)
        optimizer.load_state_dict(ray.get(op_id).state_dict())
        return optimizer

    def calculate_grad_hess(self, remote=False):
        result = defaultdict(dict)
        if remote:
            for group in self.models:
                for config in self.models[group]:
                    print("start working on", group, config)
                    model = ray.get(self.models[group][config])
                    grads = []
                    hesses = []
                    local_loss = 0.0

                    with ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
                        map_result = list(executor.map(lambda data:
                                                       ray.get(cal_grad_hess.remote(
                                                           model,
                                                           self.criterion, data[0], data[1])),
                                                       self.train[group]))
                        for loss, grad, hess in map_result:
                            local_loss += loss
                            grads.append(grad)
                            hesses.append(hess)
                    result[group][config] = {
                        "grad": ray.put(torch.cat(grads)),
                        "hess": ray.put(torch.cat(hesses)),
                        "loss": local_loss
                    }
                    print("finish working on", group, config)
                    gc.collect(1)
            return result
        else:
            for group in self.models:
                for config in self.models[group]:
                    print("start working on", group, config)
                    # TODO: check if it's possible to directly seriealize lightgbm model using ray
                    model = ray.get(self.models[group][config])
                    inputs = self.train_data[group].data
                    targets = self.train_data[group].target
                    outputs = torch.from_numpy(model.predict(inputs.numpy()))
                    outputs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, targets.double())
                    grad = outputs - targets
                    hess = outputs * (1. - outputs)

                    result[group][config] = {
                        "grad": ray.put(grad),
                        "hess": ray.put(hess),
                        "loss": loss.item()
                    }
                    print("finish working on", group, config)
                    gc.collect(1)
            return result

    def boosting(self, states):
        result = defaultdict(dict)
        states = states[self.rank]
        for group, state in states.items():
            result[group] = defaultdict(dict)
            for param_config in state:
                print("boosting", group, param_config)
                model = ray.get(self.models[group][param_config])

                grad = torch.cat([ray.get(g) for g in state[param_config]["grads"]]).numpy()
                hess = torch.cat([ray.get(g) for g in state[param_config]["hesses"]]).numpy()
                loss = sum(state[param_config]['losses']) / self.num_examples[group]

                def loglikelihood(grad, hess, preds, train_data):
                    return grad, hess

                model = lgb.train({**self.model_config.get_train_params(group), **self.params[group][param_config]},
                                  self.train_data[group].to_lightgbm(),
                                  init_model=model if self.num_epochs > 0 else None,
                                  num_boost_round=1,
                                  fobj=lambda x, y: loglikelihood(grad, hess, x, y),
                                  keep_training_booster=True)

                result[group][param_config]['model'] = ray.put(model)
                result[group][param_config]['loss'] = loss

        return result

    def get_rank_ip(self):
        return {str(self.rank): socket.gethostbyname(socket.gethostname())}

    def set_rank_ips(self, ips):
        self.distribution["ips"] = ips

    def train_hybrid_gbm(self):
        result = defaultdict(dict)

        self.config["base_port"] = self.collective_max(self.config["base_port"])
        for group in self.models:
            result[group] = defaultdict(dict)
            for config in self.models[group]:
                print("start working on", group, config)
                model = ray.get(self.models[group][config])
                add_params = {}
                data_parallel = len(self.distribution[group]) > 1
                if data_parallel:
                    add_params["local_listen_port"] = self.config["base_port"]
                    add_params["machines"] = ",".join([self.distribution['ips'][str(rank)] + ":"
                                                       + str(add_params["local_listen_port"])
                                                       for rank in self.distribution[group]])
                    add_params["num_machines"] = len(self.distribution[group])
                    add_params["tree_learner"] = "data"
                    self.config['base_port'] += 1

                val_loss = [0]
                val_accuracy = [0]

                # The preds is already prob
                def logloss_eval(preds, data):
                    y_true = data.get_label()
                    targets = torch.from_numpy(y_true)
                    outputs = torch.from_numpy(preds)
                    loss = self.criterion(outputs, targets.double()).item()

                    preds = (outputs > 0.5).long()
                    correct_count = (preds == targets).double().sum().item()

                    val_loss[0] = loss / self.num_val_examples[group]
                    val_accuracy[0] = correct_count / self.num_val_examples[group]

                    return [("val_loss", val_loss[0], False), ("val_accuracy", val_accuracy[0], True)]

                model = lgb.train({**self.model_config.get_train_params(group),
                                   **self.params[group][config],
                                   **add_params},
                                  self.train_data[group].to_lightgbm(),
                                  valid_sets=self.val_data[group].to_lightgbm(),
                                  init_model=model if self.num_epochs > 0 else None,
                                  num_boost_round=1,
                                  feval=logloss_eval,
                                  keep_training_booster=True)

                result[group][config]['model'] = ray.put(model)
                result[group][config]['loss'] = val_loss[0]
                result[group][config]['val_loss'] = val_loss[0]
                result[group][config]['val_accuracy'] = val_accuracy[0]

        self.num_epochs += 1
        return result

    def train_hybrid_gbm_all(self, num_epochs):
        all_result = {i: {group: defaultdict(dict) for group in self.models} for i in range(num_epochs)}
        # self.config["base_port"] = self.collective_max(self.config["base_port"])
        for group in self.models:
            for config in self.models[group]:
                print("start working on", group, config)
                add_params = {}
                data_parallel = len(self.distribution[group]) > 1
                if data_parallel:
                    add_params["local_listen_port"] = self.config["base_port"]
                    add_params["machines"] = ",".join([self.distribution['ips'][str(rank)] + ":"
                                                       + str(add_params["local_listen_port"])
                                                       for rank in self.distribution[group]])
                    add_params["num_machines"] = len(self.distribution[group])
                    add_params["tree_learner"] = "data"
                    self.config['base_port'] += 1

                def lightGBMLoggingCallback(env):
                    _, _, score, _ = env.evaluation_result_list[0]
                    all_result[env.iteration][group][config]['val_loss'] = score
                    _, _, score, _ = env.evaluation_result_list[1]
                    all_result[env.iteration][group][config]['val_accuracy'] = score

                # The preds is already prob
                def logloss_eval(preds, data):
                    y_true = data.get_label()

                    return [("val_accuracy", ((preds > 0.5) == y_true).sum() / self.num_val_examples[group], True)]

                model = lgb.train({**self.model_config.get_train_params(group),
                                   **self.params[group][config],
                                   **add_params},
                                  self.train_data[group].to_lightgbm(),
                                  valid_sets=self.val_data[group].to_lightgbm(),
                                  num_boost_round=num_epochs,
                                  feval=logloss_eval,
                                  callbacks=[lightGBMLoggingCallback])

        return all_result


@ray.remote(num_cpus=1)
def cal_grad_hess(model, criterion, inputs, targets):
    outputs = torch.from_numpy(model.predict(inputs.numpy()))
    outputs = torch.sigmoid(outputs)
    loss = criterion(outputs, targets.double())
    grad = outputs - targets
    hess = outputs * (1. - outputs)
    return loss.item(), grad, hess


class DPOperator(BaseOperator):
    def setup(self, config, init=True):
        if init:
            return
        super(DPOperator, self).setup(config)
        init_logging(config["log_path"])

        if not hasattr(self, "train_data") or not hasattr(self, "val_data"):
            # Setup data loaders only once
            safe_log_in_dict(location=dist.get_rank(), epoch="", group=config["group_name"], timing="start",
                             action="load data", params="", result="")
            self.train_data = ParallelDataset(config["train_files"][dist.get_rank()], **config["data_params"])
            self.val_data = ParallelDataset(config["val_files"][dist.get_rank()], **config["data_params"])
            safe_log_in_dict(location=dist.get_rank(), epoch="", group=config["group_name"], timing="finish",
                             action="load data", params="", result="")

        train_loader = DataLoader(self.train_data,
                                  **config["loader_params"])
        val_loader = DataLoader(self.val_data,
                                **config["val_loader_params"])
        self.register_data(train_loader=train_loader, validation_loader=val_loader)
        # Setup model.
        model = config["model"](**config["model_params"])

        # Setup optimizer.
        optimizer = config["optim"](model, **{**config["optim_params"], **{"lr": config["lr"]}})

        # Setup loss.
        criterion = config["criterion"](reduction="sum")

        self.model, self.optimizer, self.criterion = \
            self.register(models=model, optimizers=optimizer,
                          criterion=criterion)
        self.num_examples = config["num_examples"]
        self.l1 = config["l1"] if "l1" in config else None
        self.workload = config["workload"]
        # self.init_metrics()
        # self._timestamp_path = config["timestamp_path"]
        self.num_epochs = 0
        self._rank = dist.get_rank()
        self._config = config

    def train_epoch(self, iterator, info):
        self.num_epochs += 1
        if self.workload == Workload.ALGEBRAIC:
            safe_log_in_dict(location=self._rank, epoch=self.num_epochs, group=self.config["group_name"],
                             timing="start", action="train", params={"lr": self.config["lr"], "l1": self.l1},
                             result="")
            self.model.train()
            local_loss = 0.0
            count = 0
            self.optimizer.zero_grad()
            for batch_idx, (inputs, targets) in enumerate(iterator):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                local_loss += loss.item()
                count += inputs.size()[0]

            if self.l1 is not None:
                for param in self.model.parameters():
                    local_loss += self.l1 * torch.norm(param, 1)
                local_loss = local_loss.item()

            loss = self.collective_avg_grad(self.model, local_loss) / self.num_examples

            def closure():
                return loss.item()
            self.optimizer.step(closure)

            safe_log_in_dict(location=self._rank, epoch=self.num_epochs, group=self.config["group_name"],
                             timing="finish", action="train", params={"lr": self.config["lr"], "l1": self.l1},
                             result={"train_loss": loss.item()})
            return {
                "train_loss": loss.item(),
            }
        elif self.workload == Workload.SEQUENTIAL:
            safe_log_in_dict(location=self._rank, epoch=self.num_epochs, group=self.config["group_name"],
                             timing="start", action="train", params={"lr": self.config["lr"], "l2": self._config["l2"]},
                             result="")
            local_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(iterator):
                self.optimizer.zero_grad()
                batch_examples = np.prod(list(targets.size()[1:]))
                if self._use_gpu:
                    inputs = inputs.to("cuda")
                    targets = targets.to("cuda")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                local_loss += loss.item() / batch_examples

            local_loss = local_loss / self.num_examples
            loss = self.collective_loss(local_loss, self._use_gpu).item()
            safe_log_in_dict(location=self._rank, epoch=self.num_epochs, group=self.config["group_name"],
                             timing="finish", action="train", params={"lr": self.config["lr"], "l2": self.config["l2"]},
                             result={"train_loss": loss})
            return {
                "train_loss": loss,
            }

    def validate(self, val_iterator, info):
        safe_log_in_dict(location=self._rank, epoch=self.num_epochs, group=self.config["group_name"],
                         timing="start", action="validate", params={"lr": self.config["lr"], "l1": self.l1},
                         result="")
        self.model.eval()
        val_loss = 0.0
        correct_count = 0.0
        total_count = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_iterator):
                batch_examples = np.prod(list(targets.size()[1:]))
                if self._use_gpu:
                    inputs = inputs.to("cuda")
                    targets = targets.to("cuda")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() / batch_examples
                if len(targets) > 2:
                    outputs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, 1)
                correct_count += (preds == targets).double().sum().item() / batch_examples
                total_count += len(targets)


        loss = self.collective_loss(val_loss, self._use_gpu).item()
        c_count = self.collective_loss(correct_count, self._use_gpu).item()
        t_count = self.collective_loss(total_count, self._use_gpu).item()

        if self.num_epochs == 10:
            del self._train_loader
            del self._validation_loader
        safe_log_in_dict(location=self._rank, epoch=self.num_epochs, group=self.config["group_name"],
                         timing="finish", action="train", params={"lr": self.config["lr"], "l1": self.l1},
                         result={"val_loss": loss / t_count,"val_accuracy": c_count / t_count})
        return {
            "val_loss": loss / t_count,
            "val_accuracy": c_count / t_count
        }


@ray.remote(num_cpus=1)
def get_grad(model, criterion, inp, tar):
    batch_examples = np.prod(list(tar.size()[1:]))
    outputs = model(inp)
    loss = criterion(outputs, tar)
    loss.backward()
    return model, loss.item() / batch_examples

global_data = None


@ray.remote(num_cpus=1)
def _run_ddp_local(rank, world_size, model, criterion, loader_params):
    local_loss = 0.0
    loader = DataLoader(global_data, **loader_params,
                        sampler=DataParallelSampler(global_data, num_replicas=world_size, rank=rank))
    for batch_idx, (inputs, targets) in enumerate(loader):
        batch_examples = np.prod(list(targets.size()[1:]))
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        local_loss += loss.item() / batch_examples

    return model, local_loss


class TPOperator(BaseOperator):
    def setup(self, config, init=True):
        # if init:
        #     torch.set_num_interop_threads(config["num_cpus"])
        torch.set_num_threads(config["num_cpus"])
        init_logging(config["log_path"])

        group = config["group_name"]
        config["train_files"] = config["metadata"]["groups"][group]["train_files"]
        config["val_files"] = config["metadata"]["groups"][group]["val_files"]
        config["num_examples"] = config["metadata"]["groups"][group]["total_examples"]

        self.group_name = group
        self.l1 = config["l1"] if "l1" in config else 0
        self.lr = config["lr"] if "lr" in config else 1
        self._timestamp_path = config["timestamp_path"]
        self.num_examples = config["num_examples"]
        self.epoch_count = 0
        self.workload = config["workload"]
        self.remote = config.get("remote")

        if "train_dataset" in config:
            self.train_dataset = config["train_dataset"]
        if "val_dataset" in config:
            self.val_dataset = config["val_dataset"]

        if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
            safe_log_in_dict(location=socket.gethostname(), epoch="", group=self.group_name, timing="start",
                             action="load data", params={"lr": self.lr, "l1": self.l1}, result="")

            if "remote" in config["data_params"]:
                config["data_params"]["placement"] = \
                    placement_group([{socket.gethostname(): config["data_params"]["num_jobs"],
                                      "CPU": config["data_params"]["num_jobs"]}], strategy="STRICT_PACK")
            self.train_dataset = ParallelDataset(config["train_files"], **{**config["data_params"], **{"train": True}})
            self.val_dataset = ParallelDataset(config["val_files"], **config["data_params"])

            if "placement" in config["data_params"]:
                remove_placement_group(config["data_params"]["placement"])

            safe_log_in_dict(location=socket.gethostname(), epoch="", group=self.group_name, timing="finish",
                             action="load data", params={"lr": self.lr, "l1": self.l1}, result="")

        self.train_loader = DataLoader(self.train_dataset, **config["loader_params"])
        val_loader_params = config["val_loader_params"] \
            if config["val_loader_params"] is not None \
            else config["loader_params"]
        self.val_loader = DataLoader(self.val_dataset, **val_loader_params)

        # Setup model.
        model = config["model"](**config["model_params"])
        # Setup optimizer.
        optimizer = config["optim"](model, **{**config["optim_params"], **{"lr": self.lr}})
        # Setup loss.
        criterion = config["criterion"](reduction="sum")


        # self.model, self.optimizer, self.criterion = model, optimizer, criterion
        self.model, self.optimizer, self.criterion = \
            self.register(models=model, optimizers=optimizer,
                          criterion=criterion)
        self.register_data(train_loader=self.train_loader, validation_loader=self.val_loader)
        self._config = config
        self.copies = dict()
        if self._use_gpu:
            self.model = self.model.cuda()

    def train_epoch(self, iterator, info):
        self.epoch_count += 1

        self.model.train()
        local_loss = 0.0
        if self.workload == Workload.ALGEBRAIC:
            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="start", action="train", params={"lr": self.config["lr"], "l1": self.l1}, result="")
            self.optimizer.zero_grad()
            if self.remote:
                refs = []
                for rank in range(self.config["num_cpus"]):
                    refs.append(_run_ddp_local.options(num_cpus=1, resources={socket.gethostname(): 1})
                                .remote(rank, self.config["num_cpus"], self.model,
                                        self.criterion, self.config["loader_params"]))
                map_result = ray.get(refs)
                for partial_model, loss in map_result:
                    local_loss += loss
                    _merge_model(self.model, partial_model)
            else:
                for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    batch_examples = np.prod(list(targets.size()[1:]))
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    local_loss += loss.item() / batch_examples

            if self.l1 is not None:
                for param in self.model.parameters():
                    local_loss += self.l1 * torch.norm(param, 1)
                local_loss = local_loss.item()

            local_loss = local_loss / self.num_examples

            def closure():
                return local_loss
            self.optimizer.step(closure)

            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="finish", action="train", params={"lr": self.config["lr"], "l1": self.l1},
                             result={"train_loss": local_loss})
        elif self.workload == Workload.SEQUENTIAL:
            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="start", action="train", params={"lr": self.config["lr"], "l2": self.config["l2"]},
                             result="")
            for batch_idx, (inputs, targets) in enumerate(iterator):
                self.optimizer.zero_grad()
                batch_examples = np.prod(list(targets.size()[1:]))
                if self._use_gpu:
                    # inputs = torch.autograd.Variable(inputs).cuda()
                    # targets = torch.autograd.Variable(targets).cuda()
                    inputs = inputs.to("cuda")
                    targets = targets.to("cuda")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                local_loss += loss.item() / batch_examples
                if self._use_gpu:
                    del inputs
                    del targets
                    torch.cuda.empty_cache()

            local_loss = local_loss / self.num_examples

            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="finish", action="train", params={"lr": self.config["lr"], "l2": self.config["l2"]},
                             result={"train_loss": local_loss})
        return {
            "train_loss": local_loss
        }

    def validate(self, val_iterator, info):
        if self.workload == Workload.ALGEBRAIC:
            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="start", action="validate", params={"lr": self.config["lr"], "l1": self.l1},
                             result="")
        else:
            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="start", action="validate",
                             params={"lr": self.config["lr"], "l1": self.config["l2"]},
                             result="")
        self.model.eval()
        val_loss = 0.0
        correct_count = 0
        total_count = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                batch_examples = np.prod(list(targets.size()[1:]))
                if self._use_gpu:
                    # inputs = torch.autograd.Variable(inputs).cuda()
                    # targets = torch.autograd.Variable(targets).cuda()
                    inputs = inputs.to("cuda")
                    targets = targets.to("cuda")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() / batch_examples
                if len(targets) > 2:
                    outputs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, 1)
                correct_count += (preds == targets).double().sum().item() / batch_examples
                total_count += len(targets)
                if self._use_gpu:
                    del inputs
                    del targets
                    torch.cuda.empty_cache()

        if self.workload == Workload.ALGEBRAIC:
            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="finish", action="validate", params={"lr": self.config["lr"], "l1": self.l1},
                             result={"val_loss": val_loss / total_count, "val_accuracy": correct_count / total_count})
        else:
            safe_log_in_dict(location=socket.gethostname(), epoch=self.epoch_count, group=self.group_name,
                             timing="finish", action="validate",
                             params={"lr": self.config["lr"], "l1": self.config["l2"]},
                             result={"val_loss": val_loss / total_count, "val_accuracy": correct_count / total_count})

        return {
            "val_loss": val_loss / total_count,
            "val_accuracy": correct_count / total_count
        }


class GBMOperator(BaseOperator):
    def setup(self, config, init=True):
        if init and not config.get("task_parallel"):
            return
        if config.get("task_parallel"):
            group = config["group_name"]
            config["train_files"] = config["metadata"]["groups"][group]["train_files"]
            config["val_files"] = config["metadata"]["groups"][group]["val_files"]
            config["num_examples"] = config["metadata"]["groups"][group]["total_examples"]
        torch.set_num_threads(config["num_cpus"])
        init_logging(config["log_path"])
        self.group_name = config["group_name"]
        self.data_parallel = config.get("data_parallel")

        if not hasattr(self, "train_data") or not hasattr(self, "val_data"):
            if self.data_parallel:
                safe_log_in_dict(location=dist.get_rank(), epoch="", group=self.group_name,
                                 timing="start", action="load data", params="", result="")
                train_dataset = ParallelDataset(config["train_files"][dist.get_rank()], **config["data_params"])
                val_dataset = ParallelDataset(config["val_files"][dist.get_rank()], **config["data_params"])
                safe_log_in_dict(location=dist.get_rank(), epoch="", group=self.group_name,
                                 timing="finish", action="load data", params="", result="")
            else:
                safe_log_in_dict(location=socket.gethostname(), epoch="", group=self.group_name,
                                 timing="start", action="load data", params="", result="")
                train_dataset = ParallelDataset(config["train_files"], **config["data_params"])
                val_dataset = ParallelDataset(config["val_files"], **config["data_params"])
                safe_log_in_dict(location=socket.gethostname(), epoch="", group=self.group_name,
                                 timing="finish", action="load data", params="", result="")

            self.train_data = train_dataset.to_lightgbm()
            self.val_data = val_dataset.to_lightgbm()

        self.train_params = dict()
        self.train_params["learning_rate"] = config["learning_rate"]
        self.train_params["num_leaves"] = config["num_leaves"]

        # Setup model.
        self.num_examples = config["num_examples"]
        # dummy model
        self.model = config["model"]()
        self.model_params = config["model_params"]
        self.num_epochs = config["num_epochs"]
        self.criterion = config["criterion"](reduction="sum")

        self.register_data(
            train_loader=DataLoader(DummyDataset(10)), validation_loader=DataLoader(DummyDataset(10))
        )
        self._config = config

    def train_epoch(self, iterator, info):
        return self.train(info)

    def train(self, info=None):
        if self.data_parallel:
            safe_log_in_dict(location=dist.get_rank(), epoch="", group=self.group_name,
                             timing="start", action="train",
                             params={"num_leaves": self.train_params["num_leaves"],
                                     "learning_rate": self.train_params["learning_rate"]},
                             result="")
        else:
            safe_log_in_dict(location=socket.gethostname(), epoch="", group=self.group_name,
                             timing="start", action="train",
                             params={"num_leaves": self.train_params["num_leaves"],
                                     "learning_rate": self.train_params["learning_rate"]},
                             result="")

        local_params = self.model_params["params"].copy()
        if self.data_parallel:
            local_params["local_listen_port"] = info["port"]
            local_params["machines"] = ",".join([ip + ":" +
                                                 str(info["port"]) for ip in self.model_params["params"]["machines"]])
        local_loss = [0.0]
        local_correct = [0.0]
        local_total = [0.0]

        # The preds is already prob
        def logloss_eval(preds, data):
            y_true = data.get_label()

            local_correct[0] = ((preds > 0.5) == y_true).sum()
            local_total[0] = len(y_true)
            return [("val_accuracy", local_correct[0] / local_total[0], True)]

        self.result = {}

        def lightGBMLoggingCallback(env):
            if len(env.evaluation_result_list) > 1:
                local_loss[0] = env.evaluation_result_list[0][2] * local_total[0]
            if self.data_parallel:
                loss = self.collective_loss(float(local_loss[0])).item()
                correct_count = self.collective_loss(float(local_correct[0])).item()
                total_count = self.collective_loss(float(local_total[0])).item()

                val_loss = loss / total_count
                val_accuracy = correct_count / total_count
                self.result = {"val_loss": val_loss, "val_accuracy": val_accuracy}

                safe_log_in_dict(location=dist.get_rank(), epoch=env.iteration, group=self.group_name,
                                 timing="finish", action="train",
                                 params={"num_leaves": self.train_params["num_leaves"],
                                         "learning_rate": self.train_params["learning_rate"]},
                                 result=self.result)
            else:
                val_loss = local_loss[0] / local_total[0]
                val_accuracy = local_correct[0] / local_total[0]
                self.result = {"val_loss": val_loss, "val_accuracy": val_accuracy}

                safe_log_in_dict(location=socket.gethostname(), epoch=env.iteration, group=self.group_name,
                                 timing="finish", action="train",
                                 params={"num_leaves": self.train_params["num_leaves"],
                                         "learning_rate": self.train_params["learning_rate"]},
                                 result=self.result)
        self.model = lgb.train({**self.train_params,  **local_params},
                               self.train_data,
                               valid_sets=self.val_data,
                               num_boost_round=self.num_epochs,
                               feval=logloss_eval,
                               callbacks=[lightGBMLoggingCallback])

        return self.result
