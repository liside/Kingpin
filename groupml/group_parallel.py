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
from functools import reduce
import json
import pathlib
import time
import datetime

import ray
from ray.util.sgd import TorchTrainer
from hdfs import InsecureClient

from .partitioning import wrap_around_assignment, constrained_wrap_around_assignment
from .configuration import Workload
from .operators import GLOperator, apply_to_all
from .dataset import Storage


def _get_id_from_indexes(indexes=None):
    if indexes is None or len(indexes) == 0:
        return "-"
    else:
        return "-".join([str(index) for index in indexes])


class GroupLearningEstimator(object):
    def __init__(self, path, num_workers, num_cpus_per_worker, use_gpu, configuration, storage=Storage.HDFS):
        self.path = path
        self.groups = configuration.get_group_names()
        self.num_workers = num_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.use_gpu = use_gpu
        self.configuration = configuration
        self.states = defaultdict(dict)
        self.storage = storage
        self.init_state()

    def init_state(self):
        self.states = defaultdict(dict)
        for group in self.groups:
            search_space = self.configuration.get_search_space(group)
            if search_space is None:
                m_id, op_id = self._init_model(group)
                key = _get_id_from_indexes()
                self.states[group]["models"] = {key: m_id}
                self.states[group]["optimizers"] = {key: op_id}
                self.states[group]["params"] = {key: {}}
                self.states[group]["losses"] = {key: {}}
            else:
                self.states[group]["models"] = dict()
                self.states[group]["optimizers"] = dict()
                self.states[group]["params"] = dict()
                self.states[group]["losses"] = dict()
                self._generate_search_param(group, list(search_space.keys()), [], {})

    def train_epoch(self, trainer, metadata, mode=Workload.ALGEBRAIC):
        apply_to_all(trainer, "start_collect_metrics", {})

        if mode == Workload.ALGEBRAIC:
            print("==============calculating gradients======================")
            success, states = apply_to_all(trainer, "calculate_grad", {})

            print("==============aggregating gradients======================")
            assignment = dict()

            for worker in metadata:
                for group in metadata[worker]["groups"]:
                    group_name = group["name"]
                    if group_name not in assignment:
                        assignment[group_name] = worker
                        # seen.add(group_name)

            new_states = {rank: defaultdict(dict) for rank in range(self.num_workers)}

            for worker, state in enumerate(states):
                for group in state:
                    des_worker = assignment[group]
                    if group not in new_states[des_worker]:
                        new_states[des_worker][group] = {}
                        for param_config in states[worker][group]:
                            new_states[des_worker][group][param_config] = {
                                "grads": [],
                                "losses": []
                            }

                    for param_config in states[worker][group]:
                        new_states[des_worker][group][param_config]["grads"]\
                            .append(states[worker][group][param_config]["grad"])
                        new_states[des_worker][group][param_config]["losses"]\
                            .append(states[worker][group][param_config]["loss"])

            success, states = apply_to_all(trainer, "aggregate_update", dict(states=new_states))

            print("==============Updating Models======================")
            new_states = reduce(lambda a, b: dict(a, **b), states)

            print("==============Collect Metrics======================")
            apply_to_all(trainer, "finish_collect_metrics", {})

            return trainer.train(reduce_results=False, info=new_states)
        elif mode == Workload.SEQUENTIAL:
            configs = {group: list(self.states[group]["models"].keys()) for group in self.states}
            # TODO: 1. think of different number of configs for different groups
            # TODO: 2. pass in the number of search space
            distribution = defaultdict(int)
            for rank in range(self.num_workers):
                for group in metadata[rank]["groups"]:
                    distribution[group["name"]] += 1
            max_spread = max(list(distribution.values()))

            num_configs = len(self.states[list(self.states.keys())[0]]["losses"])
            if max_spread > num_configs:
                raise Exception("Can't train because max spread > num configs")
            print("====================max spread is", max_spread ,"=========================")
            # clear losses in the states
            for group in self.states:
                for config in self.states[group]["losses"]:
                    self.states[group]["losses"][config] = 0.0

            for i in range(max_spread):
                print("==============Round " + str(i) + " with epoch======================")
                runnables = {rank: defaultdict(list) for rank in range(self.num_workers)}
                # TODO: fix
                print("num of configs at a time", int(num_configs / max_spread))
                for j in range(int(num_configs / max_spread)):
                    for rank in range(self.num_workers):
                        for group in metadata[rank]["groups"]:
                            name = group["name"]
                            to_run = configs[name].pop(0)
                            runnables[rank][name].append(to_run)
                            configs[name].append(to_run)

                success, states = apply_to_all(trainer, "train_one_config",
                                               dict(runnables=runnables, states=self.states))

                for state in states:
                    for group in state:
                        for config in state[group]:
                            self.states[group]["models"][config] = state[group][config]["model"]
                            self.states[group]["optimizers"][config] = state[group][config]["optimizer"]
                            self.states[group]["losses"][config] += state[group][config]["loss"]

            # refresh on each worker
            new_states = dict()
            for group in self.states:
                new_states[group] = dict()
                for config in self.states[group]["models"]:
                    new_states[group][config] = {
                        "model": self.states[group]["models"][config],
                        "optimizer": self.states[group]["optimizers"][config],
                        "loss": self.states[group]["losses"][config]
                    }

            print("==============Collect Metrics======================")
            apply_to_all(trainer, "finish_collect_metrics", {})

            return trainer.train(reduce_results=False, info=new_states)
        elif mode == Workload.SAMPLING:
            print("==============calculating grad and hess======================")
            success, states = apply_to_all(trainer, "calculate_grad_hess", {"remote": False})

            print("======================boosting======================")
            assignment = dict()

            for worker in metadata:
                for group in metadata[worker]["groups"]:
                    group_name = group["name"]
                    if group_name not in assignment:
                        assignment[group_name] = worker

            new_states = {rank: defaultdict(dict) for rank in range(self.num_workers)}

            for worker, state in enumerate(states):
                for group in state:
                    des_worker = assignment[group]
                    if group not in new_states[des_worker]:
                        new_states[des_worker][group] = {}
                        for param_config in states[worker][group]:
                            new_states[des_worker][group][param_config] = {
                                "grads": [],
                                "hesses": [],
                                "losses": []
                            }

                    for param_config in states[worker][group]:
                        new_states[des_worker][group][param_config]["grads"]\
                            .append(states[worker][group][param_config]["grad"])
                        new_states[des_worker][group][param_config]["hesses"]\
                            .append(states[worker][group][param_config]["hess"])
                        new_states[des_worker][group][param_config]["losses"]\
                            .append(states[worker][group][param_config]["loss"])

            success, states = apply_to_all(trainer, "boosting", dict(states=new_states))

            print("==============Updating Models======================")
            new_states = reduce(lambda a, b: dict(a, **b), states)

            print("==============Collect Metrics======================")
            apply_to_all(trainer, "finish_collect_metrics", {})

            return trainer.train(reduce_results=False, info=new_states)
        elif mode == Workload.HYBRID:
            print("================get ips====================")
            success, states = apply_to_all(trainer, "get_rank_ip", {})

            print("===================update ips======================")
            success, _ = apply_to_all(trainer, "set_rank_ips", {"ips": reduce(lambda a, b: dict(a, **b), states)})

            print("==================train and validate======================")
            success, new_states = apply_to_all(trainer, "train_hybrid_gbm", {})

            print("==============Collect Metrics======================")
            apply_to_all(trainer, "finish_collect_metrics", {})

            agg_states = defaultdict(dict)
            for state in new_states:
                for group in state:
                    if group not in agg_states:
                        agg_states[group] = defaultdict(dict)
                    for config in state[group]:
                        if config not in agg_states[group]:
                            agg_states[group][config]["model"] = state[group][config]["model"]
                            agg_states[group][config]["loss"] = state[group][config]["loss"]
                            agg_states[group][config]["val_loss"] = state[group][config]["val_loss"]
                            agg_states[group][config]["val_accuracy"] = state[group][config]["val_accuracy"]

                        agg_states[group][config]["loss"] += state[group][config]["loss"]
                        agg_states[group][config]["val_loss"] += state[group][config]["val_loss"]
                        agg_states[group][config]["val_accuracy"] += state[group][config]["val_accuracy"]

            return trainer.train(reduce_results=False, info=agg_states), new_states

    def _train_all_epochs(self, trainer, num_epochs, mode):
        if mode == Workload.HYBRID:
            print("================get ips====================")
            success, states = apply_to_all(trainer, "get_rank_ip", {})

            print("===================update ips======================")
            success, _ = apply_to_all(trainer, "set_rank_ips", {"ips": reduce(lambda a, b: dict(a, **b), states)})

            print("==================train and validate======================")
            success, states = apply_to_all(trainer, "train_hybrid_gbm_all", {"num_epochs": num_epochs})

            result = {i: [state[i] for state in states] for i in range(num_epochs)}

            return result

    def train(self, num_epochs=10, mode=Workload.ALGEBRAIC, log_dir="gl_results/"):
        """
        Train all groups and all configs of hyper-parameters using grid search.
        :param num_epochs:
        :return:
        """
        # data_ops = self._setup_data_ops(self.groups) if mode == Workload.SAMPLING else (None, None)
        data_ops = (None, None)
        original_metadata, dataloaders = data_ops

        if mode == Workload.HYBRID:
            metadata = constrained_wrap_around_assignment(self.path, self.groups,
                                                          self.num_workers, metadata=original_metadata)
            self._populate_models(metadata)
            trainer = self._get_ray_trainer(metadata, log_dir)

            val_result = self._train_all_epochs(trainer, num_epochs, mode)
            for i in range(num_epochs):
                pathlib.Path(log_dir + str(i)).mkdir(parents=True, exist_ok=True)
                with open(log_dir + str(i) + "/val.json", "w") as outfile:
                    json.dump(self._agg_val_stats(val_result[i]), outfile, indent=4)

        else:
            metadata = wrap_around_assignment(self.path, self.groups, self.num_workers, metadata=original_metadata)
            self._populate_models(metadata)
            trainer = self._get_ray_trainer(metadata, log_dir)
            for i in range(num_epochs):
                print("-------------epoch", i, "------------------")
                pathlib.Path(log_dir + str(i)).mkdir(parents=True, exist_ok=True)
                stats = self.train_epoch(trainer, metadata, mode)
                with open(log_dir + str(i) + "/train.json", "w") as outfile:
                    json.dump(stats, outfile, indent=4)
                stats = self.validate(trainer, metadata)
                with open(log_dir + str(i) + "/val.json", "w") as outfile:
                    json.dump(stats, outfile, indent=4)
        trainer.shutdown(force=True)
        del dataloaders

    def validate(self, trainer, metadata):
        if self.states is None:
            print("Models have not been initialized!!!")

        assignment = defaultdict(set)
        seen = set()
        for worker in metadata:
            for group in metadata[worker]["groups"]:
                group_name = group["name"]
                if group_name not in seen:
                    assignment[worker].add(group_name)
                    seen.add(group_name)

        val_stats = trainer.validate(reduce_results=False, info=assignment)
        agg_stats = self._agg_val_stats(val_stats)
        return agg_stats

    def _agg_val_stats(self, val_stats):
        agg_stats = defaultdict(dict)
        for stat in val_stats:
            for group in stat:
                for config in stat[group]:
                    if config not in agg_stats[group]:
                        agg_stats[group][config] = {
                            "val_loss": 0.0,
                            "val_accuracy": 0.0
                        }
                    agg_stats[group][config]["val_loss"] += stat[group][config]["val_loss"]
                    agg_stats[group][config]["val_accuracy"] += stat[group][config]["val_accuracy"]
        return agg_stats

    def tune(self, search_alg, num_epochs=10):
        """
        Train all groups and all configs of hyper-parameters

        :param search_alg:
        :param num_epochs:
        :return:
        """
        # TODO: dynamically rerun wrap around assignment if necessary after each epoch
        return

    def _check_idle(self):
        # TODO Implement reassign logic
        return False

    def _get_ray_trainer(self, metadata, log_path):
        return TorchTrainer(
            training_operator_cls=GLOperator,
            num_workers=self.num_workers,
            num_cpus_per_worker=self.num_cpus_per_worker,
            use_gpu=self.use_gpu,
            config={
                "num_workers": self.num_workers,
                "metadata": metadata,
                "model_config": self.configuration,
                "add_dist_sampler": False,
                "num_cpus": self.num_cpus_per_worker,
                "base_port": 18888,
                "log_path": log_path + "timestamp.log"
            },
            backend="gloo",
            wrap_ddp=False,
            add_dist_sampler=False
        )

    def _populate_models(self, metadata):
        for worker in metadata:
            metadata[worker]["models"] = {}
            metadata[worker]["optimizers"] = {}
            metadata[worker]["params"] = {}
            for group in metadata[worker]["groups"]:
                name = group["name"]
                metadata[worker]["models"][name] = self.states[name]["models"]
                metadata[worker]["optimizers"][name] = self.states[name]["optimizers"]
                metadata[worker]["params"][name] = self.states[name]["params"]

    def _generate_search_param(self, group, keys, indexes, params):
        key = keys[0]
        search_space = self.configuration.get_search_space(group)

        if len(keys) == 1:
            for i, val in enumerate(search_space[key]):
                new_params = {**params, **{key: val}}
                id = _get_id_from_indexes(indexes + [key + "=" + str(val)])
                m_id, op_id = self._init_model(group, new_params)
                self.states[group]["models"][id] = m_id
                self.states[group]["optimizers"][id] = op_id
                self.states[group]["params"][id] = new_params
                self.states[group]["losses"][id] = 0.0
        else:
            for i, val in enumerate(search_space[key]):
                self._generate_search_param(group, keys[1:], indexes + [key + "=" + str(val)], {**params, **{key: val}})

    def _init_model(self, group, params=None):
        if params is None:
            model = self.configuration.get_new_model(group)
            optimizer = self.configuration.get_new_optimizer(group, model)
        else:
            model_param_names = set(self.configuration.get_model(group).__code__.co_varnames)
            optim_param_names = set(self.configuration.get_optimizer(group).__code__.co_varnames)
            model_params = {}
            optim_params = {}
            for name in params:
                if name in model_param_names:
                    model_params[name] = params[name]
                if name in optim_param_names:
                    optim_params[name] = params[name]

            model = self.configuration.get_new_model(group, model_params)
            optimizer = self.configuration.get_new_optimizer(group, model, optim_params)
        m_id, op_id = ray.put(model), ray.put(optimizer)
        del model
        del optimizer

        return m_id, op_id