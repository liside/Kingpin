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


import math
import json
from collections import defaultdict
from hdfs import InsecureClient
import heapq


def wrap_around_assignment(path, groups, num_workers, num_models=None, metadata=None):
    def optimal_makespan(groups, weights, val=False):
        if val:
            all_sizes = {group: sum([f["num_examples"] for f in metadata["groups"][group]["val_files"]])
                         for group in groups}
        else:
            all_sizes = {group:metadata["groups"][group]["total_examples"] for group in groups}
        for group in weights:
            all_sizes[group] *= weights[group]
        return sum(all_sizes.values()) * 1.0 / num_workers

    if num_models is None:
        num_models = {group: 1 for group in groups}

    if metadata is None:
        if "hdfs" in path:
            client = InsecureClient('http://namenode:9870')
            with client.read(path[len("hdfs://"):], encoding='utf-8') as reader:
                metadata = json.load(reader)
        else:
            f = open(path)
            metadata = json.load(f)

    groups = sorted(groups, reverse=True, key=lambda a: metadata["groups"][a]["total_examples"])
    optimal = math.ceil(optimal_makespan(groups, num_models))

    res = {i: {"type": metadata["type"], "dimension": metadata["dimension"], "groups": []} for i in range(num_workers)}

    cur_filled = 0
    cur_worker = 0
    for name in groups:
        group = metadata["groups"][name]
        weight = num_models[name]
        for partition in group["train_files"]:
            if len(res[cur_worker]["groups"]) == 0 or res[cur_worker]["groups"][-1]["name"] != name:
                res[cur_worker]["groups"].append({
                    "name": name,
                    "total_examples": metadata["groups"][name]["total_examples"],
                    "val_files": [],
                    "train_files": [],
                    "total_val_examples": sum([f["num_examples"] for f in metadata["groups"][name]["val_files"]])
                })
            res[cur_worker]["groups"][-1]["train_files"].append(partition)
            cur_filled += partition["num_examples"] * weight
            if cur_filled > optimal:
                cur_worker += 1
                cur_filled = 0

    # add validation files as well
    assignment = {name: defaultdict(int) for name in groups}

    for worker in res:
        for group in res[worker]["groups"]:
            assignment[group["name"]][worker] = sum([f["num_examples"] for f in group["train_files"]])
    # print(json.dumps(assignment, indent=4))
    for group in assignment:
        size = len(assignment[group])
        num_of_files = len(metadata["groups"][group]["val_files"])
        if num_of_files / size < 1:
            print("Some workers are not going to get validation files")
        #     print("unable to schedule validation files!!!!!!")
        #     return None
        start_index = 0
        end_index = 0
        for worker in assignment[group]:
            share = int(num_of_files * assignment[group][worker] / metadata["groups"][group]["total_examples"])
            if share < 1:
                share = 1
            end_index += share
            for i, g in enumerate(res[worker]["groups"]):
                if group == g["name"]:
                    if start_index < num_of_files:
                        res[worker]["groups"][i]["val_files"] += \
                            metadata["groups"][group]["val_files"][start_index:end_index]
                    break
            start_index = end_index

        if end_index < num_of_files:
            worker = list(assignment[group].keys())[-1]
            for i, g in enumerate(res[worker]["groups"]):
                if group == g["name"]:
                    res[worker]["groups"][i]["val_files"] += metadata["groups"][group]["val_files"][end_index:]
                    break

    return res


def constrained_wrap_around_assignment(path, groups, num_workers, num_models=None, metadata=None):
    def optimal_makespan(groups, weights, val=False):
        if val:
            all_sizes = {group: sum([f["num_examples"] for f in metadata["groups"][group]["val_files"]])
                         for group in groups}
        else:
            all_sizes = {group:metadata["groups"][group]["total_examples"] for group in groups}
        for group in weights:
            all_sizes[group] *= weights[group]
        return sum(all_sizes.values()) * 1.0 / num_workers

    if num_models is None:
        num_models = {group: 1 for group in groups}

    if metadata is None:
        if "hdfs" in path:
            client = InsecureClient('http://namenode:9870')
            with client.read(path[len("hdfs://"):], encoding='utf-8') as reader:
                metadata = json.load(reader)
        else:
            f = open(path)
            metadata = json.load(f)

    groups = sorted(groups, reverse=True, key=lambda a: metadata["groups"][a]["total_examples"])
    optimal = math.ceil(optimal_makespan(groups, num_models))

    res = {i: {"type": metadata["type"], "dimension": metadata["dimension"], "groups": []} for i in range(num_workers)}

    workers = [(0, i) for i in range(num_workers)]

    def get_best_approximate_partitions(num_examples):
        min_diff = optimal * num_workers
        res = 0

        for i in range(1, num_workers + 1):
            local_diff = abs(num_examples / i - optimal)
            if local_diff < min_diff:
                res = i
                min_diff = local_diff

        return res

    distribution = defaultdict(list)
    for name in groups:
        group = metadata["groups"][name]
        weight = num_models[name]
        total_examples = metadata["groups"][name]["total_examples"]
        num_splits = get_best_approximate_partitions(total_examples)
        split_size = int(total_examples / num_splits)
        for split in range(num_splits):
            existing, cur_worker = heapq.heappop(workers)
            distribution[name].append(cur_worker)
            cur_filled = 0
            print("current worker is", cur_worker)
            for partition in group["train_files"]:
                if len(res[cur_worker]["groups"]) == 0 or res[cur_worker]["groups"][-1]["name"] != name:
                    res[cur_worker]["groups"].append({
                        "name": name,
                        "total_examples": metadata["groups"][name]["total_examples"],
                        "val_files": [],
                        "train_files": [],
                        "total_val_examples": sum([f["num_examples"] for f in metadata["groups"][name]["val_files"]])
                    })
                res[cur_worker]["groups"][-1]["train_files"].append(partition)
                cur_filled += partition["num_examples"] * weight
                if cur_filled >= split_size:
                    break
            heapq.heappush(workers, (existing + cur_filled, cur_worker))

    for i in range(num_workers):
        res[i]["distribution"] = distribution
    # add validation files as well
    assignment = {name: defaultdict(int) for name in groups}

    for worker in res:
        for group in res[worker]["groups"]:
            assignment[group["name"]][worker] = sum([f["num_examples"] for f in group["train_files"]])

    for group in assignment:
        size = len(assignment[group])
        num_of_files = len(metadata["groups"][group]["val_files"])
        if num_of_files / size < 1:
            print("Some workers are not going to get validation files")

        start_index = 0
        end_index = 0
        for worker in assignment[group]:
            share = int(num_of_files * assignment[group][worker] / metadata["groups"][group]["total_examples"])
            if share < 1:
                share = 1
            end_index += share
            for i, g in enumerate(res[worker]["groups"]):
                if group == g["name"]:
                    if start_index < num_of_files:
                        res[worker]["groups"][i]["val_files"] += \
                            metadata["groups"][group]["val_files"][start_index:end_index]
                    break
            start_index = end_index

        if end_index < num_of_files:
            worker = list(assignment[group].keys())[-1]
            for i, g in enumerate(res[worker]["groups"]):
                if group == g["name"]:
                    res[worker]["groups"][i]["val_files"] += metadata["groups"][group]["val_files"][end_index:]
                    break

    return res


