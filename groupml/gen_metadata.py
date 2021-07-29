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


import random
import pandas as pd
import os
import subprocess
import glob
import json
from hdfs import InsecureClient
from concurrent.futures import ThreadPoolExecutor


def line_count(filename, hdfs_used=False, client=None):
    if hdfs_used:
        with client.read(filename, encoding='utf-8') as reader:
            num_lines = sum(1 for _ in reader)
            return num_lines
    else:
        return int(subprocess.check_output(['wc', '-l', filename]).split()[0])


def check_hdfs_path(path):
    return path.startswith("hdfs")


class Metadata(object):
    def __init__(self, num_groups, data_type, dimension):
        self.num_groups = num_groups
        self.data_type = data_type
        self.dimension = dimension
        self.groups = dict()

    def add_group(self, name, path, val_frac):
        print("group", name)
        self.groups[name] = dict()
        all_partitions = []
        hdfs_used = check_hdfs_path(path)
        client = None
        if hdfs_used:
            client = InsecureClient('http://namenode:9870')
            path = path[len("hdfs://"):]

        def fetch_info(filename):
            if filename.endswith(".png"):
                file = path + filename
                return {
                    "file_path": file,
                    "num_examples": 1
                }
        if hdfs_used:
            files = client.list(path)
            executor = ThreadPoolExecutor(max_workers=160)
            all_partitions += [i for i in executor.map(fetch_info, files) if i is not None]
        else:
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if filename.endswith(".csv"):
                        file = path + filename
                        all_partitions.append({
                            "file_path": file,
                            "num_examples": line_count(file)
                        })
                    if filename.endswith(".png"):
                        file = path + filename
                        all_partitions.append({
                            "file_path": file,
                            "num_examples": 1
                        })

        num_files = len(all_partitions)
        if val_frac * num_files < 1:
            df = pd.concat([pd.read_csv(f, header=None) for f in glob.glob(path + "*.csv")], ignore_index=True)
            num_examples = df.shape[0]
            val_examples = int(num_examples * val_frac)
            val = df[:(val_examples if val_examples != 0 else 1)]
            train = df[val_examples:]
            for f in glob.glob(path + "*"):
                os.remove(f)
            train.to_csv(path + "train.csv", index=False)
            val.to_csv(path + "val.csv", index=False)
            self.groups[name]["train_files"] = [{
                "file_path": path + "train.csv",
                "num_examples":  train.shape[0]
            }]
            self.groups[name]["val_files"] = [{
                "file_path": path + "val.csv",
                "num_examples": val.shape[0]
            }]
            self.groups[name]["total_examples"] = train.shape[0]
        else:
            num_val_files = int(val_frac * num_files)
            self.groups[name]["train_files"] = all_partitions[:num_files - num_val_files]
            self.groups[name]["val_files"] = all_partitions[num_files - num_val_files:]
            self.groups[name]["total_examples"] = sum([p["num_examples"] for p in self.groups[name]["train_files"]])

    def to_json(self):
        return {
            "num_groups": self.num_groups,
            "type": self.data_type,
            "dimension": self.dimension,
            "groups": self.groups
        }


# metadata = Metadata(47, "image", 512*512)
# for root, dirs, files in os.walk("images/"):
#     for group in dirs:
#         print(group)
#         metadata.add_group(group, "images/" + group + "/", 0.1)
#
# with open('new_metadata.json', 'w') as fp:
#     json.dump(metadata.to_json(), fp, indent=4)