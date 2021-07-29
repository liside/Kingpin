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
from torch.utils.data import Dataset
from filelock import FileLock
import datetime
import json
import signal


logging_path = "/users/liside/groupml/default.log"
lock = FileLock("/tmp/logging.lock")


class RandomDataset(Dataset):
    def __init__(self, length, size):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], torch.rand(1).long()

    def __len__(self):
        return self.len


def init_logging(path):
    global logging_path
    logging_path = path


def safe_log(message):
    with lock:
        with open(logging_path, "a+") as f:
            # f.write(str(datetime.datetime.utcnow()) + ", " + message + "\n")
            f.write(json.dumps({**{"time": str(datetime.datetime.utcnow())},
                                **{message}
                                }) + "\n")


def safe_log_in_dict(epoch="", group="", timing="", action="", params="", result="", location=""):
    with lock:
        with open(logging_path, "a+") as f:
            # f.write(str(datetime.datetime.utcnow()) + ", " + message + "\n")
            msg = json.dumps({
                "time": str(datetime.datetime.utcnow()),
                "location": location,
                "epoch": epoch,
                "group": group,
                "timing": timing,
                "action": action,
                "params": params,
                "result": result
            }) + "\n"
            print(msg)
            f.write(msg)


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

