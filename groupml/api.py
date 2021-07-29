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


import datetime
import ray

from groupml.group_parallel import GroupLearningEstimator
from groupml.configuration import Configuration, Workload
from groupml.operators import MetricsOperator
from groupml.dataset import Storage


def run_group_learning(metadata_path, model_creator, model_params, optim_creator, optim_params, data_params,
                       loader_params, groups, search_space, num_epochs=10, num_workers=8, num_cpus_per_worker=4,
                       storage=Storage.FS, workload=Workload.ALGEBRAIC, log_path="/home/ubuntu/groupml/gl_results/",
                       verbose=False, use_gpu=False, train_params=None, criterion=None, num_nodes=4):
    ray.init(address="auto", log_to_driver=verbose, _temp_dir="/mnt2/tmp/ray", ignore_reinit_error=True)

    # Log to the latest directory
    date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    log_path += date_str + "/"

    # Generate grid search space
    config = Configuration()
    config.register_groups(groups, model_creator, model_params, optim_creator, optim_params, data_params, loader_params,
                           search_space=search_space, train_params=train_params, criterion=criterion)
    config.set_workload_type(workload)

    # Start collecting metrics
    print("Going to set up", num_nodes, "metrics ops")
    metrics_operators = [MetricsOperator.remote(log_path, use_gpu) for _ in range(num_nodes)]
    metric_ops = [op.start_collect_metrics.remote() for op in metrics_operators]
    ray.get(metric_ops)

    # Train the models
    gl_learner = GroupLearningEstimator(metadata_path, num_workers, num_cpus_per_worker, True, config, storage=storage)
    gl_learner.train(num_epochs=num_epochs, mode=workload, log_dir=log_path)

    # Stop collecting metrics
    metric_ops = [op.finish_collect_metrics.remote() for op in metrics_operators]
    ray.get(metric_ops)

