from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================

import argparse
from collections import OrderedDict
import csv
import math
import os
import threading
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

import h5py
import numpy as np
import runtime
from modeling import BertConfig, CrossEntropyWrapper
from pipe import model

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--bert_config",
                    default="bert_config.json",
                    type=str,
                    help="The BERT model config")
parser.add_argument("--max_predictions_per_seq",
                    default=80,
                    type=int,
                    help="The maximum total of masked tokens in input sequence")

parser.add_argument('--data_dir', type=str, default="data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/bookscorpus/",
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str, default="gloo",
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--max-length-train', default=128, type=int,
                    help='maximum sequence length for training')
parser.add_argument('--min-length-train', default=0, type=int,
                    help='minimum sequence length for training')

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]


class Task:
    def __init__(self, indices, modules_input_names, modules_output_names, runtime):
        self.indices = indices
        self.runtime = runtime
        self._output_names = []
        self.predecessors = {}
        self.successors = {}
        for input_names, output_names in zip(modules_input_names, modules_output_names):
            self.merge_dependencies(input_names, output_names)
            self._output_names.extend(output_names)
        self.threads = {}

    def merge_dependencies(self, input_names, output_names):
        for name in input_names:
            if name not in self._output_names and name not in self.predecessors:
                self.predecessors[name] = None
        
        for name in output_names:
            if name in self.predecessors:
                self.predecessors.pop(name)


class Dependencies:
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        for _task in self.tasks:
            for name in task.predecessors:
                if name in _task._output_names:
                    task.predecessors[name] = _task
                    if name not in _task.successors:
                        _task.successors[name] = []
                    _task.successors[name].append(task)
            for name in _task.predecessors:
                if name in task._output_names:
                    _task.predecessors[name] = task
                    if name not in task.successors:
                        task.successors[name] = []
                    task.successors[name].append(_task)
        self.tasks.append(task)


class taskThread(threading.Thread):
    def __init__(self, task, is_forward, minibatch_id):
        threading.Thread.__init__(self)
        self.task = task
        self.is_forward = is_forward
        self.minibatch_id = minibatch_id

    def run(self):
        r = self.task.runtime
        if self.is_forward:
            print("Forward: %d" % (self.minibatch_id))
            # r._run_forward(self, indices, all_input_names, all_output_names, minibatch_id)
            for tensor_name in self.task.successors:
                for receiver in self.task.successors[tensor_name]:
                    self.copy_tensor(tensor_name, self.task.runtime, receiver.runtime)
        else:
            print("Backward: %d" % (self.minibatch_id))
            # r._run_backward(self, indices, all_input_names, all_output_names, minibatch_id)
    
    def copy_tensor(self, tensor_name, src_runtime, dst_runtime):
        print("Copy Tensor: %s" % (tensor_name))


class Scheduler():
    def __init__(self, task_list):
        self.threads = []
        self.minibatch_id = 0
        self.task_list = task_list

    def _iter(self, staleness):
        self.minibatch_id += 1
        for task in self.task_list:
            forward_thread = taskThread(task, True, self.minibatch_id)
            backward_thread = taskThread(task, False, self.minibatch_id)
            forward_thread.start()
            backward_thread.start()
            self.threads.append(backward_thread)
            task.threads[self.minibatch_id] = backward_thread

    def _join(self):
        for t in self.threads:
            t.join()
        

def split_list(list, n):
    if len(list) % n == 0:
        cnt = len(list) // n
    else:
        cnt = len(list) // n + 1
    for i in range(0, n):
        yield list[i*cnt:(i+1)*cnt]


def main():
    args = parser.parse_args()
    config = BertConfig.from_json_file(args.bert_config)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    criterion = CrossEntropyWrapper(config.vocab_size)

    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if
                     os.path.isfile(os.path.join(args.data_dir, f)) and 'training' in f]
    data_file = files[0]
    train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.batch_size, num_workers=4,
                                  pin_memory=True)
    loader_iter = iter(train_loader)

    bert = model(criterion)
    ranks = 4

    _modules = []
    _all_input_names = []
    _all_output_names = []
    for (module, input_names, output_names) in bert:
        _modules.append(module)
        _all_input_names.append(input_names)
        _all_output_names.append(output_names)

    # Runtime Initialization
    task_list = Dependencies()
    for i, indices in enumerate(split_list(list(range(len(_modules))), ranks)):
        r = runtime.StageRuntime(_modules, i, runtime.BERT)
        r._load_modules(indices)
        task_list.add_task(Task(indices, [_all_input_names[i] for i in indices],
                          [_all_output_names[i] for i in indices], r))
    

    s = Scheduler(task_list.tasks)
    iterations = 10
    for i in range(iterations):
        print(i)
        s._iter(4)
    s._join()


    '''
    input_size = [args.batch_size, args.max_length_train]
    training_tensor_shapes = {"out0": input_size, "out1": input_size,
                              "out27": [args.batch_size, 1, 1, args.max_length_train],
                              "target": input_size,
                              "target_label": [args.batch_size]}
    dtypes = {"out0": torch.int64, "out1": torch.int64, "out27": torch.float32,
              "target": torch.int64, "target_label": torch.int64}
    inputs_module_destinations = {"out0": 0, "out1": 0, "out27": 0}
    target_tensor_names = {"target", "target_label"}
    for module_id, (stage, inputs, outputs) in enumerate(bert[:-1]):  # Skip last layer (loss).
        input_tensors = []
        for module_input in inputs:
            if module_input in inputs_module_destinations:
                inputs_module_destinations[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes[module_input]),
                                      dtype=dtypes[module_input]).cuda()
            input_tensors.append(input_tensor)
        stage.cuda()
        # PyTorch should not maintain metadata for a backward pass on
        # synthetic inputs. Without the following line, the runtime is
        # as much as 1.5x slower in a full DP configuration.
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype
    '''

if __name__ == '__main__':
    main()