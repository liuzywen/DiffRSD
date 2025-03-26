# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from datetime import datetime
from monai.data import DataLoader
import argparse
from monai.utils import set_determinism
from .sampler import SequentialDistributedSampler, distributed_concat
from torch.utils.tensorboard import SummaryWriter

with open("argument/train_config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)


class Trainer:
    def __init__(self, env_type,
                 max_epochs,
                 batch_size,
                 load_checkpoint,
                 val_start,
                 device="cpu",
                 val_every=1,
                 num_gpus=1,
                 logdir="./logs/",
                 console_log_save_path=None,
                 master_ip='localhost',
                 master_port=17750,
                 training_script="train.py",
                 ):
        self.global_step = 0
        assert env_type in ["pytorch", "ddp", "DDP"], f"not support this env_type: {env_type}"
        self.env_type = env_type
        self.val_start = val_start
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus
        self.device = device
        self.rank = 0
        self.local_rank = 0
        self.batch_size = batch_size
        self.not_call_launch = True
        self.logdir = logdir
        self.scheduler = None
        self.model = None
        self.auto_optim = True
        self.console_log_save_path = console_log_save_path
        self.source_log = r'F:\HR\work\diffNEU\logs\log.txt'
        if config["debug"]:
            self.debug_val = True
        else:
            self.debug_val = False

        torch.backends.cudnn.enabled = True
        gpu_count = torch.cuda.device_count()
        if num_gpus > gpu_count:
            print("gpu数量不符")
            os._exit(0)

    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        if dataset is None:
            return None
        if self.env_type == 'pytorch':
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle)
        else:
            if not train:
                sampler = SequentialDistributedSampler(dataset, batch_size=batch_size)

            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            return DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=10,
                              sampler=sampler,
                              drop_last=False,
                              persistent_workers=True)

    def get_dist_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=0, help="local_rank")
        parser.add_argument('--not_call_launch',
                            action='store_true',
                            help="not call launch!")
        ds_args = parser.parse_args()
        self.rank = int(os.environ.get('RANK', 0))
        # self.local_rank = int(os.environ["LOCAL_RANK"])

        self.local_rank = ds_args.local_rank
        self.not_call_launch = ds_args.not_call_launch
        self.device = self.local_rank

        # self.master_addr = os.environ.get('MASTER_ADDR','127.0.0.1')
        # self.master_port = os.environ.get('MASTER_PORT','17500')

    def validation_single_gpu(self, val_dataset, batch_size,thresh=None):
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.model.to(self.device)
        val_outputs = []
        self.model.eval()
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            if isinstance(batch, dict):
                batch = {
                    x: batch[x].to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
            elif isinstance(batch, list):
                name = batch[2]
                batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                batch.append(name)
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            else:
                print("not support data type")
                exit(0)

            with torch.no_grad():
                val_out = self.validation_step(batch,thresh)
                assert val_out is not None

            return_list = False
            val_outputs.append(val_out)
        if isinstance(val_out, list) or isinstance(val_out, tuple):
            return_list = True

        val_outputs = torch.tensor(val_outputs)
        if not return_list:
            # 说明只有一个变量
            length = 0
            v_sum = 0.0
            for v in val_outputs:
                if not torch.isnan(v):
                    v_sum += v
                    length += 1

            if length == 0:
                v_sum = 0
            else:
                v_sum = v_sum / length
        else:
            num_val = len(val_outputs[0])
            length = [0.0 for i in range(num_val)]
            v_sum = [0.0 for i in range(num_val)]

            for v in val_outputs:
                for i in range(num_val):
                    if not torch.isnan(v[i]):
                        v_sum[i] += v[i]
                        length[i] += 1

            for i in range(num_val):
                if length[i] == 0:
                    v_sum[i] = 0
                else:
                    v_sum[i] = v_sum[i] / length[i]
        return v_sum, val_outputs

    def copy_file(self, source_file, destination_file):
        with open(source_file, 'r') as source:
            with open(destination_file, 'w') as destination:
                # 从源文件读取内容
                content = source.read()
                # 将内容写入目标文件
                destination.write(content)

    def train(self,
              train_dataset,
              optimizer=None,
              model=None,
              val_dataset=None,
              scheduler=None,
              ):

        self.writer = SummaryWriter(self.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        if scheduler is not None:
            self.scheduler = scheduler
        set_determinism(1234 + self.local_rank)

        if self.model is not None:
            # print(f"check model parameter: {next(self.model.parameters()).sum()}")
            self.model.to(self.device)
            para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
            if self.local_rank == 0:
                print(f"model parameters is {para * 4 / 1000 / 1000}M ")

        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        if val_dataset is not None:
            val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
            # 测试batch_size
            # val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
        else:
            val_loader = None

        for epoch in range(0, self.max_epochs):

            self.copy_file(source_file=self.source_log, destination_file=self.console_log_save_path)

            if self.debug_val and epoch == 0:
                # if self.model is not None:
                # self.validation_end(0.001, 40)
                self.model.eval()
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
                    if idx == 7:
                        print("val_test finishi!")
                        # input()
                        break
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list):
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    else:
                        print("not support data type")
                        exit(0)

                    with torch.no_grad():
                        val_out = self.validation_step(batch)

            if self.debug_val and epoch == 20:
                val_outputs = []
                # and epoch > 3
                if self.model is not None:
                    self.model.eval()
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list):
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    else:
                        print("not support data type")
                        exit(0)

                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                        assert val_out is not None

                    return_list = False
                    val_outputs.append(val_out)
                    if isinstance(val_out, list) or isinstance(val_out, tuple):
                        return_list = True

                ## 先汇总结果。
                val_outputs = torch.tensor(val_outputs)

                if self.local_rank == 0:
                    if not return_list:
                        # 说明只有一个变量
                        length = 0
                        v_sum = 0.0
                        for v in val_outputs:
                            if not torch.isnan(v):
                                v_sum += v
                                length += 1

                        if length == 0:
                            v_sum = 0
                        else:
                            v_sum = v_sum / length
                        self.validation_end(mean_val_outputs=v_sum, epoch=epoch)

                    else:
                        num_val = len(val_outputs[0])
                        length = [0.0 for i in range(num_val)]
                        v_sum = [0.0 for i in range(num_val)]

                        for v in val_outputs:
                            for i in range(num_val):
                                if not torch.isnan(v[i]):
                                    v_sum[i] += v[i]
                                    length[i] += 1

                        for i in range(num_val):
                            if length[i] == 0:
                                v_sum[i] = 0
                            else:
                                v_sum[i] = v_sum[i] / length[i]

                        self.validation_end(mean_val_outputs=v_sum, epoch=epoch)

            self.train_epoch(train_loader, epoch, )

            val_outputs = []

            if epoch % self.val_every == 0 and val_loader is not None and epoch != 0 and epoch > self.val_start:
                # and epoch > 3
                if self.model is not None:
                    self.model.eval()
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list):
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    else:
                        print("not support data type")
                        exit(0)

                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                        assert val_out is not None

                    return_list = False
                    val_outputs.append(val_out)
                    if isinstance(val_out, list) or isinstance(val_out, tuple):
                        return_list = True

                ## 先汇总结果。
                val_outputs = torch.tensor(val_outputs)

                if self.local_rank == 0:
                    if not return_list:
                        # 说明只有一个变量
                        length = 0
                        v_sum = 0.0
                        for v in val_outputs:
                            if not torch.isnan(v):
                                v_sum += v
                                length += 1

                        if length == 0:
                            v_sum = 0
                        else:
                            v_sum = v_sum / length
                        self.validation_end(mean_val_outputs=v_sum, epoch=epoch)

                    else:
                        num_val = len(val_outputs[0])
                        length = [0.0 for i in range(num_val)]
                        v_sum = [0.0 for i in range(num_val)]

                        for v in val_outputs:
                            for i in range(num_val):
                                if not torch.isnan(v[i]):
                                    v_sum[i] += v[i]
                                    length[i] += 1

                        for i in range(num_val):
                            if length[i] == 0:
                                v_sum[i] = 0
                            else:
                                v_sum[i] = v_sum[i] / length[i]

                        self.validation_end(mean_val_outputs=v_sum, epoch=epoch)

            if self.scheduler is not None:
                self.scheduler.step()
            if self.model is not None:
                self.model.train()

    def train_epoch(self,
                    loader,
                    epoch,
                    ):
        if self.model is not None:
            self.model.train()
        if self.local_rank == 0:
            with tqdm(total=len(loader), leave=False) as t:

                for idx, batch in enumerate(loader):
                    self.global_step += 1
                    # 获取当前日期和时间
                    now = datetime.now()
                    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    t.set_description(f'{formatted_time} Epoch {epoch}')
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].contiguous().to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list):
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)

                    else:
                        print("not support data type")
                        exit(0)

                    with torch.autograd.set_detect_anomaly(True):
                        if self.model is not None:
                            for param in self.model.parameters(): param.grad = None
                        loss = self.training_step(batch, epoch)

                        if self.auto_optim:
                            loss.backward()
                            self.optimizer.step()

                            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                            t.set_postfix(loss=loss.item(), lr=lr)
                        t.update(1)
        else:
            for idx, batch in enumerate(loader):
                self.global_step += 1
                if isinstance(batch, dict):
                    batch = {
                        x: batch[x].contiguous().to(self.device)
                        for x in batch if isinstance(batch[x], torch.Tensor)
                    }
                elif isinstance(batch, list):
                    batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]

                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)

                else:
                    print("not support data type")
                    exit(0)

                for param in self.model.parameters(): param.grad = None

                loss = self.training_step(batch, epoch)
                if self.auto_optim:
                    loss.backward()
                    self.optimizer.step()

            for param in self.model.parameters(): param.grad = None

    def training_step(self, batch, epoch):
        raise NotImplementedError

    def validation_step(self, batch,thresh=None):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs, epoch):
        pass

    def log(self, k, v, step):
        if self.env_type == "pytorch":
            self.writer.add_scalar(k, scalar_value=v, global_step=step)

        else:
            if self.local_rank == 0:
                self.writer.add_scalar(k, scalar_value=v, global_step=step)

    def load_state_dict(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v

        self.model.load_state_dict(new_sd, strict=strict)

        print(f"model parameters are loaded successed.")
