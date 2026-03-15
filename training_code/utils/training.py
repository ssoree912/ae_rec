'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
from .losses import SupConLoss
from networks import create_architecture, count_parameters
import matplotlib.pyplot as plt
import pprint
import random
import torch.nn.init as init





def add_training_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for the training

    parser.add_argument(
        "--arch", type=str, default="res50nodown",
        help="architecture name"
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="./checkpoints/",
        type=str,
        help="Path to the dataset to use during training",
    )
    parser.add_argument("--pretrain", type=str, default=None, help="pretrained weights")
    parser.add_argument(
        "--optim", type=str, default="adam", help="optim to use [sgd, adam]"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight decay"
    )
    parser.add_argument("--ckpt", type=str, default=None, help="path to load some custom weights")
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="momentum term of adam"
    )
    parser.add_argument(
    "--eps_adv", type=float, nargs='+', default=None, help="epsilon values for FGSM-based adversarial training"
    )
    parser.add_argument(
                "--lambda_bce", type=float, default=None, help="Weight of bce loss when performing contrastive training"
                    )
    parser.add_argument(
        "--proj_ratio", type=int, default=None, help="Factor to scale down the 2048 dimensional space"
    )
    parser.add_argument('--start_fresh', action='store_true', help='Setting this true makes the training start from random weights, not pretrained')
    parser.add_argument('--use_leaky', action='store_true', help='Use leaky ReLU to potentially avoid dying relu')
    parser.add_argument('--flex_rz', action='store_true', help='Use random resized crop to all kinds of resolution from 128-1024')
    parser.add_argument('--only_coco', action='store_true', help='Use only the coco dataset')
    parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning on the penultimate layer')
    parser.add_argument('--use_proj', action='store_true', help='Use a projection layer, before contrastive training')
    parser.add_argument('--use_inversions', action='store_true', help='Use Inversions to train')
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="run on CPU",
    )

    parser.add_argument(
        "--continue_epoch",
        type=str,
        default=None,
        help="Whether the network is going to be trained",
    )

    # ICML paper settings
    parser.add_argument('--fix_backbone', action='store_true', help='Perform linear probing while freezing the rest of the model')
    parser.add_argument('--stay_positive', type=str, default=None, help='Whether to rely on sigmoiding the weights or clamping them')
    parser.add_argument(
        "--unfreeze_last_k", type=int, default=0, help="How many unfrozen blocks in case of fix backbone"
    )
    parser.add_argument(
                "--final_dropout", type=float, default=0.5, help="Dropout in the final layer"
                    )
    return parser

class TrainingModel(torch.nn.Module):

    def __init__(self, opt, subdir='.'):
        super(TrainingModel, self).__init__()

        self.opt = opt
        self.total_steps = 0
        self.resume_epoch = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, subdir)
        self.device = torch.device('cpu') if opt.no_cuda else torch.device('cuda:0')
        
        #Granting functionality to start with random init instead of pretrained resnet50 weights
        self.model = create_architecture(opt.arch, pretrained=not opt.start_fresh,  num_classes=1,leaky=opt.use_leaky,ckpt=opt.ckpt, use_proj=self.opt.use_proj, proj_ratio=opt.proj_ratio,dropout=opt.final_dropout)

        num_parameters = count_parameters(self.model)
        print(f"Arch: {opt.arch} with #trainable {num_parameters}")

        print('lr:', opt.lr)
        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)

        if self.opt.fix_backbone:
            self.freeze_backbone(unfreeze_last_k=self.opt.unfreeze_last_k)

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        if opt.pretrain:
            self.model.load_state_dict(
                torch.load(opt.pretrain, map_location="cpu")["model"]
            )
            print("opt.pretrain ", opt.pretrain)
        if opt.continue_epoch is not None:
            self.load_networks(opt.continue_epoch)
        self.model.to(self.device)

    def freeze_backbone(self, unfreeze_last_k: int = 0):
        """
        Freezes all backbone layers except for the last `k` blocks and the final fully connected (fc) layer.
        Additionally zero-initializes the weights and biases for layers not frozen.

        Args:
            unfreeze_last_k (int): Number of blocks from the end to keep unfrozen.
        """
        # Get all blocks in the model (assuming blocks are named layer1, layer2, etc.)
        backbone_blocks = [name for name, _ in self.model.named_children() if name.startswith("layer")]
        
        # Determine the blocks to freeze and unfreeze
        blocks_to_unfreeze = backbone_blocks[-unfreeze_last_k:] if unfreeze_last_k > 0 else []
        for name, param in self.model.named_parameters():
            # Check if the parameter belongs to a backbone block
            block_name = name.split('.')[0]
            if block_name in blocks_to_unfreeze or 'fc' in name:  # Unfreeze
                param.requires_grad = True
                if 'fc' in name:
                    param.data.zero_()
            else:  
                param.requires_grad = False
                module = dict(self.model.named_modules())[block_name]
                module.eval()

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train_on_batch(self, data):
        self.total_steps += 1
        self.model.train()
        #grads = {name: [] for name, param in self.model.named_parameters() if param.requires_grad}
        if self.opt.batched_syncing:
            rdata = data[0]
            fdata = data[1]
            input = torch.cat((rdata['img'], fdata['img']), dim=0).to(self.device)
            label = torch.cat((rdata['target'], fdata['target']), dim=0).to(self.device).float()
        else:
            input = data['img'].to(self.device)
            label = data['target'].to(self.device).float()
        output, feats = self.model(input, return_feats=self.opt.use_contrastive)


        if len(output.shape) == 4:
            ss = output.shape
            loss = self.loss_fn(
                output,
                label[:, None, None, None].repeat(
                (1, int(ss[1]), int(ss[2]), int(ss[3]))
                ),
            )
        else:
            loss = self.loss_fn(output.squeeze(1), label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Stay-Positive Update (ICML)
        if self.opt.stay_positive == 'clamp':
            with torch.no_grad():
                self.model.fc.weight.data.clamp_(min=0)
        return loss.cpu()

    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epoch': epoch,
        }

        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, epoch):
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)

        try:
            self.total_steps = state_dict['total_steps']
        except:
            self.total_steps = 0

        try:
            self.resume_epoch = int(state_dict['epoch'])
        except:
            try:
                self.resume_epoch = int(epoch)
            except:
                self.resume_epoch = 0

        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        except:
            pass


    def predict(self, data_loader):
        model = self.model.eval()
        with torch.no_grad():
            y_true, y_pred, y_path = [], [], []
            for data in tqdm.tqdm(data_loader):
                img = data['img']
                label = data['target'].cpu().numpy()
                paths = list(data['path'])
                out_tens,_ = model(img.to(self.device))
                out_tens = out_tens.cpu().numpy()[:, -1]
                assert label.shape == out_tens.shape

                y_pred.extend(out_tens.tolist())
                y_true.extend(label.tolist())
                y_path.extend(paths)

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred, y_path
