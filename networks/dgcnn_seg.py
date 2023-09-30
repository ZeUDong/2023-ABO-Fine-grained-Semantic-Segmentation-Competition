#!/usr/bin/env python
# -*-coding:utf-8-
"""
@Author: Yue Wang
@Contact:yuewangx@mit.edu
@File: model.py
@Time:2018/10/13 6:35 PM
"""
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = - xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims) ->(batch_size*num_po
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(negative_slope=0.2)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x1 = self.residual_function(x)
        # print(x1.shape)
        x2 = self.shortcut(x)
        # print(x2.shape)
        x1 = x1+x2
        # print(x1.shape)
        x1 = self.relu(x1)
        return x1


class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=256, input_dim=3):
        super(DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512+256, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # self.conv1 = BasicBlock(input_dim*2, 64)
        # self.conv2 = BasicBlock(64*2, 64)
        # self.conv3 = BasicBlock(64*2, 128)
        # self.conv4 = BasicBlock(128*2, 256)
        # self.conv6 = BasicBlock(128*2, 256)
        # self.conv5 = BasicBlock(512+256, emb_dims)

        # self.Linear1 = nn.Linear(args.emb_dims*2, 512, bias-False)
        # self.bn6=nn.BatchNorm1d(512)
        # self.dp1= nn.Dropout(p=args.dropout)
        # self.linear2=nn.Linear(512,256)
        # self.bnz= nn.BatchNormld(256)
        # self.dp2=nn.Dropout(p=args.dropout)
        # self.Linear3=nn.Linear(256,output_channels)

    def forward(self, x):
        # input B, 6, N
        # output B,Emd*2
        batch_size = x.size(0)
        point_size = x.size(2)

        x = get_graph_feature(x, k=self.k)
        #print(x.shape)  [2, 12, 8192, 20]
        x = self.conv1(x)   
        #print(x.shape)  [2, 64, 8192, 20]
        #bb
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = self.conv6(x4)
        x5 = x.max(dim=-1, keepdim=True)[0]
        x5 = x5.repeat(1, 1, point_size)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv5(x)
        # x1 = F.adaptive_max_poolid(x,1).view(batch_size,-1)
        # x2 = F.adaptive_avg_poolid(x,1),view(batch_size,-1)
        # x= torch.cat((x1,x2),1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.Linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        return x  # B,Emd,N
