# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 17:32
# @Author  : zhangguangyi
# @File    : contrastive_loss.py
"""The contrastive loss."""
import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """

    """

    __constants__ = ['alpha']

    def __init__(self, is_supervised: bool = True, alpha: float = 0.05):
        super().__init__()
        self.is_supervised = is_supervised
        self.alpha = alpha

    def supervised(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """

        Args:
            y_pred:[x, x+, x-, ...]
            y_true:

        Returns:

        """
        y_true = torch.arange(y_pred.shape[0], device=y_pred.device)
        use_row = torch.where((y_true + 1) % 3 != 0)[0]
        y_true = (use_row - use_row % 3 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=y_pred.device) * 1e12
        # 选取有效的行
        sim = torch.index_select(sim, 0, use_row)
        # 相似度矩阵除以温度系数
        sim = sim / self.alpha
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, y_true)
        return loss

    def unsupervised(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """

        Args:
            y_pred: get twice, [a,a,b,b,c,c,...]
            y_true:

        Returns:

        """
        y_true = torch.arange(y_pred.shape[0], device=y_pred.device)
        y_true = (y_true - y_true % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=y_pred.device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / self.alpha
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, y_true)
        return loss
        pass

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.is_supervised:
            loss = self.supervised(y_pred, y_true)
        else:
            loss = self.unsupervised(y_pred, y_true)
        return loss
