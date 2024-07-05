# -*- coding: utf-8 -*-
"""
Created on 02/06/2022

@author: XLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Downsample_Conv2d_2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, is_drop):
        super(Downsample_Conv2d_2, self).__init__()
        self.is_drop = is_drop

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        if is_drop:
            self.dropout1 = nn.Dropout(p=0.6)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        if self.is_drop:
            outputs = self.dropout1(outputs)
        return outputs


class Downsample_Conv2d_3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, is_drop):
        super(Downsample_Conv2d_3, self).__init__()
        self.is_drop = is_drop

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        if is_drop:
            self.dropout1 = nn.Dropout(p=0.6)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        if self.is_drop:
            outputs = self.dropout1(outputs)
        return outputs


class Upsample_Block(nn.Module):
    def __init__(self, in_size, up_dim):
        super(Upsample_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_size, up_dim, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs1 = self.conv1(inputs1)
        outputs2 = torch.cat([outputs1, inputs2], 1)
        outputs2 = self.upsample(outputs2)

        return outputs2


class ChannelAttention_Block(nn.Module):
    def __init__(self, out_size, ratio):
        super(ChannelAttention_Block, self).__init__()
        self.out_size = out_size
        self.weight1 = nn.Parameter(torch.Tensor(out_size, int(out_size / ratio)), requires_grad=True)
        self.bias1 = nn.Parameter(torch.Tensor(int(out_size / ratio)), requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(int(out_size / ratio), out_size), requires_grad=True)
        self.bias2 = nn.Parameter(torch.Tensor(out_size), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        x = torch.mean(inputs, 3)
        x = torch.mean(x, 2)
        x = torch.matmul(x, self.weight1) + self.bias1
        x = torch.matmul(x, self.weight2) + self.bias2
        x = x.reshape(-1, self.out_size, 1, 1)
        outputs = inputs * x
        return outputs


class Unet_Al_Seg(nn.Module):
    def __init__(self, feature_scale=1, up_dim=32, n_classes=2, in_channels=1, is_batchnorm=True, is_drop=True):
        super(Unet_Al_Seg, self).__init__()
        self.is_drop = is_drop

        filters = [64, 128, 256, 512, 512]
        filters = [int(x / feature_scale) for x in filters]
        # ============= downsample =====================================
        """ conv1 """
        self.conv1 = Downsample_Conv2d_2(in_channels, filters[0], is_batchnorm, is_drop)
        """ conv2 """
        self.conv2 = Downsample_Conv2d_2(filters[0], filters[1], is_batchnorm, is_drop)
        """ conv3 """
        self.conv3 = Downsample_Conv2d_3(filters[1], filters[2], is_batchnorm, is_drop)
        """ conv4 """
        self.conv4 = Downsample_Conv2d_3(filters[2], filters[3], is_batchnorm, is_drop)
        """ conv5 """
        self.conv5 = Downsample_Conv2d_3(filters[3], filters[4], is_batchnorm, is_drop)
        # ============= downsample =====================================
        self.conv5_up = nn.Sequential(
            nn.Conv2d(filters[4], up_dim, kernel_size=3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.conv4_up = Upsample_Block(filters[3], up_dim)
        self.conv3_up = Upsample_Block(filters[2], up_dim)
        self.conv2_up = Upsample_Block(filters[1], up_dim)
        self.conv1_up = Upsample_Block(filters[0], up_dim)
        # ============classify==================================
        self.channel_attenion1 = ChannelAttention_Block(out_size=up_dim * 5, ratio=2)
        if is_drop:
            self.dropout1_class = nn.Dropout(p=0.5)
        self.conv1_class = nn.Conv2d(up_dim * 5, up_dim * 5 // 2, kernel_size=3, stride=1, padding=1)
        if is_drop:
            self.dropout2_class = nn.Dropout(p=0.5)
        self.conv2_class = nn.Conv2d(up_dim * 5 // 2, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv5_up = self.conv5_up(conv5)
        conv4_up = self.conv4_up(conv4, conv5_up)
        conv3_up = self.conv3_up(conv3, conv4_up)
        conv2_up = self.conv2_up(conv2, conv3_up)
        conv1_up = self.conv1_up(conv1, conv2_up)
        output = self.channel_attenion1(conv1_up)
        if self.is_drop:
            output = self.dropout1_class(output)
        output = self.conv1_class(output)
        if self.is_drop:
            output = self.dropout2_class(output)
        output = self.conv2_class(output)
        output = F.softmax(output, 1)
        return output


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, pred, gt):
        N = gt.size()[0]
        smooth = 1.0e-5
        inputs = pred[:, 1, :, :]
        targets = gt[:, 1, :, :]
        inse = inputs * targets
        inse = inse.sum(1).sum(1)
        l = inputs.sum(1).sum(1)
        r = targets.sum(1).sum(1)
        dice = (2.0 * inse + smooth) / (l + r + smooth)
        loss = 1 - dice.sum() / N

        return loss


class BinaryDiceLoss_MultiObj(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss_MultiObj, self).__init__()

    def forward(self, pred_list, gt_list):
        Diceloss_all = 0
        Num = len(pred_list)
        for ii in range(Num):
            pred = pred_list[ii]
            gt = gt_list[ii]

            N = gt.size()[0]
            smooth = 1.0e-5
            inputs = pred[:, 1, :, :]
            targets = gt[:, 1, :, :]
            inse = inputs * targets
            inse = inse.sum(1).sum(1)
            l = inputs.sum(1).sum(1)
            r = targets.sum(1).sum(1)
            dice = (2.0 * inse + smooth) / (l + r + smooth)
            loss = 1 - dice.sum() / N

            Diceloss_all = Diceloss_all + loss

        return Diceloss_all



class CrossEntropyLoss_MultiObj(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_MultiObj, self).__init__()

    def forward(self, pred_list, gt_list, wt_list):
        CEloss_all = 0
        Num = len(pred_list)
        smooth = 1.0e-5
        for ii in range(Num):
            pred = pred_list[ii]
            gt = gt_list[ii]
            wt = wt_list[ii]
            N = gt.size()[0]
            cost = gt * torch.log(pred + smooth)
            cost = cost.sum(1)*wt
            cost = -cost.sum(1).sum(1).sum()/N
            CEloss_all = CEloss_all + cost
        return CEloss_all

class Unet_ToothSeg_Landmark_Alveolar_Prior(nn.Module):
    def __init__(self, feature_scale=1, up_dim=32, n_classes_th=2, in_channels=1, is_batchnorm=True, is_drop=True):
        super(Unet_ToothSeg_Landmark_Alveolar_Prior, self).__init__()
        self.is_drop = is_drop
        filters = [64, 128, 256, 512, 512]
        filters = [int(x / feature_scale) for x in filters]
        # ============= downsample =====================================
        """ conv1 """
        self.conv1 = Downsample_Conv2d_2(in_channels, filters[0], is_batchnorm, is_drop)
        """ conv2 """
        self.conv2 = Downsample_Conv2d_2(filters[0], filters[1], is_batchnorm, is_drop)
        """ conv3 """
        self.conv3 = Downsample_Conv2d_3(filters[1], filters[2], is_batchnorm, is_drop)
        """ conv4 """
        self.conv4 = Downsample_Conv2d_3(filters[2], filters[3], is_batchnorm, is_drop)
        """ conv5 """
        self.conv5 = Downsample_Conv2d_3(filters[3], filters[4], is_batchnorm, is_drop)
        # ============= downsample =====================================
        self.conv5_up = nn.Sequential(
            nn.Conv2d(filters[4], up_dim, kernel_size=3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.conv4_up = Upsample_Block(filters[3], up_dim)
        self.conv3_up = Upsample_Block(filters[2], up_dim)
        self.conv2_up = Upsample_Block(filters[1], up_dim)
        self.conv1_up = Upsample_Block(filters[0], up_dim)
        # ============classify Tooth==================================
        self.channel_attenion_th = ChannelAttention_Block(out_size=up_dim * 5, ratio=2)
        if is_drop:
            self.dropout1_class_th = nn.Dropout(p=0.5)
        self.conv1_class_th = nn.Conv2d(up_dim * 5 + 2, up_dim * 5 // 2, kernel_size=3, stride=1, padding=1)
        if is_drop:
            self.dropout2_class_th = nn.Dropout(p=0.5)
        self.conv2_class_th = nn.Conv2d(up_dim * 5 // 2, n_classes_th, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, dist_al):
        # =========Downsample====================
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        # =========Upsample=======================
        conv5_up = self.conv5_up(conv5)
        conv4_up = self.conv4_up(conv4, conv5_up)
        conv3_up = self.conv3_up(conv3, conv4_up)
        conv2_up = self.conv2_up(conv2, conv3_up)
        conv1_up = self.conv1_up(conv1, conv2_up)
        # =========Segment tooth========================
        output_th = self.channel_attenion_th(conv1_up)
        output_th = torch.cat([output_th, dist_al], 1)
        if self.is_drop:
            output_th = self.dropout1_class_th(output_th)
        output_th = self.conv1_class_th(output_th)
        if self.is_drop:
            output_th = self.dropout2_class_th(output_th)
        output_th = self.conv2_class_th(output_th)
        output_th = F.softmax(output_th, 1)

        return output_th
# if __name__ == '__main__':
#     model = Unet_ToothSeg_Landmark_Alveolar_Prior()
#     print(model)
#     writer.add_graph(model,input_to_model=torch.rand((1,320,320)))
#     writer.close()
