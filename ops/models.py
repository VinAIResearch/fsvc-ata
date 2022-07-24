# Code for "TAM: Temporal Adaptive Module for Video Recognition"
# arXiv: 2005.06803
# Zhaoyang liu*, Limin Wang, Wayne Wu, Chen Qian, Tong Lu
# zyliumy@gmail.com

import numpy as np
import torch
import torchvision
from ops.basic_ops import Identity
from ops.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip
from torch import nn
from torch.nn.init import constant_, normal_
from torchvision.transforms import Compose


class TSN(nn.Module):
    def __init__(
        self,
        num_class,
        num_segments,
        modality,
        base_model="resnet101",
        new_length=None,
        dropout=0.8,
        img_feature_dim=256,
        crop_num=1,
        partial_bn=True,
        print_spec=True,
        pretrain="imagenet",
        fc_lr5=False,
    ):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.num_class = num_class
        self.reshape = True
        self.dropout = dropout
        self.crop_num = crop_num
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(
                (
                    """
                   Initializing TSN with base model: {}.
                   TSN Configurations:
                   input_modality:     {}
                   num_segments:       {}
                   new_length:         {}
                   dropout_ratio:      {}
                   img_feature_dim:    {}
                   """.format(
                        base_model,
                        self.modality,
                        self.num_segments,
                        self.new_length,
                        self.dropout,
                        self.img_feature_dim,
                    )
                )
            )

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)

        if self.modality == "Flow":
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == "RGBDiff":
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class * self.num_segments)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, "weight"):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print("=> base model: {}".format(base_model))

        if "resnet" in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = "fc"
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == "Flow":
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == "RGBDiff":
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        else:
            raise ValueError("Unknown base model: {}".format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        conv_op = (
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose1d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
        )

        for m in self.modules():
            if isinstance(m, conv_op):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        params_group = [
            {
                "params": first_conv_weight,
                "lr_mult": 5 if self.modality == "Flow" else 1,
                "decay_mult": 1,
                "name": "first_conv_weight",
            },
            {
                "params": first_conv_bias,
                "lr_mult": 10 if self.modality == "Flow" else 2,
                "decay_mult": 0,
                "name": "first_conv_bias",
            },
            {"params": normal_weight, "lr_mult": 1, "decay_mult": 1, "name": "normal_weight"},
            {"params": normal_bias, "lr_mult": 2, "decay_mult": 0, "name": "normal_bias"},
            {"params": bn, "lr_mult": 1, "decay_mult": 0, "name": "BN scale/shift"},
            {"params": custom_ops, "lr_mult": 1, "decay_mult": 1, "name": "custom_ops"},
            # for fc
            {"params": lr5_weight, "lr_mult": 5, "decay_mult": 1, "name": "lr5_weight"},
            {"params": lr10_bias, "lr_mult": 10, "decay_mult": 0, "name": "lr10_bias"},
        ]

        return params_group

    def forward(self, input, no_reshape=False, feature=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == "RGBDiff":
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            if feature:
                setattr(self.base_model, self.base_model.last_layer_name, Identity())
                base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            else:
                base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        if self.training:
            base_out = self.new_fc(base_out)
            base_out = base_out.view((-1, self.num_segments) + (self.num_segments, self.num_class))
            D = base_out
            norm_D = base_out.softmax(2)
            return D, norm_D

        return base_out.view((-1, self.num_segments) + base_out.size()[1:])

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view(
            (
                -1,
                self.num_segments,
                self.new_length + 1,
                input_c,
            )
            + input.size()[2:]
        )
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(
            2 * self.new_length,
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == "BNInception":
            import torch.utils.model_zoo as model_zoo

            sd = model_zoo.load_url("https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1")
            base_model.load_state_dict(sd)
            print("=> Loading pretrained Flow weight done...")
        else:
            print("#" * 30, "Warning! No Flow pretrained model is found")
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()), 1
            )
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(
            new_kernel_size[1],
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True, dataset="Kinetics"):
        label_transforms = None
        if "somethingv2" in dataset:
            label_transforms = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
        elif "something" in dataset:
            label_transforms = {3: 5, 5: 3, 31: 42, 42: 31, 53: 67, 67: 53}

        if self.modality == "RGB":
            if flip:
                crop_op = GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75, 0.66])
                flip_op = GroupRandomHorizontalFlip(False, label_transforms)
                return Compose([crop_op, flip_op])
            else:
                print("#" * 20, "NO FLIP!!!")
                return Compose([GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75, 0.66])])
        elif self.modality == "Flow":
            return Compose(
                [GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=True)]
            )
        elif self.modality == "RGBDiff":
            return Compose(
                [GroupMultiScaleCrop(self.input_size, [1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_flow=False)]
            )
