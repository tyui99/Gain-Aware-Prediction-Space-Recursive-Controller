from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller_core import InternalRecursiveController


def _to_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        return x[:, :3, :, :]
    pad_channels = 3 - x.shape[1]
    pad = x[:, -1:, :, :].repeat(1, pad_channels, 1, 1)
    return torch.cat([x, pad], dim=1)

class _LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _TokenMixerBlock(nn.Module):
    def __init__(self, channels, drop=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = _LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x + residual


class _Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.block = _TokenMixerBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.block(x)


class InternalBackboneA(nn.Module):
    """Compact encoder-decoder backbone used by one paper preset."""

    def __init__(self, in_channels=1, num_classes=2, widths=(32, 64, 128, 256), depths=(1, 1, 2, 2), return_features=False):
        super().__init__()
        w1, w2, w3, w4 = widths
        d1, d2, d3, d4 = depths
        self.return_features = bool(return_features)
        self.feature_channels = [int(w1), int(w2), int(w3), int(w4)]
        self.output_kind = "logits"

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(*[_TokenMixerBlock(w1) for _ in range(d1)])
        self.down1 = _Downsample(w1, w2)
        self.stage2 = nn.Sequential(*[_TokenMixerBlock(w2) for _ in range(d2)])
        self.down2 = _Downsample(w2, w3)
        self.stage3 = nn.Sequential(*[_TokenMixerBlock(w3) for _ in range(d3)])
        self.down3 = _Downsample(w3, w4)
        self.stage4 = nn.Sequential(*[_TokenMixerBlock(w4) for _ in range(d4)])

        self.up3 = _UpBlock(w4, w3, w3)
        self.up2 = _UpBlock(w3, w2, w2)
        self.up1 = _UpBlock(w2, w1, w1)

        self.head = nn.Sequential(
            nn.Conv2d(w1, w1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.GELU(),
            nn.Conv2d(w1, num_classes, kernel_size=1),
        )

    def forward(self, x):
        s1 = self.stage1(self.stem(x))
        s2 = self.stage2(self.down1(s1))
        s3 = self.stage3(self.down2(s2))
        s4 = self.stage4(self.down3(s3))

        x = self.up3(s4, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        logits = self.head(x)
        if self.return_features:
            return {"logits": logits, "features": [s1, s2, s3, s4]}
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output

class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output
class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups= nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  #  size/2, channel: nIn--->nOut
        
        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        
        self.bn = nn.BatchNorm2d(2*nOut, eps=1e-3)
        self.act = nn.PReLU(2*nOut)
        self.reduce = Conv(2*nOut, nOut,1,1)  #reduce dimension: 2*nOut--->nOut
        
        self.F_glo = FGlo(nOut, reduction)    

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur],1)  #  the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)     #channel= nOut
        
        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1) # local feature
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate) # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo= FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output

class InputInjection(nn.Module):
    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for _ in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class _ContextBackboneCore(nn.Module):
    """
    Compact context backbone used by one paper preset.
    """
    def __init__(self, classes=19, M= 3, N= 21, dropout_flag = False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(3, 32, 3, 2)      # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)                          
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)      

        self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  #down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + 3)
        
        #stage 2
        self.level2_0 = ContextGuidedBlock_Down(32 +3, 64, dilation_rate=2,reduction=8)  
        self.level2 = nn.ModuleList()
        for _ in range(0, M-1):
            self.level2.append(ContextGuidedBlock(64 , 64, dilation_rate=2, reduction=8))  #CG block
        self.bn_prelu_2 = BNPReLU(128 + 3)
        
        #stage 3
        self.level3_0 = ContextGuidedBlock_Down(128 + 3, 128, dilation_rate=4, reduction=16) 
        self.level3 = nn.ModuleList()
        for _ in range(0, N-1):
            self.level3.append(ContextGuidedBlock(128 , 128, dilation_rate=4, reduction=16)) # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False),Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        #init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d')!= -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1,  output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
       
        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = F.upsample(classifier, input.size()[2:], mode='bilinear',align_corners = False)   #Upsample score map, factor=8
        return out



class InternalBackboneB(nn.Module):
    def __init__(self, num_classes: int = 2, M: int = 3, N: int = 21, dropout_flag: bool = False, return_features: bool = False):
        super().__init__()
        self.model = _ContextBackboneCore(classes=num_classes, M=M, N=N, dropout_flag=dropout_flag)
        self.return_features = bool(return_features)
        self.feature_channels = [35, 131, 256]
        self.output_kind = 'logits'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_rgb = _to_three_channels(x)
        m = self.model
        output0 = m.level1_0(input_rgb)
        output0 = m.level1_1(output0)
        output0 = m.level1_2(output0)
        inp1 = m.sample1(input_rgb)
        inp2 = m.sample2(input_rgb)
        output0_cat = m.b1(torch.cat([output0, inp1], 1))
        output1_0 = m.level2_0(output0_cat)
        output1 = output1_0
        for i, layer in enumerate(m.level2):
            output1 = layer(output1_0 if i == 0 else output1)
        output1_cat = m.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = m.level3_0(output1_cat)
        output2 = output2_0
        for i, layer in enumerate(m.level3):
            output2 = layer(output2_0 if i == 0 else output2)
        output2_cat = m.bn_prelu_3(torch.cat([output2_0, output2], 1))
        classifier = m.classifier(output2_cat)
        logits = F.interpolate(classifier, input_rgb.size()[2:], mode='bilinear', align_corners=False)
        if self.return_features:
            return {'logits': logits, 'features': [output0_cat, output1_cat, output2_cat]}
        return logits

class _RuntimeAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, plugin: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.plugin = plugin
        self.last_aux = {}
        self.last_trace = []
        self.last_pred_seq = []

    @staticmethod
    def _split_backbone_output(output):
        if isinstance(output, torch.Tensor):
            return output, None
        if isinstance(output, dict):
            logits = output.get('logits')
            features = output.get('features')
            if logits is None:
                for value in output.values():
                    if isinstance(value, torch.Tensor) and value.ndim == 4:
                        logits = value
                        break
            return logits, features
        if isinstance(output, (list, tuple)):
            if len(output) == 2 and isinstance(output[0], torch.Tensor):
                return output[0], output[1]
            for value in output:
                if isinstance(value, torch.Tensor) and value.ndim == 4:
                    return value, output
        raise TypeError(f'Unsupported backbone output type: {type(output)!r}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        logits, features = self._split_backbone_output(output)
        refined = self.plugin(logits, input_ref=x, backbone_features=features)
        self.last_aux = dict(getattr(self.plugin, 'last_aux', {}) or {})
        self.last_trace = list(getattr(self.plugin, 'last_trace', []) or [])
        self.last_pred_seq = list(getattr(self.plugin, 'last_pred_seq', []) or [])
        return refined



def _make_plugin(
    num_classes: int,
    max_steps: int,
    min_steps: int,
    process_mode: str,
    halt_patience: int,
    halt_threshold: float,
    local_branch: str = 'off',
) -> nn.Module:
    use_local_branch = str(local_branch or 'off') == 'axial'
    halt_mode = 'gain_aware_act' if 'patience' in str(process_mode) else 'double_evidence_patience'
    return InternalRecursiveController(
        num_classes=num_classes,
        hidden_dim=64,
        max_steps=max_steps,
        min_steps=min_steps,
        evidence_mode='logits_plus_image',
        feature_contract='none',
        pyramid_merge='mean',
        enable_global_branch=True,
        enable_local_branch=use_local_branch,
        fusion_mode='confidence_scalar',
        halt_mode=halt_mode,
        halt_threshold=halt_threshold,
        halt_patience=halt_patience,
        readout_mode='last',
        budget_profile='micro_state_v1',
        cue_mode='single_scale_compressed',
        state_stride=8,
        transition_groups=4,
        transition_cheap_core='grouped_hybrid',
        micro_state_dim=4,
        micro_state_rank=2,
        micro_global_mode='groupwise_rank1',
        micro_global_groups=4,
        micro_global_spatial_rank=1,
        micro_edge_stats_enhanced=False,
        micro_local_confidence=use_local_branch,
        micro_local_op='axial_edge_restore' if use_local_branch else 'edge_restore',
        micro_state_gain_enhanced=True,
        micro_state_gain_decomposition=False,
        micro_fusion_mode='step_scalar',
        micro_halt_stabilized=False,
        micro_halt_gain_ema=False,
        micro_halt_budget_aware=False,
        micro_halt_momentum=0.7,
    )


class AssembledRuntimeB(_RuntimeAdapter):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        M: int = 3,
        N: int = 21,
        dropout_flag: bool = False,
        **plugin_kwargs,
    ):
        _ = in_channels
        backbone = InternalBackboneB(num_classes=num_classes, M=M, N=N, dropout_flag=dropout_flag)
        plugin = _make_plugin(num_classes=num_classes, **plugin_kwargs)
        super().__init__(backbone=backbone, plugin=plugin)


class AssembledRuntimeA(_RuntimeAdapter):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        widths=None,
        depths=None,
        **plugin_kwargs,
    ):
        backbone = InternalBackboneA(
            in_channels=in_channels,
            num_classes=num_classes,
            widths=tuple(widths or [32, 64, 128, 256]),
            depths=tuple(depths or [1, 1, 2, 2]),
        )
        plugin = _make_plugin(num_classes=num_classes, **plugin_kwargs)
        super().__init__(backbone=backbone, plugin=plugin)
