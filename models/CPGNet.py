import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
from torchvision.transforms import functional as T


def norm01_in_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b = x.shape[0]
    x2 = x.view(b, -1)
    x_min = x2.min(dim=1, keepdim=True).values
    x_max = x2.max(dim=1, keepdim=True).values
    return ((x2 - x_min) / (x_max - x_min + eps)).view_as(x)


class ConvBNReLU(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoLPEdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_kernel = torch.tensor([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)

        self.y_kernel = torch.tensor([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
        ], dtype=torch.float32).reshape(1, 1, 3, 3)

    @torch.no_grad()
    def forward(self, x):
        device = x.device
        x_kernel = self.x_kernel.to(device=device, dtype=x.dtype)
        y_kernel = self.y_kernel.to(device=device, dtype=x.dtype)

        gray_x = T.rgb_to_grayscale(x, num_output_channels=1)
        x_x = F.conv2d(gray_x, x_kernel, stride=1, padding=1)
        y_x = F.conv2d(gray_x, y_kernel, stride=1, padding=1)
        e_x = torch.sqrt(torch.pow(x_x, 2) + torch.pow(y_x, 2) + 1e-12)
        r_x = norm01_in_sample(e_x)
        return r_x


class PolarizationIntegrationModule(nn.Module):
    def __init__(self, in_channel=3, pool_size=5):
        super().__init__()
        self.conv_sattn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.degree_max_map = nn.Sequential(
            ConvBNReLU(c1=in_channel, c2=in_channel, k=3, s=1),
            nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2),
            nn.Sigmoid()
        )
        self.edge = DoLPEdgeExtractor()
        self.conv_aop1 = ConvBNReLU(c1=in_channel, c2=in_channel, k=3, s=1)
        self.conv_dop1 = ConvBNReLU(c1=in_channel, c2=in_channel, k=3, s=1)
        self.conv_pol1 = ConvBNReLU(c1=in_channel * 2, c2=in_channel, k=3, s=1)

    def forward(self, x):
        aop, dop = x
        _max, _ = torch.max(dop, dim=1, keepdim=True)
        _avg = torch.mean(dop, dim=1, keepdim=True)
        w_aop = aop * (self.conv_sattn(torch.cat([_max, _avg], dim=1)) + self.degree_max_map(dop))
        w_dop = dop + self.edge(dop)
        pol = self.conv_pol1(torch.cat([self.conv_aop1(w_aop), self.conv_dop1(w_dop)], dim=1))
        return pol


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PRETRAINED_PVT_PATH = os.path.join(PROJECT_ROOT, "pretrained_pvt", "pvt_v2_b2.pth")


def _load_backbone_weights(backbone: nn.Module, ckpt_path: str) -> None:
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained backbone not found: {ckpt_path}")

    saved = torch.load(ckpt_path, map_location="cpu")
    if isinstance(saved, dict) and ("state_dict" in saved or "model" in saved):
        saved = saved.get("state_dict", saved.get("model"))

    model_dict = backbone.state_dict()
    state_dict = {
        k: v for k, v in saved.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(state_dict)
    backbone.load_state_dict(model_dict)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        n, c, h, w = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        n, c, h, w = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)
        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class FrequencyMagnitudeMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        mag = self.process1(mag)

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(h, w), norm='backward')
        return x_out


class EdgeGuidedFrequencyModule(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        self.norm = LayerNorm2d(c)
        self.freq = FrequencyMagnitudeMLP(nc=c, expand=expand)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, y):
        x_step2 = self.norm(y)
        x_freq = self.freq(x_step2)
        return y + (y * x_freq) * self.gamma


class MultiDilationConvBlock(nn.Module):
    def __init__(self, dilation_series, padding_series, num_classes: int):
        super().__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    32, num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True
                )
            )

        for m in self.conv2d_list:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = out + self.conv2d_list[i + 1](x)
        return out


class FeatureEnhancementGate(nn.Module):
    def __init__(self, channels: int = 32, r: int = 4):
        super().__init__()
        out_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl = self.local_att(x)
        xg = self.global_att(x)
        wei = self.sig(xl + xg)
        return wei


class CPGNet(nn.Module):
    def __init__(self, channel: int = 32, backbone_ckpt: str = PRETRAINED_PVT_PATH):
        super().__init__()
        self.num_iterations = 2

        self.backbone = pvt_v2_b2()
        _load_backbone_weights(self.backbone, backbone_ckpt)

        self.cr1 = self._conv3x3_or_1x1(64, channel, kernel_size=1)
        self.cr2 = self._conv3x3_or_1x1(128, channel, kernel_size=1)
        self.cr3 = self._conv3x3_or_1x1(320, channel, kernel_size=1)
        self.cr4 = self._conv3x3_or_1x1(512, channel, kernel_size=1)

        self.fuse_pol = PolarizationIntegrationModule(in_channel=3, pool_size=5)
        self.pol_film1 = self._guidance_head(3, channel)
        self.pol_film2 = self._guidance_head(3, channel)
        self.pol_film3 = self._guidance_head(3, channel)
        self.pol_film4 = self._guidance_head(3, channel)
        self.film_scale = nn.Parameter(torch.tensor(0.0))

        dilation = [1, 6, 8, 12]
        padding = [1, 6, 8, 12]
        self.conv0 = self._pge_block(dilation, padding, channel)
        self.conv1 = self._pge_block(dilation, padding, channel)
        self.conv2 = self._pge_block(dilation, padding, channel)
        self.conv3 = self._pge_block(dilation, padding, channel)
        self.atts = nn.ModuleList([
            FeatureEnhancementGate(channels=channel),
            FeatureEnhancementGate(channels=channel),
            FeatureEnhancementGate(channels=channel),
            FeatureEnhancementGate(channels=channel),
        ])

        self.fre_gf0 = EdgeGuidedFrequencyModule(channel, expand=2)
        self.fre_gf1 = EdgeGuidedFrequencyModule(channel, expand=2)
        self.fre_gf2 = EdgeGuidedFrequencyModule(channel, expand=2)
        self.fre_gf3 = EdgeGuidedFrequencyModule(channel, expand=2)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.edge_scale = nn.Parameter(torch.tensor(0.0))

        self.conv4 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)
        self.conv5 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)
        self.conv6 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)
        self.conv7 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)

        self.cbr1 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)
        self.cbr2 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)
        self.cbr3 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)
        self.cbr4 = self._conv3x3_or_1x1(channel, channel, kernel_size=3)

        self.map = nn.Conv2d(channel, 1, 7, 1, 3)
        self.out_map = nn.Conv2d(channel, 1, 7, 1, 3)

        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = True

    @staticmethod
    def _conv3x3_or_1x1(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    @staticmethod
    def _guidance_head(in_channels: int, channel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, channel * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel * 2),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _pge_block(dilation_series, padding_series, channel: int) -> nn.Sequential:
        return nn.Sequential(
            MultiDilationConvBlock(dilation_series, padding_series, channel),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def _build_polarization_guidance(self, aolp: torch.Tensor, dolp: torch.Tensor):
        guidance = self.fuse_pol((aolp, dolp))
        edge_prior = self.fuse_pol.edge(dolp)
        return guidance, edge_prior

    def _apply_conditional_guidance(
        self,
        feature: torch.Tensor,
        guidance: torch.Tensor,
        guidance_head: nn.Module
    ) -> torch.Tensor:
        film = guidance_head(guidance)
        film = F.interpolate(film, size=feature.shape[2:], mode="bilinear", align_corners=False)
        alpha, beta = torch.chunk(film, 2, dim=1)

        scale = torch.sigmoid(self.film_scale)
        alpha = torch.tanh(alpha) * scale
        beta = torch.tanh(beta) * scale
        return feature * (1.0 + alpha) + beta

    def _encode_guided_rgb_features(
        self,
        rgb: torch.Tensor,
        guidance: torch.Tensor
    ):
        rgb_features = self.backbone(rgb)
        guided_features = []
        rgb_reductions = [self.cr1, self.cr2, self.cr3, self.cr4]
        guidance_heads = [self.pol_film1, self.pol_film2, self.pol_film3, self.pol_film4]
        for feature, reduction, guidance_head in zip(
            rgb_features,
            rgb_reductions,
            guidance_heads
        ):
            reduced = reduction(feature)
            guided_features.append(
                self._apply_conditional_guidance(reduced, guidance, guidance_head)
            )
        return guided_features

    def _edge_guided_frequency_refinement(
        self,
        feature: torch.Tensor,
        edge_prior: torch.Tensor,
        efm_block: nn.Module
    ) -> torch.Tensor:
        edge_feature = F.interpolate(
            edge_prior,
            size=feature.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        edge_feature = self.edge_proj(edge_feature)
        edge_feature = efm_block(edge_feature)
        return feature + torch.tanh(self.edge_scale) * edge_feature

    def _pge_efm_stage(
        self,
        feature: torch.Tensor,
        stage_index: int,
        edge_prior: torch.Tensor
    ) -> torch.Tensor:
        pge_blocks = [self.conv0, self.conv1, self.conv2, self.conv3]
        efm_blocks = [self.fre_gf0, self.fre_gf1, self.fre_gf2, self.fre_gf3]
        enhanced = pge_blocks[stage_index](feature)
        enhanced = enhanced * self.atts[stage_index](enhanced)
        enhanced = efm_blocks[stage_index](enhanced)
        return self._edge_guided_frequency_refinement(
            enhanced,
            edge_prior,
            efm_blocks[stage_index]
        )

    def _decode_once(
        self,
        guided_features,
        edge_prior: torch.Tensor,
        feedback_feature: Optional[torch.Tensor],
        output_size
    ):
        x_1, x_2, x_3, x_4 = guided_features
        f_1 = x_1 if feedback_feature is None else x_1 + feedback_feature

        gf0 = self._pge_efm_stage(f_1, 0, edge_prior)
        h1 = self.conv5(gf0)
        gf0 = F.interpolate(gf0, size=x_2.size()[2:], mode="bilinear", align_corners=False)

        gf1 = self._pge_efm_stage(gf0 + x_2, 1, edge_prior)
        h2 = self.conv6(gf1)
        gf1 = F.interpolate(gf1, size=x_3.size()[2:], mode="bilinear", align_corners=False)

        gf2 = self._pge_efm_stage(gf1 + x_3, 2, edge_prior)
        h3 = self.conv7(gf2)
        gf2 = F.interpolate(gf2, size=x_4.size()[2:], mode="bilinear", align_corners=False)

        gf3 = self._pge_efm_stage(gf2 + x_4, 3, edge_prior)
        gf3 = self.conv4(gf3)
        coarse_prediction = self.map(gf3)

        rf4 = self.cbr1(x_4 + gf3)
        rf4 = F.interpolate(rf4, size=x_3.size()[2:], mode="bilinear", align_corners=False)

        rf3 = self.cbr2(rf4 + h3)
        rf3 = F.interpolate(rf3, size=x_2.size()[2:], mode="bilinear", align_corners=False)

        rf2 = self.cbr3(rf3 + h2)
        rf2 = F.interpolate(rf2, size=x_1.size()[2:], mode="bilinear", align_corners=False)

        rf1 = self.cbr4(rf2 + h1)
        refined_prediction = self.out_map(rf1)

        coarse_prediction = F.interpolate(
            coarse_prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )
        refined_prediction = F.interpolate(
            refined_prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )
        return coarse_prediction, refined_prediction, rf1

    def forward(self, rgb: torch.Tensor, aolp: torch.Tensor, dolp: torch.Tensor):
        polarization_guidance, edge_prior = self._build_polarization_guidance(aolp, dolp)
        guided_features = self._encode_guided_rgb_features(rgb, polarization_guidance)

        coarse_predictions = []
        refined_predictions = []
        feedback_feature = None
        for _ in range(self.num_iterations):
            coarse_prediction, refined_prediction, feedback_feature = self._decode_once(
                guided_features,
                edge_prior,
                feedback_feature,
                rgb.shape[2:]
            )
            coarse_predictions.append(coarse_prediction)
            refined_predictions.append(refined_prediction)

        return coarse_predictions, refined_predictions
