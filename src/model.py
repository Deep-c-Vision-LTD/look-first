"""
IntentNet v3: intent-conditioned saliency with the two upgrades that were never
applied to the SPL-rejected v2 (which scored CC 0.707 vs Gazeformer's 0.738 on
COCO-Search18 val):

  1. input resolution 224 -> 448 (patch grid 16x16 -> 32x32), and
  2. readout upgraded from a 2-conv FiLM stack to an ASPP readout with FiLM
     conditioning (the plain-conv -> ASPP swap was the largest single gain in
     the unconditional saliency line).

Backbone (DINOv2-base) and text encoder (CLIP ViT-B/32) stay frozen, matching
v2. FiLM generators are zero-initialized so the model starts as an
unconditional saliency model and learns the conditioning delta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPTextModel, CLIPTokenizer



class ASPPModule(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU()))
        # global context branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.proj(torch.cat(outs, dim=1))


class FiLM(nn.Module):
    """Text embedding -> per-channel (gamma, beta); zero-init => identity."""
    def __init__(self, cond_dim, n_ch):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * n_ch)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.n_ch = n_ch

    def forward(self, x, cond):
        gb = self.proj(cond)                      # B, 2C
        gamma, beta = gb[:, :self.n_ch], gb[:, self.n_ch:]
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ASPPFiLMReadout(nn.Module):
    def __init__(self, in_ch=768, ch=128, cond_dim=512, dilations=(1, 3, 6, 12)):
        super().__init__()
        self.aspp = ASPPModule(in_ch, ch, dilations)
        self.film1 = FiLM(cond_dim, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film2 = FiLM(cond_dim, ch)
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, feat, cond):
        x = self.aspp(feat)
        x = F.gelu(self.film1(x, cond))
        x = F.gelu(self.film2(self.conv2(x), cond))
        return self.head(x)


class IntentASPP(nn.Module):
    def __init__(self, vision_backbone="facebook/dinov2-base",
                 text_backbone="openai/clip-vit-base-patch32",
                 readout_dim=128, use_center_bias=True):
        super().__init__()
        self.vit = AutoModel.from_pretrained(vision_backbone)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        feat_dim = self.vit.config.hidden_size

        self.text_enc = CLIPTextModel.from_pretrained(text_backbone)
        for p in self.text_enc.parameters():
            p.requires_grad = False
        self.text_enc.eval()
        text_dim = self.text_enc.config.hidden_size

        self.readout = ASPPFiLMReadout(feat_dim, readout_dim, text_dim)
        self.use_center_bias = use_center_bias
        if use_center_bias:
            self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16))
        self.tokenizer = CLIPTokenizer.from_pretrained(text_backbone)

    @torch.no_grad()
    def visual_features(self, pixel_values):
        out = self.vit(pixel_values=pixel_values, interpolate_pos_encoding=True)
        tok = out.last_hidden_state[:, 1:, :]
        B, N, C = tok.shape
        h = w = int(round(N ** 0.5))
        return tok.transpose(1, 2).reshape(B, C, h, w)

    @torch.no_grad()
    def text_embed(self, texts, device):
        toks = self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        out = self.text_enc(**toks)
        eos = toks["attention_mask"].sum(dim=1) - 1
        return out.last_hidden_state[torch.arange(len(eos)), eos]

    def readout_forward(self, feat, cond):
        s = self.readout(feat, cond)
        if self.use_center_bias:
            cb = F.interpolate(self.center_bias, size=s.shape[-2:],
                               mode="bilinear", align_corners=False)
            s = s + cb
        return s
