#!/usr/bin/env python3
"""Build CLAM models from checkpoint keys and tensor shapes.

Infers encoder depth, input dimension, bottleneck size, and adversarial-head
sizes. The result directory's _gelu suffix selects GELU. Other names select
ReLU. Supports vanilla CLAM and models with adversarial heads.
"""
import os
import re

import torch

from final_train import ExperimentConfig


def build_from_ckpt(ckpt_path, dirname, test_dirs, device, input_dim=None):
    import run_zeroshot_dann as D
    sd = torch.load(ckpt_path, map_location='cpu')

    depth = 1
    for k in sd:
        m = re.match(r'enc_blocks\.(\d+)\.', k)
        if m:
            depth = max(depth, int(m.group(1)) + 2)
    bott = sd['enc_proj.1.weight'].shape[0] if 'enc_proj.1.weight' in sd else 0
    reg_dim = sd['slide_reg.3.weight'].shape[0] if 'slide_reg.3.weight' in sd else 0
    n_dom = sd['slide_disc.3.weight'].shape[0] if 'slide_disc.3.weight' in sd else 1
    disc_h = (sd['slide_disc.0.weight'].shape[0]
              if 'slide_disc.0.weight' in sd else 128)
    # 站点对抗 / 站点截距。推理路径（forward）**不用** site_bias —— 站点截距
    # 是训练期的固定效应，外部站点未知，推理时恒为 0。这里建出来只为让
    # load_state_dict(strict=True) 能对上键。
    n_sites = int(sd['site_bias'].numel()) if 'site_bias' in sd else 0
    idim = input_dim or sd['encoder.1.weight'].shape[1]
    legacy = '_gelu' not in os.path.basename(dirname)

    cfg = ExperimentConfig(name='s', cv_pool_dirs=[], test_dirs=test_dirs,
                           input_dim=idim)

    # Checkpoints without domain heads use the vanilla CLAM model.
    if 'slide_disc.0.weight' not in sd:
        from final_train import build_model
        m0 = build_model(cfg, device)
        m0.load_state_dict(sd, strict=True)
        m0.eval()
        return m0, dict(enc_depth=1, bottleneck=0, reg_dim=0,
                        disc_hidden=0, legacy_act=True, input_dim=idim,
                        n_sites=0, legacy_clam=True)
    m = D.DeepCDA_CLAM_SB(input_dim=idim, size_arg=cfg.size_arg,
                          k_sample=cfg.k_sample, n_domains=n_dom,
                          disc_hidden=disc_h, reg_dim=reg_dim,
                          enc_depth=depth, legacy_act=legacy,
                          bottleneck=bott, n_sites=n_sites).to(device)
    miss, unexp = m.load_state_dict(sd, strict=True)
    m.eval()
    return m, dict(enc_depth=depth, bottleneck=bott, reg_dim=reg_dim,
                   disc_hidden=disc_h, legacy_act=legacy, input_dim=idim,
                   n_sites=n_sites)
