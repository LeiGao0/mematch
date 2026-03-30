import argparse
import logging
import os
import pprint
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset
# from model.semseg.dpt import DPT
# from model.semseg.dpt_cl import DPT
from model.semseg.dpt_CASAtt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.utils import init_log, count_params


def _strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state


def build_model_from_cfg(cfg, device):
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    key = cfg['backbone'].split('_')[-1]
    assert key in model_configs, f"Unknown backbone size: {key}"
    model = DPT(**{**model_configs[key], 'nclass': cfg['nclass']})

    # try loading pretrained backbone if available
    pretrained_path = os.path.join('./pretrained', f"{cfg['backbone']}.pth")
    if os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.backbone.load_state_dict(state_dict)
        except Exception:
            try:
                model.backbone.load_state_dict(_strip_module_prefix(state_dict))
            except Exception:
                # silently ignore if backbone preload fails; evaluation may still work with checkpoint
                pass

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Test script for UniMatch-V2 (evaluate on val set)')
    parser.add_argument('--config', default='/home/b760/research/gl/unimatchv2/configs/RailSem19.yaml', type=str)
    parser.add_argument('--checkpoint', default='scripts/64exp/RailSem19/UniMatch-V2/resnet50/gl/best.pth', type=str,
                        help='Path to checkpoint containing `model` and optionally `model_ema`')
    parser.add_argument('--multiplier', default=14, type=int, help='Multiplier passed to evaluate()')
    parser.add_argument('--device', default=None, type=str, help='Device (e.g. cuda:0 or cpu). If omitted uses cuda if available')
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    logger = init_log('test', logging.INFO)
    logger.propagate = 0
    logger.info('Config: \n%s', pprint.pformat(cfg))

    device = torch.device(args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = build_model_from_cfg(cfg, device)

    # load checkpoint (may contain both 'model' and 'model_ema')
    sd = None
    sd_ema = None
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(ckpt, dict):
            sd = ckpt.get('model', None)
            sd_ema = ckpt.get('model_ema', None)
            # If it's a plain state_dict (no wrapper keys), treat ckpt as sd
            if sd is None and sd_ema is None:
                sd = ckpt
        else:
            sd = ckpt

        if sd is not None:
            try:
                model.load_state_dict(sd)
            except Exception:
                try:
                    model.load_state_dict(_strip_module_prefix(sd))
                except Exception as e:
                    logger.warning('Failed to load state_dict into model: %s', e)
        else:
            logger.warning('No `model` state found in checkpoint; proceeding with randomly initialized model')

        # prepare EMA model if available in checkpoint
        model_ema = None
        if sd_ema is not None:
            model_ema = deepcopy(model)
            try:
                model_ema.load_state_dict(sd_ema)
            except Exception:
                try:
                    model_ema.load_state_dict(_strip_module_prefix(sd_ema))
                except Exception as e:
                    logger.warning('Failed to load state_dict into model_ema: %s', e)
    else:
        logger.warning('Checkpoint %s not found; proceeding with randomly initialized model', args.checkpoint)
        model_ema = None

    # build val dataset and loader
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'test')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'

    # run evaluation for main model
    logger.info('Running evaluation on device %s, eval_mode=%s', device, eval_mode)
    model.to(device)
    mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=args.multiplier)

    logger.info('===== Evaluation result (model) =====')
    logger.info('Mean IoU: %.2f', mIoU)
    for (cls_idx, iou) in enumerate(iou_class):
        logger.info('Class [%d %s] IoU: %.2f', cls_idx, CLASSES[cfg['dataset']][cls_idx], iou)

    # if an EMA model was loaded from checkpoint, also evaluate it (matches training validation behaviour)
    if 'model_ema' in locals() and model_ema is not None:
        model_ema.to(device)
        model_ema.eval()
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=args.multiplier)
        logger.info('===== Evaluation result (model_ema) =====')
        logger.info('Mean IoU (EMA): %.2f', mIoU_ema)
        for (cls_idx, iou) in enumerate(iou_class_ema):
            logger.info('Class [%d %s] IoU (EMA): %.2f', cls_idx, CLASSES[cfg['dataset']][cls_idx], iou)


if __name__ == '__main__':
    main()
