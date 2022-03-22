# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import numpy as np
import cv2
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model

from main import get_args_parser as get_main_args_parser

from datasets.coco_eval import CocoEvaluator
import torchvision.transforms as transforms
import datasets.transforms as T


def show_demo_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--output_dir', type=float, default=None, help='path to save demo images')
    parser.add_argument('--save', action='store_true', help='whether save the demo images')
    parser.add_argument('--line', type=int, default=3, help='line width to draw bounding boxes')
    parser.add_argument('--thresh', type=float, default=0.9, help='score threshold for showing boxes')

    return parser


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b


# RGB reversed for cv2
class2color = {
    'bus': (36, 140, 135),  # My
    'bicycle': (42, 42, 165),  # Brown
    'car': (0, 255, 0),  # Lime
    'motorcycle': (226, 43, 138),  # BlueViolet
    'person': (230, 128, 94),  # My Blue
    'rider': (163, 28, 191),  # My pink
    'train': (170, 178, 32),  # LightSeaGreen
    'truck': (23, 150, 187),  # My brwon
}


def plot_results(pil_img, prob, labels, boxes, output_dir, save_name, lineWidth=2, dataset=None):
    """Visual debugging of detections."""
    assert dataset is not None
    if dataset in ['cityscapes']:
        idx2cls = ['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus']
    elif 'coco' in dataset:
        idx2cls = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                   'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                   'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    else:
        raise NotImplementedError("unknown dataset!")

    im2show = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    for p, cls_idx, (xmin, ymin, xmax, ymax) in zip(prob, labels, boxes.tolist()):
        xmin = int(np.round(xmin))
        ymin = int(np.round(ymin))
        xmax = int(np.round(xmax))
        ymax = int(np.round(ymax))
        cls_name = idx2cls[cls_idx-1]
        # color = class2color[cls_name]
        color = (0, 255, 0)
        cv2.rectangle(im2show, (xmin, ymin), (xmax, ymax), color, lineWidth)
        cv2.putText(im2show, '%s: %.3f' % (cls_name, p), (xmin, ymin + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    result_path = os.path.join(output_dir, save_name)
    cv2.imwrite(result_path, im2show)


@torch.no_grad()
def evaluate_and_demo(model, criterion, postprocessors, data_loader, base_ds, device,
                      val_transforms, use_meta=False, main_args=None, show_args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # imgid + model + thresh + line
        model_name = main_args.resume.split('/')[-2]
        save_name = 'imgid{}_{}_thresh{}_line{}.png'.format(
            str(targets[0]['image_id'].item()), model_name, str(show_args.thresh), str(show_args.line)
        )
        # input with only ToTensor transformation
        samples_pil = transforms.ToPILImage()(samples.tensors.squeeze()).convert("RGB")  # pil image w/o resizing
        samples, targets = val_transforms(samples_pil, targets[0])
        samples = samples.unsqueeze(0)
        targets = [targets]

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if use_meta:
            meta_info = {
                'size': torch.stack([t['size'][[1,0]] for t in targets]),  # (bs, 2)  W, H
            }
            outputs = model(samples, meta_info)
        else:
            outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        assert len(results) == 1
        # (100,), (100,), (100, 4)
        probs, labels, bboxes_scaled = results[0]['scores'], results[0]['labels'], results[0]['boxes']
        keep = probs > show_args.thresh

        if show_args.save:
            if show_args.output_dir is None:
                show_args.output_dir = os.path.join('demo', model_name)
            Path(show_args.output_dir).mkdir(parents=True, exist_ok=True)
            plot_results(samples_pil, probs[keep], labels[keep], bboxes_scaled[keep],
                         output_dir=show_args.output_dir, save_name=save_name,
                         lineWidth=show_args.line, dataset=main_args.dataset_file)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator


def main(args, show_args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_val = build_dataset(image_set='val', args=args)
    val_transforms = dataset_val._transforms
    dataset_val._transforms = T.Compose([
            T.ToTensor()
        ])

    if args.distributed:
        if args.cache_mode:
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    use_meta = True if 'roi' in args else False
    test_stats, coco_evaluator = evaluate_and_demo(
        model, criterion, postprocessors, data_loader_val, base_ds, device,
        val_transforms, use_meta=use_meta, main_args=args, show_args=show_args
    )
    return


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    show_args, _ = show_demo_args_parser().parse_known_args()
    main_args = get_main_args_parser().parse_args(_)
    main_args.batch_size = 1
    main(main_args, show_args)

