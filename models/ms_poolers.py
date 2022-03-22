# Modified from detectron2.modeling.poolers
# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List
import torch
from torch import nn
from torchvision.ops import RoIPool

from detectron2.layers import ROIAlign, ROIAlignRotated, cat, nonzero_tuple, shapes_to_tensor
from detectron2.structures import Boxes
from detectron2.modeling.poolers import convert_boxes_to_pooler_format

"""
To export ROIPooler to torchscript, in this file, variables that should be annotated with
`Union[List[Boxes], List[RotatedBoxes]]` are only annotated with `List[Boxes]`.

TODO: Correct these annotations when torchscript support `Union`.
https://github.com/pytorch/pytorch/issues/41412
"""


class MSROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        """
        sampling_ratio is made a list of int, to allow specialized sampling ratio for each scale,
        other args are kept the same as ROIPooler

        Args:
            output_size (tuple[int] or list[int]): output size of the pooled region, now only support square bbox.
                e.g., [4, 8, 16, 32]. The length of the tuple or list must equals to the length of scales.
        """
        super().__init__()

        # if isinstance(output_size, int):
        #     output_size = (output_size, output_size)
        # assert len(output_size) == 2
        # assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        assert len(output_size) == len(scales)
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    (size, size), spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for size, scale in zip(output_size, scales)
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    (size, size), spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for size, scale in zip(output_size, scales)
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool((size, size), spatial_scale=scale)
                for size, scale in zip(output_size, scales)
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated((size, size), spatial_scale=scale, sampling_ratio=sampling_ratio)
                for size, scale in zip(output_size, scales)
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        """
        This is different to ROIPooler.
        It repeatedly samples roi from features of all scale for each bbox.

        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            List[Tensor]:
                A list of tensors, each of shape (M, C, self.output_size[level], self.output_size[level])
                where M is the total number of boxes aggregated over all N batch images (for each feature scale)
                and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )
        if len(box_lists) == 0:
            raise NotImplementedError("len(box_lists) == 0")

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        output_list = []
        # pool all scale for each bbox, NOTE this is more computational intensive
        for level, pooler in enumerate(self.level_poolers):
            pooler_fmt_boxes_level = pooler_fmt_boxes
            output = pooler(x[level], pooler_fmt_boxes_level)
            output_list.append(output)

        return output_list