# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, two_stage_match=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.two_stage_match = two_stage_match

    def get_cost(self, outputs, tgt_ids, tgt_bbox):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        return C

    @torch.no_grad()
    def forward(self, outputs, targets):
        if self.two_stage_match and self.training:
            return self.forward_two_stage(outputs, targets)
        else:
            return self.forward_single_stage(outputs, targets)

    @torch.no_grad()
    def forward_single_stage(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        C = self.get_cost(outputs, tgt_ids, tgt_bbox)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward_two_stage(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # get non-repeat target labels and boxes
        tgt_ids = torch.cat([v["labels"][v['repeat'] == 1] for v in targets])  # (batch_inst, )
        tgt_bbox = torch.cat([v["boxes"][v['repeat'] == 1] for v in targets])  # (batch_inst, 4)
        # first cost matrix
        C = self.get_cost(outputs, tgt_ids, tgt_bbox)  # [bs, num_query, batch_inst]
        # first matching
        sizes_nr = [len(v["boxes"][v['repeat'] == 1]) for v in targets]  # list of num_inst of the batch
        indices_non_repeat = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes_nr, -1))]
        # matched ones: c[i][indices_non_repeat[i][0], indices_non_repeat[i][1]]

        # get repeat target labels and boxes
        tgt_ids = torch.cat([v["labels"][v['repeat'] == 0] for v in targets])
        tgt_bbox = torch.cat([v["boxes"][v['repeat'] == 0] for v in targets])
        # second cost matrix
        C = self.get_cost(outputs, tgt_ids, tgt_bbox)  # [bs, num_query, batch_inst]
        sizes_r = [len(v["boxes"][v['repeat'] == 0]) for v in targets]  # list of num_inst of the batch
        for i, c in enumerate(C.split(sizes_r, -1)):
            c[i][indices_non_repeat[i][0], :] = 1e6  # large cost ensures the matched queries won't be matched again
        indices_repeat = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes_r, -1))]

        # merge matching results
        indices = []
        for batch_idx, ((i_n, j_n), (i_r, j_r)) in enumerate(zip(indices_non_repeat, indices_repeat)):
            i = torch.cat([torch.as_tensor(i_n, dtype=torch.int64), torch.as_tensor(i_r, dtype=torch.int64)],)
            j = torch.cat([torch.as_tensor(j_n, dtype=torch.int64),
                           torch.as_tensor(j_r, dtype=torch.int64) + sizes_nr[batch_idx]])
            indices.append((i, j))
        return indices


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                            two_stage_match=args.two_stage_match)
