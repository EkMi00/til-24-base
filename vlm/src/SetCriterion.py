import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from BoxUtils import BoxUtils


class SetCriterion(nn.Module):
	""" This class computes the loss for DETR.
	The process happens in two steps:
		1) we compute hungarian assignment between ground truth boxes and the outputs of the model
		2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
	"""
	def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
		""" Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			weight_dict: dict containing as key the names of the losses and as values their relative weight.
			eos_coef: relative classification weight applied to the no-object category
			losses: list of all the losses to be applied. See get_loss for list of available losses.
		"""
		super().__init__()
		self.num_classes = num_classes
		self.matcher = matcher
		self.weight_dict = weight_dict
		self.eos_coef = eos_coef
		self.losses = losses
		empty_weight = torch.ones(self.num_classes  + 1)
		empty_weight[-1] = self.eos_coef
		# self.empty_weight = empty_weight
		self.register_buffer('empty_weight', empty_weight)

	def loss_labels(self, outputs, targets, indices, num_boxes):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		# logging.info(f"loss_labels - {outputs.keys()}")
		assert 'logits' in outputs
		src_logits = outputs['logits']

		idx = self._get_src_permutation_idx(indices)
		target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)]).to(torch.int64)
		target_classes = torch.full(src_logits.shape[:2], self.num_classes,
									dtype=torch.int64, device=src_logits.device).to(torch.int64)
		target_classes[idx] = target_classes_o
		if self.empty_weight is not None:
			assert self.empty_weight.shape[0] == self.num_classes + 1

		loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
		losses = {'loss_ce': loss_ce}
		return losses

	@torch.no_grad()
	def loss_cardinality(self, outputs, targets, indices, num_boxes):
		""" Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
		This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
		"""
		pred_logits = outputs['logits']
		device = pred_logits.device
		tgt_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
		# Count the number of predictions that are NOT "no-object" (which is the last class)
		card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
		card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
		losses = {'cardinality_error': card_err}
		return losses

	def loss_boxes(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
			targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
			The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
		"""
		assert 'pred_boxes' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs['pred_boxes'][idx]
		target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

		loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

		losses = {}
		losses['loss_bbox'] = loss_bbox.sum() / num_boxes

		loss_giou = 1 - torch.diag(BoxUtils.generalized_box_iou(
			BoxUtils.box_cxcywh_to_xyxy(src_boxes),
			BoxUtils.box_cxcywh_to_xyxy(target_boxes))
		)
		losses['loss_giou'] = loss_giou.sum() / num_boxes
		return losses

	def _get_src_permutation_idx(self, indices):
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		# permute targets following indices
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
		loss_map = {
			'labels': self.loss_labels,
			'cardinality': self.loss_cardinality,
			'boxes': self.loss_boxes,
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

	def forward(self, outputs, targets):
		""" This performs the loss computation.
		Parameters:
				outputs: dict of tensors, see the output specification of the model for the format
				targets: list of dicts, such that len(targets) == batch_size.
						The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		# logging.info(f"{type(outputs)=}")
		# logging.info(f"{type(targets)=}")
		outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

		# Retrieve the matching between the outputs of the last layer and the targets
		indices = self.matcher(outputs_without_aux, targets)

		# Compute the average number of target boxes accross all nodes, for normalization purposes
		num_boxes = sum(len(t["class_labels"]) for t in targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
		return losses