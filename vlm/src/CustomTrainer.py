import torch
from transformers import Trainer
from HungarianMatcher import HungarianMatcher
from SetCriterion import SetCriterion

class CustomTrainer(Trainer):
    def __init__(self, categories, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model2 = 
        self.categories = categories
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # print(labels)

        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["attention_mask"] = inputs["attention_mask"][0]

        # print(inputs)

        # print(inputs["input_ids"].shape,inputs["attention_mask"].shape)

        outputs = model(**inputs, return_dict=True)
        # print(outputs)
        loss = self.custom_loss(outputs, labels)
        loss_ce = loss['loss_ce'].cpu().item()
        loss_bbox = loss['loss_bbox'].cpu().item()
        loss_giou = loss['loss_giou'].cpu().item()
        # cardinality_error = loss['cardinality_error'].cpu().item()
        # print(
        #     f"loss_ce={loss_ce:.2f}",
        #     f"loss_bbox={loss_bbox:.2f}",
        #     f"loss_giou={loss_giou:.2f}",
        #     f"cardinality_error={cardinality_error:.2f}",
        #     sep="\t")
        loss = sum(loss.values())[0] #add
        return (loss, outputs) if return_outputs else loss


    def custom_loss(self, logits, labels):
        num_classes = self.categories + 1
        matcher = HungarianMatcher(cost_class = 1, cost_bbox = 5, cost_giou = 2)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        losses = ['labels', 'boxes', 'cardinality']
        # losses = ['labels', 'boxes']
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.005, losses=losses)
        criterion.to(self.device)
        loss = criterion(logits, labels)
        # print(labels)
        return loss