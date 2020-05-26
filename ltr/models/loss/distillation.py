import torch.nn as nn
import torch
from torch.nn import functional as F


class TeacherSoftLoss(nn.Module):
    """
    Loss to match iou outputs of teacher and student ATOMIoUNet
    """
    def __init__(self, reg_loss=nn.MSELoss()):
        super().__init__()
        self.reg_loss = reg_loss

    def forward(self, iou_student, iou_teacher):
        return self.reg_loss(iou_student, iou_teacher)

class AdaptiveHardLoss(nn.Module):
    """
    Loss to match iou outputs of the student ATOMIoUNet with ground truth iou.
    Produce hard loss when student is worse than teacher. Once student gets close to
    teacher, stop the loss to avoid overfitting.
    """
    def __init__(self, reg_loss=nn.MSELoss(), threshold=0.005):
        super().__init__()
        self.reg_loss = reg_loss
        self.threshold = threshold

    def forward(self, iou_student, iou_teacher, iou_gt):
        loss_teacher = self.reg_loss(iou_teacher, iou_gt)
        loss_student = self.reg_loss(iou_student, iou_gt)
        gap = loss_teacher - loss_student
        
        return loss_student if gap > self.threshold else 0.

class ModulationVectorLoss(nn.Module):
    pass

class TargetResponseLoss(nn.Module):
    pass

class DistillationBasic(nn.Module):
    """
    Objective for basic distillation, matching only iou outputs.
    Returns TeacherSoftLoss + AdaptiveHardLoss
    """
    def __init__(self, reg_loss=nn.MSELoss(), threshold_ah=None):
        super().__init__()
        # subcomponent losses, can turn off adaptive hard by setting threshold to None
        self.teacher_soft_loss = TeacherSoftLoss(reg_loss)
        self.adaptive_hard_loss = None
        if threshold_ah is not None:
            self.adaptive_hard_loss = AdaptiveHardLoss(reg_loss, threshold_ah)
            

    def forward(self, iou_student, iou_teacher, iou_gt):
        loss = self.teacher_soft_loss(iou_student, iou_teacher)
        if self.adaptive_hard_loss is not None:
            loss += self.adaptive_hard_loss(iou_student, iou_teacher, iou_gt)
        return loss