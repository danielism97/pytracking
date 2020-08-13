import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms

from ltr.dataset import Lasot, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
import ltr.models.bbreg.atom as atom_models
import ltr.models.bbreg.DRNet as DR_models
from ltr import actors
from ltr.trainers import LTRDistillationTrainer
import ltr.data.drtransforms as dltransforms
from ltr.models.loss import distillation
from ltr.admin import loading


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'DRNet distillation with CFKD.'
    settings.print_interval = 1                                 # How often to print loss and other info
    settings.batch_size = 32                                    # Batch size
    settings.num_workers = 8                                    # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]              # Normalize std (default pytorch ImageNet values)
    settings.search_area_factor = 5.0                           # Image patch size relative to target size
    settings.feature_sz = 18                                    # Size of feature map
    settings.output_sz = settings.feature_sz * 16               # Size of input image patches

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    settings.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

    # Train datasets
    lasot_train = Lasot(split='train')
    trackingnet_train = TrackingNet(set_ids=list(range(11)))
    coco_train = MSCOCOSeq()

    # Validation datasets
    trackingnet_val = TrackingNet(set_ids=list(range(11,12)))

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # Data processing to do on the training pairs
    data_processing_train = processing.DirectProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=settings.proposal_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # Data processing to do on the validation pairs
    data_processing_val = processing.DirectProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=settings.proposal_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ATOMSampler([lasot_train, trackingnet_train, coco_train], [1,1,1],
                                        samples_per_epoch=1800*settings.batch_size, max_gap=50,
                                        processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # The sampler for validation
    dataset_val = sampler.ATOMSampler([trackingnet_val], [1], samples_per_epoch=500*settings.batch_size, max_gap=50,
                                      processing=data_processing_val)

    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Load teacher network
    teacher_net = DR_models.drnet_resnet50(backbone_pretrained=True)
    teacher_path = '/home/ddanier/pytracking/pytracking/networks/drnet_res50.pth'
    teacher_net = loading.load_weights(teacher_net, teacher_path, strict=True)
    print('*******************Teacher net loaded successfully*******************')
    
    # Create student network and actor
    student_net = DR_models.drnet_mobilenetsmall(backbone_pretrained=True)
    objective = distillation.CFKDLoss(reg_loss=nn.MSELoss(), w_cf=0., w_fd=0.,
                                      cf_layers=['conv1','layer1','layer2','layer3'])

    # Create actor, which wraps network and objective
    actor = actors.DRCompressionActor(student_net, teacher_net, objective)

    # Optimizer
    optimizer = optim.Adam([{'params': actor.student_net.feature_extractor.parameters(), 'lr': 1e-8},
                            {'params': actor.student_net.bb_regressor.parameters()}], lr=1e-2)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Create trainer
    trainer = LTRDistillationTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(50, load_latest=False, fail_safe=True)