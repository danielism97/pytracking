import torch.nn as nn
import torch.optim as optim
from ltr.dataset import Lasot, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
import ltr.models.bbreg.atom as atom_models
from ltr import actors
from ltr.trainers import LTRDistillationTrainer
import ltr.data.transforms as tfm
from ltr.models.loss import distillation
from ltr.admin import loading


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'distilled ATOM IoUNet with default settings according to the paper.'
    settings.batch_size = 64
    settings.num_workers = 8
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    settings.feature_sz = 18
    settings.output_sz = settings.feature_sz * 16
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(11)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    trackingnet_val = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(11,12)))

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # Data processing to do on the validation pairs
    data_processing_val = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=proposal_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ATOMSampler([lasot_train, trackingnet_train, coco_train], [1,1,1],
                                samples_per_epoch=1000*settings.batch_size, max_gap=50, processing=data_processing_train)

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
    teacher_net = atom_models.atom_resnet18(backbone_pretrained=True)
    teacher_path = '/home/ddanier/pytracking/pytracking/networks/atom_default.pth'
    teacher_net = loading.load_weights(teacher_net, teacher_path, strict=True)
    print('*******************Teacher net loaded successfully*******************')
    
    # Create student network and actor
    student_net = atom_models.atom_resnet18tiny(backbone_pretrained=False)

    ##########################################################
    ### Distil backbone first, turn off grad for regressor ###
    ##########################################################
    for p in student_net.bb_regressor.parameters():
        p.requires_grad_(False)
    
    objective = distillation.CFKDLoss(reg_loss=nn.MSELoss(), w_ts=0., w_ah=0., w_cf=1., w_fd=1.)
    actor = actors.AtomCompressionActor(student_net, teacher_net, objective)

    # Optimizer
    optimizer = optim.Adam(actor.student_net.feature_extractor.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Create trainer
    trainer = LTRDistillationTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(30, load_latest=False, fail_safe=True)

    ########################################################
    ## Distil regressor next, turn off grad for backbone ###
    ########################################################
    for p in trainer.actor.student_net.bb_regressor.parameters():
        p.requires_grad_(True)
    for p in trainer.actor.student_net.feature_extractor.parameters():
        p.requires_grad_(False)

    objective = distillation.CFKDLoss(reg_loss=nn.MSELoss(), w_ts=1., w_ah=0.01, w_cf=0., w_fd=0.)
    trainer.actor.objective = objective

    # Optimizer
    trainer.optimizer = optim.Adam(trainer.actor.student_net.bb_regressor.parameters(), lr=1e-2)

    trainer.lr_scheduler = optim.lr_scheduler.StepLR(trainer.optimizer, step_size=15, gamma=0.1)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(80, load_latest=False, fail_safe=True)
