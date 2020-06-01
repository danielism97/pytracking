from . import BaseActor


class AtomActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        # Compute loss
        loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats


class AtomBBKLActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM with BBKL"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_density', and 'gt_density'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        bb_scores = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        bb_scores = bb_scores.view(-1, bb_scores.shape[2])
        proposal_density = data['proposal_density'].view(-1, data['proposal_density'].shape[2])
        gt_density = data['gt_density'].view(-1, data['gt_density'].shape[2])

        # Compute loss
        loss = self.objective(bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': loss.item()}

        return loss, stats

class AtomDistillationActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM with basic distillation"""
    def __init__(self, student_net, teacher_net, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.student_net = student_net
        self.teacher_net = teacher_net
        self.objective = objective

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_s, ref_feats_s, test_feats_s = self.student_net(data['train_images'], 
                                                            data['test_images'], 
                                                            data['train_anno'], 
                                                            data['test_proposals'],
                                                            mode='train')
        iou_t, ref_feats_t, test_feats_t = self.teacher_net(data['train_images'], 
                                                            data['test_images'], 
                                                            data['train_anno'], 
                                                            data['test_proposals'],
                                                            mode='train')

        # get target boxes for TRloss
        num_sequences = data['train_images'].shape[-4]
        num_train_images = data['train_images'].shape[0] if data['train_images'].dim() == 5 else 1
        target_bb = data['train_anno'].reshape(num_train_images, num_sequences, 4)

        iou_s = iou_s.view(-1, iou_s.shape[2])
        iou_t = iou_t.view(-1, iou_t.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        iou = {'iou_student': iou_s, 'iou_teacher': iou_t, 'iou_gt': iou_gt}
        features = {'ref_feats_s': ref_feats_s, 
                    'test_feats_s': test_feats_s, 
                    'ref_feats_t': ref_feats_t, 
                    'test_feats_t': test_feats_t, 
                    'target_bb': target_bb}

        # Compute loss
        loss = self.objective(iou, features)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.student_net.to(device)
        self.teacher_net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.student_net.train(mode)
        self.teacher_net.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)