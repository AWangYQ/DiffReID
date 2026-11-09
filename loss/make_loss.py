# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch
from loss.supcontrast import SupConLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    contras_loss = SupConLoss("cuda")
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, i2tscore = None, layer=None, num_feat=10):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:2]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:3]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    concat_feat =torch.cat((feat[2], feat[3]), 0)
                    concat_target = torch.cat((target, target.unsqueeze(1).repeat(1, num_feat).reshape(-1)))
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    repeat_target = target.unsqueeze(1).repeat(1, num_feat).reshape(-1)
                    # repeat_target = target
                    denoise_triplet_loss = [triplet(feats, repeat_target)[0] for feats in feat[4:]]
                    denoise_triplet_loss = sum(denoise_triplet_loss) / layer
                    loss = loss + xent(score[2], target) + triplet(concat_feat, concat_target)[0] + denoise_triplet_loss

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        
                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


