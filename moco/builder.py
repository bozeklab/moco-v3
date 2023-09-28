# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from unetr_vits import unetr_vit_base_patch16, cell_vit_base_patch16
from util.misc import LayerNorm
from util.pos_embed import interpolate_pos_embed


def _prepare_model(chkpt_dir_vit, **kwargs):
    # build ViT encoder
    num_nuclei_classes = kwargs.pop('num_nuclei_classes')
    num_tissue_classes = kwargs.pop('num_tissue_classes')
    embed_dim = kwargs.pop('embed_dim')
    extract_layers = kwargs.pop('extract_layers')
    drop_rate = kwargs['drop_path_rate']

    vit_encoder = unetr_vit_base_patch16(num_classes=num_tissue_classes, **kwargs)

    # load ViT model
    checkpoint = torch.load(chkpt_dir_vit, map_location='cpu')

    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(vit_encoder, checkpoint_model)

    msg = vit_encoder.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    model = cell_vit_base_patch16(num_nuclei_classes=num_nuclei_classes,
                                  embed_dim=embed_dim,
                                  extract_layers=extract_layers,
                                  drop_rate=drop_rate,
                                  encoder=vit_encoder)
    return model


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=768, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = _prepare_model(chkpt_dir_vit='/data/pwojcik/moco-v3/encoder-1600.pth',
                                           init_values=None,
                                           drop_path_rate=0.1,
                                           num_nuclei_classes=6,
                                           num_tissue_classes=19,
                                           embed_dim=768,
                                           extract_layers=[3, 6, 9, 12])

        self.momentum_encoder = _prepare_model(chkpt_dir_vit='/data/pwojcik/moco-v3/encoder-1600.pth',
                                               init_values=None,
                                               drop_path_rate=0.1,
                                               num_nuclei_classes=6,
                                               num_tissue_classes=19,
                                               embed_dim=768,
                                               extract_layers=[3, 6, 9, 12])

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        self.base_encoder.freeze_encoder()

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def compute_unigrad_loss(self, pred, target, neg_weight=0.02):
        #pred = self.student_norm(pred)
        with torch.no_grad():
            target = self.teacher_norm(target)

        dense_pred = pred.reshape(-1, pred.shape[-1])
        dense_target = target.reshape(-1, target.shape[-1])

        # compute pos term
        pos_term = ((dense_pred - dense_target) ** 2).sum(-1).mean()

        # compute neg term
        correlation = (dense_target.T @ dense_target) / dense_target.shape[0]
        torch.distributed.all_reduce(correlation)
        correlation = correlation / torch.distributed.get_world_size()

        neg_term = torch.diagonal(dense_pred @ correlation @ dense_pred.T).mean()

        loss = (pos_term + neg_weight * neg_term) / pred.shape[-1]

        return loss

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, boxes1, boxes2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        boxes2 = boxes2.detach()
        mask1 = torch.all(boxes1 != -1, dim=-1)
        mask2 = torch.all(boxes2 != -1, dim=-1)
        mask2 = mask2.detach()
        mask = torch.logical_and(mask1, mask2)

        q1 = self.predictor(self.base_encoder(x1, boxes1, mask))
        q2 = self.predictor(self.base_encoder(x2, boxes2, mask))

        with torch.no_grad():  # no gradient
            _mask = mask.clone()

            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1, boxes1, _mask)
            k2 = self.momentum_encoder(x2, boxes2, _mask)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, hidden_dim=768):
        #self.base_encoder.head.weight.shape[1]
        #del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
