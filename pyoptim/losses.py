import collections

import torch
from torch import nn
from torchvision import transforms

import clip
import lpips


# Adapted from https://github.com/yael-vinker/CLIPasso/blob/92262f702b6592b6c25c80def6284ad06225eadd/models/loss.py
class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None
        for i in range(12):
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(self.make_hook(i))

    def make_hook(self, i):
        def hook(_module, _input, output):
            if len(output.shape) == 3:
                self.featuremaps[i] = output.permute(1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[i] = output
        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]
        return fc_features, featuremaps


class ImgImgCLIPLoss(nn.Module):
    def __init__(self, device="cuda", geom_layers=[3], n_augs=4, sem_wt=0):
        super().__init__()
        self.device = device
        self.geom_layers = geom_layers
        self.n_augs = n_augs
        self.sem_wt = sem_wt

        self.model, prep = clip.load("ViT-B/32", device, jit=False, download_root="models/clip/")
        self.visual_encoder = CLIPVisualEncoder(self.model)
        self.model.eval()

        self.normalizers = transforms.Compose([
            prep.transforms[0],  # Resize
            prep.transforms[1],  # CenterCrop
            prep.transforms[-1], # Normalize
        ])

        augs = []
        augs.append(transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
        augs.append(transforms.RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augs.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augs = transforms.Compose(augs)

    def forward(self, img, tgt):
        x = img.float().to(self.device)
        y = tgt.float().to(self.device)

        x_norm = self.normalizers(x)
        y_norm = self.normalizers(y)

        x_augs = [x_norm]
        y_augs = [y_norm]
        for _ in range(self.n_augs):
            aug_pair = self.augs(torch.cat([x, y]))
            x_augs.append(aug_pair[0].unsqueeze(0))
            y_augs.append(aug_pair[1].unsqueeze(0))
        
        xs = torch.cat(x_augs, dim=0).to(self.device)
        ys = torch.cat(y_augs, dim=0).to(self.device)

        xs_sem_feats, xs_geom_feats = self.visual_encoder(xs)
        ys_sem_feats, ys_geom_feats = self.visual_encoder(ys)

        sem_loss = (1 - torch.cosine_similarity(xs_sem_feats, ys_sem_feats, dim=-1)).mean()

        geom_loss = 0
        for geom_layer in self.geom_layers:
            geom_loss += torch.square(xs_geom_feats[geom_layer] - ys_geom_feats[geom_layer]).mean()
        
        return geom_loss + self.sem_wt * sem_loss


class ImgImgLPIPSLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.lpip = lpips.LPIPS(net="vgg").to(device)
        self.device = device

    def forward(self, img, tgt):
        x = img.float().to(self.device)
        y = tgt.float().to(self.device)

        return self.lpip(x, y)