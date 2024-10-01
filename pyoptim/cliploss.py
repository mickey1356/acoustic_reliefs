import numpy as np
import torch
import collections
from PIL import Image
from torchvision import transforms
from torch import nn
import clip

# from https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/sketch_utils.py#L279
def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im

# from https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/painterly_rendering.py#L45
def get_target(path, device, fix_scale=False, image_scale=224):
    target = Image.open(path)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    if fix_scale:
        target = fix_image_scale(target)

    transforms_ = []
    transforms_.append(transforms.Resize(
        image_scale, interpolation=transforms.InterpolationMode.BICUBIC))
    transforms_.append(transforms.CenterCrop(image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)

    target_ = data_transforms(target).unsqueeze(0).to(device)
    return target_


# from https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/models/loss.py#L407
class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None
        self.device = device
        self.n_channels = 3
        self.kernel_h = 32
        self.kernel_w = 32
        self.step = 32
        self.num_patches = 49

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x, mode="train"):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        # fc_features = self.clip_model.encode_image(x, attn_map, mode).float()
        # Each featuremap is in shape (5,50,768) - 5 is the batchsize(augment), 50 is cls + 49 patches, 768 is the dimension of the features
        # for each k (each of the 12 layers) we only take the vectors
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


# from https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/models/loss.py#L511
def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]

# from https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/models/loss.py#L516
def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]

# from https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/models/loss.py#L521
def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    # print(xs_conv_features[0].shape, torch.cosine_similarity(xs_conv_features[0], ys_conv_features[0], dim=1).shape, torch.cosine_similarity(xs_conv_features[0], ys_conv_features[0], dim=-1).shape)
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=-1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


# from https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/models/loss.py#L530
class CLIPConvLoss(torch.nn.Module):
    def __init__(self, device, clip_model_name="ViT-B/32", num_augs=4, clip_conv_layer=11, clip_fc_loss_weight=None):
        # mask is a binary tensor with shape (1,3,224,224)
        super(CLIPConvLoss, self).__init__()
        self.device = device

        self.clip_model_name = clip_model_name
        assert self.clip_model_name in [
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = "L2"  # args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.loss_log_name = "vit"
            self.visual_encoder = CLIPVisualEncoder(self.model, self.device)
        else:
            assert False, "ResNet-based models are not supported in this code"

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()

        self.num_augs = num_augs  # args.num_aug_clip

        augemntations = []
        if self.num_augs > 0:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
            # augemntations.append(transforms.RandomResizedCrop(
                # 224, scale=(0.4, 0.9), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_loss_weight = clip_fc_loss_weight
        self.clip_conv_layer = clip_conv_layer
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """

        conv_loss_dict = {}

        x = sketch.to(self.device)
        y = target.to(self.device)

        sketch_augs, img_augs = [self.normalize_transform(x)], [self.normalize_transform(y)]

        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)
        # print("================================")
        # print(xs.requires_grad, ys.requires_grad)
        # sketch_utils.plot_batch(xs, ys, f"{self.args.output_dir}/jpg_logs", self.counter, use_wandb=False, title="fc_aug{}_iter{}_{}.jpg".format(1, self.counter, mode))

        xs_fc_features, xs_conv_features = self.visual_encoder(xs, mode=mode)
        ys_fc_features, ys_conv_features = self.visual_encoder(ys, mode=mode)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        conv_loss_dict[f"clip_{self.loss_log_name}_l{self.clip_conv_layer}"] = conv_loss[self.clip_conv_layer]

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            # fc_loss = torch.nn.functional.mse_loss(xs_fc_features, ys_fc_features).mean()
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features, ys_fc_features, dim=1)).mean()
            conv_loss_dict[f"fc_{self.loss_log_name}"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict
