import torch
import torch.optim
import matplotlib.pyplot as plt
import tqdm

import pyoptim.sdsloss as sds
import pyoptim.helpers as H


def cvt(img):
    img_lim = img.clamp(0, 1)
    return img_lim

def sdstest():
    token = H.read_token("TOKEN")
    sds_loss = sds.SDSLoss("cuda", "cute cat", token)
    dat_augs = sds.get_data_augs(350)

    img = torch.rand((400, 400, 3)).cuda()
    img.requires_grad = True

    img_np = cvt(img).cpu().detach().numpy()
    plt.imshow(img_np, cmap="gray")
    plt.savefig("test_0.png")

    opt = torch.optim.Adam([img], lr=1e-6)

    pbar = tqdm.trange(5000, dynamic_ncols=True)

    for _ in pbar:
        opt.zero_grad()

        img_unsq = cvt(img).permute(2, 0, 1).unsqueeze(0)
        x_aug = dat_augs(img_unsq)
        loss = 100 * sds_loss(x_aug)
        # print(f'it {it}: {loss.item()}')
        
        loss.backward()
        pbar.set_postfix_str(f"Loss: {loss.item():.6f}")
        opt.step()

    img = cvt(img)
    img_np = img.cpu().detach().numpy()
    print(img)
    plt.imshow(img_np, cmap="gray")
    plt.savefig("test.png")


if __name__ == "__main__":
    sdstest()

    # import matplotlib.pyplot as plt
    # from diffusers import StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained("models/sds/", torch_dtype=torch.float16).to("cuda")
    
    # imgs = pipe("matterhorn").images

    # print(len(imgs))
    # for i, img in enumerate(imgs):
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.savefig(f"test{i}")