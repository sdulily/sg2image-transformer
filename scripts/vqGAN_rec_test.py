import torch
import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel,GumbelVQ
from PIL import Image
import io
import os, sys
import PIL
from PIL import ImageDraw, ImageFont
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
#environment prepare
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

def load_config(config_path,display=False):
    config=OmegaConf.load(config_path)
    if display:
        container=OmegaConf.to_container(config)
        container_str=yaml.dump(container)
        print(container_str)
    return config

def load_vqgan(config,ckpt_path=None,is_gumble=False):
    if is_gumble:
        model=GumbelVQ(**config.model.params)
    else:
        model=VQModel(**config.model.params)
    if ckpt_path is not None:
        sd=torch.load(ckpt_path,map_location="cpu")["state_dict"]
        missing,unexpected=model.load_state_dict(sd,strict=False)
    return model.eval()

def preprocess_vqgan(x):
    x=2.*x - 1.
    return x

def reconstruct_with_vqgan(x,model):
    z,_,[_,_,indices]=model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    xrec=model.decode(z)
    return xrec


def preprocess(img, target_image_size=256):
    s = min(img.size)
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img
def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x
def reconstruction_pipeline(img_path,model1024):

    img_origin=Image.open(img_path)
    #img_origin.show()
    img_normal=preprocess(img_origin)

    img_normal=img_normal.to(DEVICE)
    img_normal = preprocess_vqgan(img_normal)
    rec_image_tensor=reconstruct_with_vqgan(img_normal,model1024)
    rec_image=custom_to_pil(rec_image_tensor[0])
    #rec_image.show()
    pass
if __name__ == '__main__':


    dir_path = r'C:\Users\赵鑫\Desktop\1'
    output_path=r'C:\Users\赵鑫\Desktop\2'
    img_paths = [dir_path + file_name for file_name in os.listdir(dir_path)]

    config1024 = load_config(r"C:\Users\赵鑫\Desktop\final\buchong_shiyan\2023-02-19--16-02-06_custom_vqgan_vg\configs\2023-02-19--16-02-06-project.yaml", display=True)
    model1024 = load_vqgan(config1024, ckpt_path=r"C:\Users\赵鑫\Desktop\final\buchong_shiyan\2023-02-19--16-02-06_custom_vqgan_vg\checkpoints\last.ckpt").to(DEVICE)
    target_image_size=256
    transformer=T.Compose([
        T.Resize((target_image_size,target_image_size),interpolation=PIL.Image.LANCZOS),
        T.ToTensor()
    ])
    for file_name in os.listdir(dir_path):
        origin_img_path=os.path.join(dir_path,file_name)
        #resize_img_path=dir_path+'resize_'+file_name
        rec_img_path=os.path.join(output_path,file_name)

        img_origin = Image.open(origin_img_path)

        img_resize=transformer(img_origin).to(device=DEVICE)
        #save_image(img_resize,resize_img_path)

        x = torch.unsqueeze(img_resize, 0)


        x = preprocess_vqgan(x)
        z, _, [_, _, indices] = model1024.encode(x)
        xrec = model1024.decode(z)
        rec_image = custom_to_pil(xrec[0])
        rec_image.save(rec_img_path)



