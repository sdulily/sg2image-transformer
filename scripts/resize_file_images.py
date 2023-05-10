import os
from PIL import Image
import argparse

def resize_imgs(origin_file, target_file, width, height):
    if not os.path.exists(target_file):
        os.mkdir(target_file)

    for img_name in os.listdir(origin_file):
        print(img_name)
        img = (Image.open(os.path.join(origin_file,img_name)))
        img = img.resize((width, height), Image.ANTIALIAS)
        img.save(os.path.join(target_file,img_name))

def auto_resize_small(origin_dir):
    target_dir_64_pix=origin_dir+'pix64'
    target_dir_128_pix=origin_dir+'pix128'
    resize_imgs(origin_dir,target_dir_64_pix,64,64)
    resize_imgs(origin_dir, target_dir_128_pix, 128, 128)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        type=str,
    )
    args=parser.parse_args()
    dir=args.r
    # dir1=r'C:\Users\Administrator\Desktop\实验记录\sg_transformer\zuihao\sample'
    # dir2=r'C:\Users\Administrator\Desktop\实验记录\sg_transformer\zuihao\samplepix128'
    # dir3=r'C:\Users\Administrator\Desktop\实验记录\sg_transformer\zuihao\samplepix64'
    auto_resize_small(dir)
