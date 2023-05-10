import PIL
from PIL import Image
import os
from torchvision import transforms as T

image_size=256
transform=T.Compose([
                T.Resize(image_size, interpolation=PIL.Image.LANCZOS),
                T.RandomCrop((image_size, image_size)),
            ])
transform2=T.Compose([
                T.Resize((image_size,image_size), interpolation=PIL.Image.LANCZOS),
                T.RandomCrop((image_size, image_size)),
            ])
def generate_crop(path1,path2):
    image=Image.open(path1)
    image = image.convert('RGB')
    image = transform(image)
    image.save(path2)
def generate_resize(path1,path2):
    image=Image.open(path1)
    image = image.convert('RGB')
    image = transform2(image)
    image.save(path2)
def generate_total_crop(dir1,dir2):
    assert os.path.exists(dir1)
    assert os.path.exists(dir2)
    for image_file_name in os.listdir(dir1):
        image_path = os.path.join(dir1, image_file_name)
        save_dir=os.path.join(dir2,image_file_name[:-4])
        generate_resize(image_path,os.path.join(save_dir,'complete_resize.jpg'))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        try:
            for i in range(15):
                save_path=os.path.join(save_dir,str(i)+'.jpg')
                generate_crop(image_path, save_path)

        except:
            print(image_path,"error")
            pass
def test1():
    path1=r'C:\Users\赵鑫\Desktop\final\buchong_shiyan\完整数据\coco_random_crop\origin\29.jpg'
    path2=r'C:\Users\赵鑫\Desktop\final\buchong_shiyan\完整数据\coco_random_crop\crop\29\1.jpg'
    generate_crop(path1,path2)


if __name__ == '__main__':
    dir1 = r'C:\Users\赵鑫\Desktop\final\buchong_shiyan\完整数据\coco_random_crop\origin'
    dir2 = r'C:\Users\赵鑫\Desktop\final\buchong_shiyan\完整数据\coco_random_crop\crop'
    generate_total_crop(dir1,dir2)