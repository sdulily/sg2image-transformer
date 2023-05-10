import os
def paser_file_path(dir):
    paths=[]
    for child_dir in os.listdir(dir):
        filenames=os.listdir(os.path.join(dir,child_dir))
        for filename in filenames:

            path=os.path.join(child_dir,filename)
            paths.append(path)

    return paths
def create_text_file(dir,text_path):
    text=paser_file_path(dir)
    with open(text_path,'w') as fp:
        fp.writelines([line + '\n' for line in text])

def create_file_list_text(dir,text_path):
    text = paser_dir_files(dir)
    with open(text_path, 'w') as fp:
        fp.writelines([line + '\n' for line in text])
def paser_dir_files(dir):
    paths = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        paths.append(path)
        pass
    return paths
if __name__ == '__main__':
    # dir=r'/mnt/dataset/giraffe/trainA'
    # text_path=r'/mnt/dataset/giraffe/trainA.txt'
    # create_giraffe(dir,text_path)

    dir = r'/mnt/dataset/font_data/standard_style_train'
    text_path = r'/mnt/dataset/font_data/standard_style_train.txt'
    create_file_list_text(dir, text_path)

    dir = r'/mnt/dataset/font_data/luxun_style_train'
    text_path = r'/mnt/dataset/font_data/luxun_style_train.txt'
    create_file_list_text(dir, text_path)

    dir = r'/mnt/dataset/font_data/lishu_style_train'
    text_path = r'/mnt/dataset/font_data/lishu_style_train.txt'
    create_file_list_text(dir, text_path)

    dir = r'/mnt/dataset/font_data/standard_style_test'
    text_path = r'/mnt/dataset/font_data/standard_style_test.txt'
    create_file_list_text(dir, text_path)

    dir = r'/mnt/dataset/font_data/luxun_style_test'
    text_path = r'/mnt/dataset/font_data/luxun_style_test.txt'
    create_file_list_text(dir, text_path)

    dir = r'/mnt/dataset/font_data/lishu_style_test'
    text_path = r'/mnt/dataset/font_data/lishu_style_test.txt'
    create_file_list_text(dir, text_path)
    pass