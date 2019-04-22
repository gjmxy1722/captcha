from PIL import Image
import os

path = r"C:/Users/jwt/Desktop/media/success/"  # 文件夹目录
save_dir = r"C:/Users/jwt/Desktop/img/test/"
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for file in files:  # 遍历文件夹
    if file.endswith('.png') :
        print(file)
        filename=path+file
        image = Image.open(filename)
    try:
        image.save(save_dir+file.split('_')[1].split('.')[0]+'.png')
    except:
        continue