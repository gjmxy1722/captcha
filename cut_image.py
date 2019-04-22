from PIL import Image
import numpy as np
import cv2
import os
from tkinter import _flatten


# 观察过本验证码图片集，可知所有的验证码的干扰线都是水平或者向下倾斜的，验证码较细
# 找到干扰线

def get_index(mylist):
    mystr = ''.join(map(str, mylist))  # 先转化为字符串
    index = []
    result0 = list(map(lambda x: len(x), [j for j in mystr.split('0') if len(j) > 0]))
    number = list(map(lambda x: len(x), [j for j in mystr.split('1') if len(j) > 0]))
    end = 0
    for i in range(len(number)):
        if (mylist[0]==0) and i==0:
            index.append(list(np.arange(number[i])))
            end += number[i]
        elif (mylist[0]==0) and i>0:
            start = end
            end = start + number[i]
            a = list(np.arange(start, end))
            index.append(a)
        else:
            start = result0[i] + end
            end = start + number[i]
            a = list(np.arange(start, end))
            index.append(a)
    return index

    
def find_near_arr(arr1,arr2):
    for item in arr2:
        if max(arr1)+1>=min(item) and max(arr1)<=max(item) and min(arr1)<=min(item):
            return item

                
def del_index(check):
    delete_index = []
    if len(check)==1:
        delete_index = check
    else:
        for i in range(len(check)-1):
            differ_list = sorted(list(set(check[i]).difference(set(check[i+1]))))
            if differ_list:  
                delete_index.append(differ_list)
            elif check[i]==check[i+1]:
                interval = int(len(check[i])/check.count(check[i]))
                for m in range(check.count(check[i])-1):                                    
                    delete_index.append(check[i][m*interval:(m+1)*interval])
                delete_index.append(check[i][(check.count(check[i])-1)*interval:])
            else:            
                delete_index.append(check[i])
                check[i+1] = sorted(list(set(check[i+1]).difference(set(check[i]))))
                
        delete_index.append(check[len(check)-1])
    return delete_index

    
def find_interfer_line(im):
    data = np.array(im.getdata()).reshape((23, 60))
    index_array = []
    for i in range(23):
        index_array.append(get_index([int(item / 255) for item in data[i, :]]))
    
    result = []
    index = []

    for i in range(22):
        if len(index_array[i])>0:
            for j in range(len(index_array[i])):
                k = i+1
                result1 = []
                result1.append(index_array[i][j])
                head = index_array[i][j]
                tag = index_array[k]

                while (k <= 22):
                    content = find_near_arr(head, tag)
                    if (content is not None) and sum([set(content).issubset(item) for item in result1])<len(content):
                        index_array[k].remove(content)
                        result1.append(content)
                        head = content
                        k += 1
                        if k<=22:
                            tag = index_array[k]
                    else:
                        break
                if len(list(set(_flatten(result1))))>13:
                    # print(i,result1)
                    index.append(i)
                    result.append(del_index(result1))
                    
    return index,result
 


def delete_line(img):
    data = img.getdata()
    data = [item for item in data]
    index_list,delete_index = find_interfer_line(img)
    # print(index_list,delete_index)
    w, h = img.size
     
    for i in range(len(index_list)):
        for j in range(len(delete_index[i])):
            b = index_list[i] + j
            for k in range(len(delete_index[i][j])):
                a = delete_index[i][j][k]
                if a>0 and a<w-1 and b>0 and b<h-1:
                    top_pixel = data[w * (b - 1) + a]==0
                    left_pixel = data[w * b + (a - 1)]==0
                    down_pixel = data[w * (b + 1) + a]==0
                    right_pixel = data[w * b + (a + 1)]==0

                    angle1 = data[w * (b - 1) + a - 1]==0
                    angle2 = data[w * (b - 1) + (a + 1)]==0
                    angle3 = data[w * (b + 1) + a - 1]==0
                    angle4 = data[w * (b + 1) + (a + 1)]==0

                    point_number = sum([top_pixel,left_pixel,down_pixel,right_pixel,angle1,angle2,angle3,angle4])

                    if (point_number <= 4) and ((not angle3) or (not angle2)):
                        img.putpixel((delete_index[i][j][k], index_list[i] + j), 255)

    return img                  
                
def find_cut_location(labels, stats, index_ori):
    # 对连通区域进行处理
    # 将符合条件的选出来：1、16行以下的为false 2、最后一列数字少于20个的不要
    logit_value = np.multiply(np.multiply(stats[:,4]>=20,stats[:,1]<16),stats[:,3]>7)
    index_ori = [item for item in index_ori if logit_value[item]]
    cut_loc = []
    label_re = []
    
    
    # 如果没有黏连
    # 五个连通区域
    if sum(logit_value)>=6:
        choose_index = index_ori[1:6]
        start_point = list(stats[choose_index,0])
        start_index = list(np.argsort(stats[choose_index,0]))
        differ = [start_point[start_index[i+1]]-start_point[start_index[i]] for i in range(len(start_point)-1)]
        differ2 = [differ[i+1]+differ[i] for i in range(len(differ)-1)]
        combine_point = differ2.index(min(differ2))+1
        
        for i in range(5):
            member1 = start_point[start_index[i]]
            member2 = choose_index[start_point.index(member1)]
            member3 = min(member1 + min(max(stats[member2,2],13),16),60)
            cut_loc.append([member1,member3])
            label_re.append([member2])  
        
        cut_loc.remove(cut_loc[combine_point])
        label_re[combine_point-1].append(label_re[combine_point][0])
        label_re.remove(label_re[combine_point])
        
        # print("combine_point",combine_point)
        # print(start_point)
        # print(label_re)
        
   # 四个区域
    elif sum(logit_value)==5:
        choose_index = index_ori[1:5]
        start_point = list(stats[choose_index,0])
        start_index = list(np.argsort(stats[choose_index,0]))
        # print(start_point)
        # print(start_index)
       
        for i in range(4):
            member1 = start_point[start_index[i]]
            member2 = choose_index[start_point.index(member1)]
            member3 = min(member1 + min(max(stats[member2,2],13),16),60)
            cut_loc.append([member1,member3])
            label_re.append([member2])  
    
    # 两个黏在一起
    elif sum(logit_value)==4:
        counts_column = []
        index_column = np.arange(stats[index_ori[1],0],stats[index_ori[1],0]+stats[index_ori[1],2])
        for i in index_column:
            counts_column.append(list(labels[:,i]).count(index_ori[1]))       
        # print(counts_column)
        # print(index_column)
        
        choose_index = index_ori[1:4]
        start_point = list(stats[choose_index,0])
        start_index = list(np.argsort(stats[choose_index,0]))
        # print("start_point:",start_point)
        # print("start_index:",start_index)
        
        for i in range(3):
            if start_index[i]==0:
                if i>0:
                    # print(cut_loc[i-1][1],start_point[start_index[i]])
                    member1 = np.max([cut_loc[i-1][1],start_point[start_index[i]]])
                else:
                    member1 = start_point[start_index[i]]      
            
                member2 = choose_index[start_point.index(start_point[start_index[i]])]
                
                if i<2:
                    inter_var = start_point[start_index[i+1]]-start_point[start_index[i]]          
                else:
                    inter_var = 60 - start_point[start_index[i]]
                        
                member3 = np.min([member1 + max(int((inter_var+1)/2),14),60])    
                
                cut_loc.append([member1,member3])
                label_re.append([member2])
                
                cut_loc.append([member3-1,np.min([member3+int((inter_var+1)/2),60])])
                label_re.append([member2])
                    
                 
            else:
                member1 = start_point[start_index[i]]
                member2 = choose_index[start_point.index(member1)]
                member3 = min(member1 + max(stats[member2,2],14),60)
                cut_loc.append([member1,member3])
                label_re.append([member2])
        
        
    # 三个黏在一起
    elif sum(logit_value)==3:
        choose_index = index_ori[1:3]
        start_point = list(stats[choose_index,0])
        start_index = list(np.argsort(stats[choose_index,0]))
        # print(stats[choose_index[start_point.index(start_point[start_index[1]])],2])
        if (stats[choose_index[start_point.index(start_point[start_index[0]])],2]>=20) and (stats[choose_index[start_point.index(start_point[start_index[1]])],2]>=20):
            for i in range(2):
                member1 = start_point[start_index[i]]
                member2 = choose_index[i]
                for j in range(2):
                    member3 = min(member1 + min(max((j+1)*int(stats[member2,2]/2),13),15),60)
                    cut_loc.append([member1,member3])
                    label_re.append([member2])
                    member1 = member3-2
        
        # 一三       
        else:
            for i in range(2):
                if start_index[i]==0:
                    if i>0:
                        # print(cut_loc[i-1][1],start_point[start_index[i]])
                        member1 = np.max([cut_loc[i-1][1],start_point[start_index[i]]])
                    else:
                        member1 = start_point[start_index[i]]      
                
                    member2 = choose_index[start_point.index(start_point[start_index[i]])]
                    
                    if i<1:
                        inter_var = start_point[start_index[i+1]]-start_point[start_index[i]]          
                    else:
                        inter_var = 60 - start_point[start_index[i]]
                            
                    member3 = np.min([member1 + max(int((inter_var+1)/3),14),60])    
                    
                    cut_loc.append([member1,member3])
                    label_re.append([member2])
                    
                    cut_loc.append([member3-1,np.min([member3+int((inter_var+1)/3),60])])
                    label_re.append([member2])
                    
                    cut_loc.append([member3-1,np.min([member3+2*int((inter_var+1)/3),60])])
                    label_re.append([member2])
                    
                else:
                    member1 = start_point[start_index[i]]
                    member2 = choose_index[start_point.index(member1)]
                    member3 = min(member1 + max(stats[member2,2],14),60)
                    cut_loc.append([member1,member3])
                    label_re.append([member2])


    # 都黏在一起
    else:
        choose_index=[]
        choose_index.append(index_ori[1])
        member1 = stats[choose_index[0],0]
        member2 = choose_index[0]
        for i in range(4):
            member3 = min(member1 + min(max((i+1)*int(stats[member2,2]/4),13),15),60)
            cut_loc.append([member1,member3])
            label_re.append([member2])
            member1 = member3-2

    return cut_loc,label_re       


def two_value(image,filename,cut_image_save_dir):
    # 灰度图
    lim = image.convert('L')
    original_data = image.getdata()
    # 灰度阈值设为160，低于这个值的点全部填白色
    threshold = 170
    table = []

    for j in range(256):
        if j < threshold:
            table.append(0)
        else:
            table.append(1)

    bim = lim.point(table, '1')
    
    for i in range(23):
        for j in range(60):
            bim.putpixel((j, i), 255*bim.getdata()[i*60+j])
         
    bim_copy = bim.copy()
    imag = delete_line(bim_copy)
    # 获取连通域
    data = imag.getdata()
    data = np.matrix(data)
    data = [1-item/255 for item in data]
    data = np.array(np.reshape(data,(23,60)), np.uint8)
    # print(type(data))
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(data)
    # print(labels)
    # print(np.amax(labels))
    
    
    # 将分类显示出来
    # for i in range(23):
        # print()
        # for j in range(60):
            # print(labels[i,j],end="")
    # print()
    
    
    
    # 对连通域中含有的像素点个数进行排序
    # print(stats)
    # print(stats[:,4])
    conect_counts = np.argsort(stats[:,4])[::-1]
    # print(conect_counts)
    
    
    #将四个连通区域分离开来
    cut_location,label = find_cut_location(labels, stats, conect_counts)
    # print(cut_location,label)
    h = 23

    if cut_location:
        cut_result=[]
        for i in range(len(cut_location)):
            image = imag.copy()
            box = (cut_location[i][0], 0, cut_location[i][1], h)
            for j in range(60):
                for k in range(23):
                    if (labels[k,j] not in label[i]):
                        image.putpixel((j, k), 255)

            if filename.split('/')[6][i] in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']:
                dirs = cut_image_save_dir + filename.split('/')[6][i]+'_'
            else:
                dirs = cut_image_save_dir + filename.split('/')[6][i]

            if not os.path.exists(dirs):
                os.makedirs(dirs)
                    
            if len(label[i])>1:
                bim.crop(box).resize((15,23)).save(dirs+'/'+filename.split('/')[6][:4]+'_'+str(i) + ".png")
            else:
                imag.crop(box).resize((15, 23)).save(dirs+'/'+filename.split('/')[6][:4]+'_'+str(i) + ".png")


if __name__=="__main__":
    path = r"C:/Users/jwt/Desktop/img/test/"  # 文件夹目录
    cut_image_dir = r"C:/Users/jwt/Desktop/captcha_code_new/cut_image_new/"
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        print(file)
        filename=path+file
        image = Image.open(filename)
        two_value(image,filename,cut_image_dir)


    
    
    
    
    



    
