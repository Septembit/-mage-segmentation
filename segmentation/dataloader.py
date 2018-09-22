import torch
import os
import torch.utils.data as data
import numpy as np
import random
from PIL import Image
def datalist(phase) :
        Image = []
        Label = []
        dataFile ="/home/yachao-li/Downloads/pycharm/lists/100k/drivable/"+phase+"_images.txt"
        labelFile = "/home/yachao-li/Downloads/pycharm/lists/100k/drivable/"+phase+"_labels.txt"
        with open(dataFile,"r") as f:
            data = f.readlines()
            for line in data:
                Image.append(os.path.join("/home/yachao-li/Downloads/bdd100k_images/bdd100k",
                                               line.split("\n")[0]))
        with open(labelFile,"r") as f:
            data = f.readlines()
            for line in data:
                Label.append(os.path.join("/home/yachao-li/Downloads/bdd100k_drivable_maps/bdd100k",
                                               line.split("\n")[0]))
        return Image, Label


def randomCrop(image,target, output_size):


    h, w = output_size
    height, width= image.size

    # print(h,w,height,width)

    i = random.randint(0, height - h)
    j = random.randint(0, width - w)

    new_img = image.crop((i, j, h + i, w + j))
    new_tar = target.crop((i, j, h + i, w + j))


    return new_img, new_tar

class dataloader(data.Dataset):

    def __init__(self,phase):
        self.List = datalist(phase)[0]
        self.Label = datalist(phase)[1]
        print("# of "+phase+" samples:", len(self.List))

    def __getitem__(self, index):
        img_path_list = self.List[index]
        label_path_list = self.Label[index]


        #temImg = np.array(Image.open(img_path_list)).astype(np.float32) / 255.0
        #temLab = np.array(Image.open(label_path_list)).astype(np.int32)
        temImg = Image.open(img_path_list).resize((325,185),Image.ANTIALIAS)
        temLab = Image.open(label_path_list).resize((325,185),Image.ANTIALIAS)
        temImg, temLab = randomCrop(temImg, temLab, (320, 180))

        temImg = np.array(temImg).astype(np.float32)

        temLab = np.array(temLab).astype(np.int32)



        image=torch.from_numpy(temImg.transpose((2,0,1)) / 255.0)
        label=torch.from_numpy(temLab).long()
        #print(image.size())
        #print(label.size())
        return  image,label

    def __len__(self):
        return len(self.List)

def dataTestlist():
        Image = []

        dataFile = "/home/yachao-li/Downloads/test.txt"
        with open(dataFile, "r") as f:
            data = f.readlines()
            for line in data:
                Image.append("/home/yachao-li/Downloads/bdd100k_images/bdd100k/images/100k/test/"+
                                          line.split("\n")[0]+".jpg")

        return Image


class testLoader(data.Dataset):

    def __init__(self):
        self.List = dataTestlist()

        print("# of test samples:", len(self.List))

    def __getitem__(self, index):
        img_path_list = self.List[index]



        #temImg = np.array(Image.open(img_path_list)).astype(np.float32) / 255.0
        #temLab = np.array(Image.open(label_path_list)).astype(np.int32)

        temImg = Image.open(img_path_list).resize((320, 180), Image.ANTIALIAS)

        temImg = np.array(temImg).astype(np.float32)



        image=torch.from_numpy(temImg.transpose((2,0,1))/ 255.0 )
        #print(image.size())
        #print(label.size())
        return  image

    def __len__(self):
        return len(self.List)