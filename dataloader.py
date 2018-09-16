import torch
import os
import cv2
import glob
import torch.utils.data as data
import numpy as np
import random
from PIL import Image
def dataTrainlist() :
        trainImage = []
        trainLabel = []
        dataFile ="lists/100k/drivable/train_images.txt"
        labelFile = "lists/100k/drivable/train_labels.txt"
        with open(dataFile,"r") as f:
            data = f.readlines()
            for line in data:
                trainImage.append(os.path.join("/home/yachao-li/Downloads/bdd100k_images/bdd100k",
                                               line.split("\n")[0]))
        with open(labelFile,"r") as f:
            data = f.readlines()
            for line in data:
                trainLabel.append(os.path.join("/home/yachao-li/Downloads/bdd100k_drivable_maps/bdd100k",
                                               line.split("\n")[0]))
        return trainImage, trainLabel


def randomCrop(image,target, output_size):


    h, w = output_size
    height, width= image.size

    # print(h,w,height,width)

    i = random.randint(0, height - h)
    j = random.randint(0, width - w)

    new_img = image.crop((i, j, h + i, w + j))
    new_tar = target.crop((i, j, h + i, w + j))


    return new_img, new_tar

class dataTrainloader(data.Dataset):

    def __init__(self):
        self.trainList = dataTrainlist()[0]
        self.trainLabel = dataTrainlist()[1]
        print("# of training samples:", len(self.trainList))

    def __getitem__(self, index):
        img_path_list = self.trainList[index]
        label_path_list = self.trainLabel[index]


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
        return len(self.trainList)