
import dataloader

import torch

dataFeeder = dataloader.dataTrainloader()
trainData = torch.utils.data.DataLoader(dataFeeder,batch_size=1)
c2 = 0
c1 = 0
c0 = 0
for i,data in enumerate(trainData):
    for tar in data[1]:
        tar=tar.numpy()
        if (tar == 2).any() :
            c2 +=1
        if (tar == 1).any():
            c1 += 1
        if (tar == 0).any():
            c0 += 1


        print("{}th processed".format(i))

print("class2 number: {}".format(c2))
print("class1 number: {}".format(c1))
print("class0 number: {}".format(c0))