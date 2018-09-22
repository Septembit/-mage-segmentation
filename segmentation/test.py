import torch
from torch import nn
import tiramisu
import utils
import argparse
import dataloader
import time
import numpy as np
from PIL import Image
torch.cuda.set_device(0)



def main():
    parser = argparse.ArgumentParser(description='this is training!')
    parser.add_argument("--batchsize","-b",type=int,default=1,help="setting batchsize.")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-epoch","-e",type=int, default=100, help="training epoch" )
    args = parser.parse_args()

    #build networks
    fcd = tiramisu.FCDenseNet67(3)
    #fcd = nn.DataParallel(fcd).to(device)
    #fcd.apply(utils.weights_init)
    #fcd.load_state_dict(torch.load("/home/yachao-li/Downloads/pycharm/weights/latest.th")["c"])
    fpath = "/home/yachao-li/Downloads/pycharm/weights/latest.th"
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in weights['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    fcd.load_state_dict(new_state_dict)
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch , weights['loss'], weights['error']))

    fcd = fcd.cuda()
    #load traindata

    dataFeeder = dataloader.testLoader()
    testData = torch.utils.data.DataLoader(dataFeeder,batch_size=args.batchsize, shuffle=False, num_workers=1, pin_memory=True)
    dataFile = "/home/yachao-li/Downloads/test.txt"
    f = open(dataFile, "r")
    data1 = f.readlines()
    print(data1[0].split("\n")[0])

    for i, data in enumerate(testData):
        data = torch.autograd.Variable(data, volatile=False).cuda()

        output = fcd(data)
        pred = utils.get_predictions(output)
        pred = pred.numpy().astype(np.uint8).reshape(180,320)
        pred = Image.fromarray(pred)
        pred = pred.resize((1280, 720), Image.ANTIALIAS)
        pred.save("/home/yachao-li/Downloads/seg/" + data1[i].split("\n")[0]+ "_drivable_id.png")


        print("{}th processed.".format(i))


    f.close()

if __name__ == '__main__':
    main()