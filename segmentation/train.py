import torch
from torch import nn
import tiramisu
import utils
import argparse
import dataloader
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='this is training!')
    parser.add_argument("--batchsize","-b",type=int,default=8,help="setting batchsize.")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-epoch","-e",type=int, default=100, help="training epoch" )
    args = parser.parse_args()

    #build networks
    fcd = tiramisu.FCDenseNet67(3)
    #fcd.apply(utils.weights_init)
    fcd.load_state_dict(torch.load("~/Downloads/pycharm/weights/weights-99-0.104-0.047.pth"))
    fcd = nn.DataParallel(fcd).to(device)
    #load traindata

    dataFeeder = dataloader.dataTrainloader()
    trainData = torch.utils.data.DataLoader(dataFeeder,batch_size=args.batchsize, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = torch.optim.Adam(fcd.parameters(), lr=args.lr, betas=(0.9,0.99))
    weight = torch.FloatTensor([1,0.9560,0.5202])
    criterion = nn.NLLLoss(weight=weight).to(device)

    for epoch in range(args.epoch):
        since = time.time()
        trn_loss = 0
        trn_error = 0
        for i, trainList in enumerate(trainData):

            train_img = torch.autograd.Variable(trainList[0]).to(device)
            train_label = torch.autograd.Variable(trainList[1]).to(device)
            #print(train_label.size())

            optimizer.zero_grad()
            output = fcd(train_img)


            #output=output.view(args.batchsize,3, -1)
            #print(output.size())
            #train_label=train_label.view(args.batchsize, 3, -1)

            loss = criterion(output,train_label)



            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

            pred = utils.get_predictions(output)
            #print(pred.size())
            error= utils.error(pred, train_label.data.cpu())
            trn_error += error
            if (i+1) % 10 == 0:
                print("Iteration {} / {}: Loss: {:.4f}, Acc: {:.4f}".format(i+1, len(trainData),loss.item(), 1 - error))
        trn_loss /= len(trainData)
        trn_error /= len(trainData)

        print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, trn_loss, 1 - trn_error))
        time_elapsed = time.time() - since
        print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        ### Checkpoint ###
        utils.save_weights(fcd, epoch, trn_loss, trn_error)

        ### Adjust Lr ###
        utils.adjust_learning_rate(args.lr, 0.995, optimizer,epoch, 2)




if __name__ == '__main__':
    main()