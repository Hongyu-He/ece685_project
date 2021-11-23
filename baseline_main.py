import os
import copy
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split, DataLoader
from utils import get_dataset, exp_details, average_weights
from options import args_parser
from update import test_inference, LocalUpdate
# from models import MLP, CNN
from models.imagenet import resnext50


if __name__ == '__main__':
    start_time = time.time()
    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    exp_details(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(
        data_dir=args.data_dir, dataset=args.dataset,
        num_users=1, iid=args.iid
    )
    train_set, val_set = random_split(
        train_dataset,
        [0.8*len(train_dataset), 0.2*len(train_dataset)]
    )

    # BUILD MODEL
    if args.model == 'resnext':
        global_model = resnext50(
            baseWidth=args.basewidth,
            cardinality=args.cardinality)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if torch.cuda.is_available():
        global_model = torch.nn.DataParallel(global_model).cuda()
    else:
        global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # cuda
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
