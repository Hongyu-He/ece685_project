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

    if torch.cuda.is_available():
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    best_acc = 0  # best validation accuracy
    start_epoch = 0
    num_epoch = args.epochs * args.local_ep  # global epoch x local epoch

    for epoch in range(start_epoch, num_epoch):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, num_epoch, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [0.6*num_epoch, 0.9*num_epoch]:
        state['lr'] *= args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']



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
