import os
import copy
import pickle
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split, DataLoader
from sampling import sample_iid
from utils import get_dataset, exp_details, average_weights
from options import args_parser
from update import test_inference, LocalUpdate, DatasetSplit
# from models import MLP, CNN
from models.imagenet import resnext50


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    path_project = os.path.abspath('..')
#     # container path
#     path_project = '/home/jovyan/work'

    # Get whole dataset
    train_dataset, test_dataset, _ = get_dataset(
        # data_dir=f'{path_project}/data/imagenette2/',
        data_dir=args.data_dir, dataset=args.dataset,
        num_users=1, iid=1
    )
    # # Train on 1/5 dataset
    # train_dict = sample_iid(train_dataset, 5)  # user_groups - dict
    # train_inds = train_dict[1]  # train subset inds - set
    # test_dict = sample_iid(test_dataset, 5)
    # test_inds = test_dict[1]  # test subset inds - set
    # train_subset = DatasetSplit(train_dataset, train_inds)
    # test_subset = DatasetSplit(test_dataset, test_inds)
    # print(f'1/5 Training set size: {len(train_subset)}')
    # print(f'1/5 Testing set size: {len(test_subset)}')
    # Train on full dataset
    train_dict = sample_iid(train_dataset, 1)  # user_groups - dict
    train_inds = train_dict[0]  # train subset inds - set
    test_dict = sample_iid(test_dataset, 1)
    test_inds = test_dict[0]  # test subset inds - set
    train_subset = DatasetSplit(train_dataset, train_inds)
    test_subset = DatasetSplit(test_dataset, test_inds)
    print(f'Full training set size: {len(train_subset)}')
    print(f'Full testing set size: {len(test_subset)}')

    # Build model
    if args.model == 'resnext':
        base_model = resnext50(
            baseWidth=args.basewidth,
            cardinality=args.cardinality)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if torch.cuda.is_available():
        base_model = torch.nn.DataParallel(base_model).cuda()
    else:
        base_model.to(device)
    base_model.train()
    print(base_model)

    # Copy weights
    base_weights = base_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1  # print after each epoch

    num_epochs = args.epochs * args.local_ep
    for epoch in tqdm(range(num_epochs)):
        print(f'\n | Base Training Round : {epoch + 1} |\n')

        # base_model.train()
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                 idxs=train_inds)
        w, loss = local_model.baseline_update_weights(
            model=base_model, baseline_round=epoch)

        # Update weights
        base_model.load_state_dict(w)

        # Calculate training loss at every epoch
        train_loss.append(loss)

        # Calculate training accuracy at every epoch
        base_model.eval()
        acc, loss = local_model.inference(model=base_model)
        train_accuracy.append(acc)

        # print training loss and acc after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nTraining Stats after {epoch + 1} baseline rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    file_name = args.results_dir + f'/{args.dataset}_{args.model}_{num_epochs}_S[1]_iid[{args.iid}]_E[' \
                                   f'{args.local_ep}]_B[{args.local_bs}].pkl'
    with open(file_name, 'wb') as f:
        train_log = {'loss': train_loss,
                     'acc': train_accuracy,
                     'weights': base_model.state_dict()}
        pickle.dump(train_log, f)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, base_model, test_dataset)

    print(f' \n Results after {num_epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


    # train_len = int(len(train_dataset)*0.8)
    # val_len = len(train_dataset) - train_len
    # train_set, val_set = random_split(train_dataset, [train_len, val_len])
    # print(len(train_set))
    # print(len(val_set))
    # print(len(test_dataset))

#     # BUILD MODEL
#     if args.model == 'resnext':
#         global_model = resnext50(
#             baseWidth=args.basewidth,
#             cardinality=args.cardinality)
#     else:
#         exit('Error: unrecognized model')
#
#     # Set the model to train and send it to device.
#     if torch.cuda.is_available():
#         global_model = torch.nn.DataParallel(global_model).cuda()
#     else:
#         global_model.to(device)
#     global_model.train()
#     print(global_model)
#
#     global_weights = global_model.state_dict()
#
#     # Training
#     # Set optimizer and criterion
#     if args.optimizer == 'sgd':
#         optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
#                                     momentum=0.9)
#     elif args.optimizer == 'adam':
#         optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
#                                      weight_decay=1e-4)
#
#     train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
#     val_loader = DataLoader(train_set, batch_size=64, shuffle=False)
#
#     if torch.cuda.is_available():
#         criterion = torch.nn.CrossEntropyLoss().cuda()
#     else:
#         criterion = torch.nn.CrossEntropyLoss().to(device)
#
# # Baseline Code
#     epoch_loss = []
#     num_epochs = args.epochs * args.local_ep
#     for epoch in tqdm(range(num_epochs)):
#         batch_loss = []
#
#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = global_model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             if batch_idx % 50 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch+1, batch_idx * len(images), len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader), loss.item()))
#             batch_loss.append(loss.item())
#
#         loss_avg = sum(batch_loss)/len(batch_loss)
#         print('\nTrain loss:', loss_avg)
#         epoch_loss.append(loss_avg)
#
#     # Plot loss
#     plt.figure()
#     plt.plot(range(len(epoch_loss)), epoch_loss)
#     plt.xlabel('epochs')
#     plt.ylabel('Train loss')
#     plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
#                                                  num_epochs))
#
#     # testing
#     test_acc, test_loss = test_inference(args, global_model, test_dataset)
#     print('Test on', len(test_dataset), 'samples')
#     print("Test Accuracy: {:.2f}%".format(100*test_acc))
