import os
import copy
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sampling import sample_iid, sample_noniid


def get_dataset(data_dir, dataset='imagenet', num_users=10, iid=1):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index sand the values are the corresponding data for
    each of those users.

    Args:
        dataset (str): Name of the dataset, default to 'imagenet'
        num_users (int): Number of local clients, default to 10
        iid (int, optional): whether to sample iid data. Defaults to 1.
            set to 0 if use non-iid data.

    Returns:
        trainset: training set
        testset: test set
        user_groups: a user group which is a dict where
            the keys are the user index sand the values are the corresponding data for
            each of those users.
    """
    if dataset == 'imagenet':
        traindir = os.path.join(data_dir, 'train')
        testdir = os.path.join(data_dir, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])

        trainset = ImageFolder(root=traindir, transform=train_transform)
        testset = ImageFolder(root=testdir, transform=test_transform)

        # sample training data amongst users
        if iid == 1:
            # Sample IID user data
            user_groups = sample_iid(trainset, num_users)
        else:
            # TODO Sample Non-IID user data
            # user_groups = sample_noniid(trainset, num_users)
            user_groups = sample_noniid(trainset, client_data_ratio = None, is_overlap = False)
            raise NotImplementedError("Can't sample Non-IID user data")

    return trainset, testset, user_groups


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sample_imgs(dataset, num, plot=True):
    """Sample and plot images from dataset.

    Args:
        dataset ([type]): [description]
        num ([type]): [description]
        plot (bool, optional): [description]. Defaults to True.
    """
    # TODO implement me


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Fed Algo  : {args.fed}')
    if args.fed == 'fedprox':
        print(f'    Mu        : {args.mu}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
