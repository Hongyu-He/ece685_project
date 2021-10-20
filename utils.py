import copy
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sampling import sample_iid, sample_noniid


def get_dataset(dataset, num_users, iid=True):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.

    Args:
        dataset ([type]): [description]
        num_users ([type]): [description]
        iid (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if dataset == 'chest_xray':
        data_dir = '../data/chest_xray/'
        # TODO try other transformers, use different transformer for train/test/val
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = ImageFolder(root=data_dir+'train', transform=apply_transform)
        testset = ImageFolder(root=data_dir+'test', transform=apply_transform)
        valset = ImageFolder(root=data_dir+'val', transform=apply_transform)

        # sample training data amongst users
        if iid:
            # Sample IID user data
            user_groups = sample_iid(trainset, num_users)
        else:
            # TODO Sample Non-IID user data
            # user_groups = sample_noniid(trainset, num_users)
            user_groups = {}
            raise NotImplementedError("Can't sample Non-IID user data")

    return trainset, testset, valset, user_groups


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
