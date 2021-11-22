import os
import copy
import time
import pickle
import numpy as np
from torch.utils import data
from tqdm import tqdm
import torch
from options import args_parser
from update import LocalUpdate, test_inference
from models.imagenet import resnext50
from utils import get_dataset, average_weights, exp_details

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(
        data_dir=args.data_dir, dataset=args.dataset, 
        num_users=args.num_users, iid=args.iid
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
    
    file_name = args.results_dir+f'/{args.dataset}_{args.model}_\
                    {args.epochs}_C[{args.frac}]_iid[{args.iid}]_\
                    E[{args.local_ep}]_B[{args.local_bs}].pkl'
    
    with open(file_name, 'rb') as f:
        logger = pickle.load(f)
    
    