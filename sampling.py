import numpy as np


def sample_iid(dataset, num_users):
    """Sample iid data from dataset for each user.

    Args:
        dataset (torch.datasets): train set to sample from
        num_users (int): number of users

    Returns:
        dict_users: dictionary of data index for each user 
            {user_id: set(data_index)}
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def sample_noniid(dataset, client_data_ratio, client_data_size):
    """Sample non-iid data from dataset for each user.

    Args:
        dataset ([torch.datasets]): dataset to sample from
        client_data_ratio ([type]): distribution of each local datasets. A matrix where
            each row represents the class distribution within one client's dataset.
        client_data_size: ratio of the dataset within each client. 
    Returns:
        dict_users: dict_users: dictionary of data index for each user 
            {user_id: set(data_index)}
    """
    # TODO implement sample non-iid

    ### return 

    pass