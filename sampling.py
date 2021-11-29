import numpy as np
from collections import Counter

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
        dict_users[i] = list(dict_users[i])
    return dict_users


def sample_noniid(dataset, client_data_ratio = None, is_overlap = False): #client_data_size
    """Sample non-iid data from dataset for each user.

    Args:
        dataset ([torch.datasets]): dataset to sample from
        client_data_ratio ([type]): distribution of each local datasets. A matrix where
            each row represents the class distribution within one client's dataset.
        is_overlap (bool): whether the datapoint is being sampled multiple times. 
    Returns:
        dict_users: dictionary of data index for each user 
            {user_id: list(data_index)}
    """
    # TODO implement sample non-iid 
    CLIENT_DATA_RATIO = np.array([[0.3, 0, 0, 0.1, 0., 0, 0.3, 0.3, 0, 0],
                                 [0., 0.3, 0, 0.3, 0., 0, 0.1, 0, 0.25, 0.05],
                                 [0., 0, 0.2, 0., 0., 0.3, 0, 0., 0.1, 0.4],
                                 [0.3, 0, 0, 0., 0.3, 0, 0, 0., 0.4, 0],
                                 [0., 0, 0.3, 0., 0.3, 0., 0, 0.4, 0., 0.]]) 
    
    if client_data_ratio == None:
        client_data_ratio = CLIENT_DATA_RATIO
    
    ### distribute to clients
    dict_users = {} 

    label_arr = get_labels(dataset)
    class_indices = get_class_indices(dataset, label_arr)

#     num_users = client_data_ratio.shape[0]
#     num_classes = client_data_ratio.shape[1]
    num_users = 5
    num_classes = 10

    num_items = int(len(dataset)/num_users)
    all_idxs = class_indices.copy()

    for client_id in range(num_users):
        client_indices = []
        
        ### for each class: 
        for class_label in range(num_classes): 
            num_of_class_data = int(client_data_ratio[client_id][class_label] * num_items) # get the number of data points from each distribution

            chosen_set = np.random.choice(all_idxs[class_label], num_of_class_data, replace = is_overlap)

            client_indices.extend(list(chosen_set))

        dict_users[client_id] = client_indices 
        
    return dict_users, label_arr, class_indices

def get_labels(dataset):
    label_arr = np.zeros(len(dataset))
    for i, (inputs, targets) in enumerate(dataset):
        label_arr[i] = targets
    return label_arr

def get_class_indices(dataset, label_arr):

    class_dict = Counter(label_arr)
    class_indices = {}
    class_indices[0] = list(range(0, class_dict[0]))
    end_num = class_dict[0]                      
    for i in range(1, 10):                   
        class_indices[i] = list(range(end_num, end_num + class_dict[i]))
        end_num += class_dict[i]
        
#     class_data = {}
#     for i in range(10):
#         class_data[i] = torch.utils.data.Subset(dataset, class_indices[i])
        
    return class_indices