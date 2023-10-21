import torch

def get_client_centroids_info(model, dataloaders, model_name, dataset_name, party_list_this_round, num_anchor=0):
    # return the centroids and num_per_class for each client
    model.eval()
    
    local_centroids = []
    local_distributions = []

    if model_name.startswith('resnet50'):
        feature_d = 2048
    elif model_name.startswith('resnet18'):
        feature_d = 512

    if dataset_name in ['cifar10', 'cinic10']:
        num_classes=10
    elif dataset_name=='cifar100':
        num_classes=100
    elif dataset_name=='tinyimagenet':
        num_classes=200
    elif dataset_name=='ham10000':
        num_classes=7
    elif dataset_name=='wiki':
        num_classes=num_anchor

    for net_id in party_list_this_round:
        dataloader = dataloaders[net_id]
        client_rep = torch.zeros((len(dataloader.dataset),feature_d))
        client_label = torch.zeros(len(dataloader.dataset))
        bs = dataloader.batch_size

        with torch.no_grad():
            for batch_id, (images, labels) in enumerate(dataloader):
                images, labels = images.cuda(), labels.cuda()
                # _, representation = model(images, last2_layer=True).detach()
                representation, _, _ = model(images)
                if ((batch_id+1)*bs)<len(dataloader.dataset):
                    client_rep[batch_id*bs:batch_id*bs+bs]= representation
                    client_label[batch_id*bs:batch_id*bs+bs] = labels
                else:
                    client_rep[batch_id*bs:] = representation
                    client_label[batch_id*bs:] = labels

        client_centroids, client_distribution = cal_center(rep=client_rep, label=client_label, num_classes=num_classes)

        local_centroids.append(client_centroids)
        local_distributions.append(client_distribution)

    model.train()

    return local_centroids, local_distributions

def cal_center(rep, label, num_classes):
    # calculation for the 'get_client_centroids_info' function
    center_of_class = torch.zeros((num_classes, rep.shape[1]))
    distribution_of_class = torch.zeros(num_classes)
    for class_id in range(num_classes):
        if (label==class_id).sum()!=0:
            center_of_class[class_id,:] = rep[label==class_id].mean(dim=0)
            distribution_of_class[class_id] = torch.sum(label==class_id)

    return center_of_class, distribution_of_class

def get_global_centroids(local_centroids, local_distributions, pre_global_centroids, momentum=0.0, equally_average=0):
    # calculate global centroids using local_centroids based on local_distributions
    global_centroids = torch.zeros_like(local_centroids[0])

    if equally_average:
        # if a client has no data sample of category x, then, assign the corresponding local anchor with last round's global anchor
        for client_id in range(len(local_centroids)):
            for class_id in range(global_centroids.shape[0]):
                if local_distributions[client_id][class_id]<1:
                    local_centroids[client_id][class_id,:] = pre_global_centroids[class_id,:]
                    # print(f'client : {client_id} | category : {class_id}')
        for client_id in range(len(local_centroids)):
            global_centroids +=  local_centroids[client_id] / len(local_centroids)
    else:
        for class_id in range(global_centroids.shape[0]):               # for each class
            total_num = 0
            for client_id in range(len(local_centroids)):
                total_num += local_distributions[client_id][class_id]
            for client_id in range(len(local_centroids)):
                weight = (local_distributions[client_id][class_id]/total_num)
                global_centroids[class_id,:] += local_centroids[client_id][class_id,:] * weight
            #     print(weight, local_centroids[client_id][class_id,:5])
            # print(global_centroids[class_id,:5])
    if global_centroids.sum()==0:
        print('First round requires no pre_global_centroids.')
    else:
        # print(pre_global_centroids.shape)
        # print(global_centroids.shape)
        global_centroids = momentum * pre_global_centroids + (1-momentum) * global_centroids

    return global_centroids

def personalized_get_client_centroids_info(nets_this_round, dataloaders, model_name, dataset_name, party_list_this_round, num_anchor=0):
    # return the centroids and num_per_class for each client
    local_centroids = []
    local_distributions = []

    if model_name=='resnet50' or model_name=='resnet50_7':
        feature_d = 2048
    elif model_name=='resnet18' or model_name=='resnet18_7':
        feature_d = 512

    if dataset_name=='cifar10':
        num_classes=10
    elif dataset_name=='cifar100':
        num_classes=100
    elif dataset_name=='tinyimagenet':
        num_classes=200
    elif dataset_name=='ham10000':
        num_classes=7
    elif dataset_name=='wiki':
        num_classes=num_anchor

    for net_id in party_list_this_round:
        model = nets_this_round[net_id]
        model.cuda()
        model.eval()

        dataloader = dataloaders[net_id]
        client_rep = torch.zeros((len(dataloader.dataset),feature_d))
        client_label = torch.zeros(len(dataloader.dataset))
        bs = dataloader.batch_size

        with torch.no_grad():
            for batch_id, (images, labels) in enumerate(dataloader):
                images, labels = images.cuda(), labels.cuda()
                # _, representation = model(images, last2_layer=True).detach()
                representation, _, _ = model(images)
                if ((batch_id+1)*bs)<len(dataloader.dataset):
                    client_rep[batch_id*bs:batch_id*bs+bs]= representation
                    client_label[batch_id*bs:batch_id*bs+bs] = labels
                else:
                    client_rep[batch_id*bs:] = representation
                    client_label[batch_id*bs:] = labels

        client_centroids, client_distribution = cal_center(rep=client_rep, label=client_label, num_classes=num_classes)

        local_centroids.append(client_centroids)
        local_distributions.append(client_distribution)
        
        model.train()
        model.to('cpu')

    return local_centroids, local_distributions