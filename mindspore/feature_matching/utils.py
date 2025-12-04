# import torch
import mindspore
def get_client_centroids_info(model, dataloaders, model_name, dataset_name, party_list_this_round, num_anchor=0,net_data_length=[]):
    # return the centroids and num_per_class for each client
    model.set_train(False)
    
    local_centroids = []
    local_distributions = []

    if model_name.startswith('resnet50'):
        feature_d = 2048
    elif model_name.startswith('resnet18'):
        feature_d = 512

    if dataset_name in ['cifar10', 'cinic10']:
        num_classes=10


    for net_id in party_list_this_round:
        dataloader = dataloaders[net_id]
        print(net_data_length)
        client_rep = mindspore.ops.zeros((net_data_length[net_id],feature_d)) #dataloader.dataset change to dataloader.source
        client_label = mindspore.ops.zeros(net_data_length[net_id]) # the same
        bs = dataloader.batch_size

        # with torch.no_grad():
        for batch_id, (images, labels) in enumerate(dataloader):
            # images, labels = images.cuda(), labels.cuda()
            # _, representation = model(images, last2_layer=True).detach()
            representation, _, _ = model(images)
            # if ((batch_id+1)*bs)<net_data_length[net_id]: # the same
            #     client_rep[batch_id*bs:batch_id*bs+bs]= representation
            #     client_label[batch_id*bs:batch_id*bs+bs] = labels
            # else:
            #     client_rep[batch_id*bs:] = representation
            #     client_label[batch_id*bs:] = labels
            real_bs = images.shape[0]  # 用 label 的真实大小，不要相信 MindSpore 的 batch_size

            start = batch_id * bs
            end = start + real_bs
            client_rep[start:end] = representation[:real_bs]
            client_label[start:end] = labels

        client_centroids, client_distribution = cal_center(rep=client_rep, label=client_label, num_classes=num_classes)
        ## for test
        ## 
        local_centroids.append(client_centroids)
        local_distributions.append(client_distribution)



    return local_centroids, local_distributions

def cal_center(rep, label, num_classes):
    # calculation for the 'get_client_centroids_info' function
    center_of_class = mindspore.ops.zeros((num_classes, rep.shape[1]))
    distribution_of_class = mindspore.ops.zeros(num_classes)
    for class_id in range(num_classes):
        if (label==class_id).sum()!=0:
            center_of_class[class_id,:] = rep[label==class_id].mean(axis=0)
            distribution_of_class[class_id] = mindspore.ops.sum(label==class_id)
    return center_of_class, distribution_of_class

def get_global_centroids(local_centroids, local_distributions, pre_global_centroids, momentum=0.0, equally_average=0):
    # calculate global centroids using local_centroids based on local_distributions
    zeroslike = mindspore.ops.ZerosLike()
    global_centroids = zeroslike(local_centroids[0])
    
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

