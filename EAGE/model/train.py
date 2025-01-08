import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from model import device
from model.AT_AE import  AT_AE
import random

def drop_long_traces(df,max_trace_len=64,event_id='event_id'):
    df=df.drop(np.unique(df[df[event_id]>max_trace_len].index))
    return df

# Cell
def RandomTraceSplitter(split_pct=0.2, seed=None):
    "Create function that splits `items` between train/val with `valid_pct` randomly."
    def _inner(trace_ids):
        o=np.unique(trace_ids)
        np.random.seed(seed)
        rand_idx = np.random.permutation(o)
        cut = int(split_pct * len(o))
        #return L(rand_idx[cut:].tolist()),L(rand_idx[:cut].tolist())
        #return L(rand_idx[cut:].tolist()), L(rand_idx[:cut].tolist())
        return list(rand_idx[cut:]), list(rand_idx[:cut])
    return _inner

# Cell
def split_traces(df,test_seed=42,validation_seed=None):

    #df=drop_long_traces(df)
    ts=RandomTraceSplitter(seed=test_seed)
    train,test=ts(df.index)
    #ts=RandomTraceSplitter(seed=validation_seed,split_pct=0.1)
    # train,valid=ts(train)
    # return train,valid,test

    return train, test

def train(dataset,n_epochs,batch_size,lr ,b1 ,b2 ,seed,hidden_dim , GAT_heads , decoder_num_layers ,threshold,TF_styles):
    if type(seed) is int:
        torch.manual_seed(seed)

    at_ae = AT_AE(dataset.attribute_dims,dataset.attribute_embedding_dims, dataset.max_len, hidden_dim, GAT_heads, decoder_num_layers, TF_styles)
    loss_func = nn.CrossEntropyLoss()

    at_ae.to(device)

    optimizer = torch.optim.Adam(at_ae.parameters(),lr=lr, betas=(b1, b2))

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=0.85
    )

    Xs = []
    for i, dim in enumerate(dataset.attribute_dims):
        Xs.append(torch.LongTensor(dataset.features[i]))

    indexes = [i for i in range(len(dataset))]


    th_df = pd.DataFrame(index=indexes)
    aaa,bbb=split_traces(th_df)

    ind_train = []
    ac=0
    for i in aaa:
        if dataset.anomaly_labels[i] == 0:
            ind_train.append(i)
        else:
            ac+=1

    ind_test=bbb

    at_ae.train()

    print("*"*10+"training"+"*"*10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        #自定义的dataloader
        random.shuffle(ind_train)
        for bathc_i in tqdm(range(batch_size, len(ind_train)+1,batch_size)):
            this_batch_indexes=ind_train[bathc_i-batch_size:bathc_i]
            nodes_list = [dataset.node_xs[i] for i in this_batch_indexes]
            Xs_list=[]
            graph_batch_list = []
            for i in range(len(dataset.attribute_dims)):
                Xs_list.append(torch.tensor(Xs[i][this_batch_indexes]))
                aaa = []
                for b in range(len(nodes_list)):
                    for l in range(len(nodes_list[b][i])):
                        aaa.append(torch.tensor(nodes_list[b][i][l]))
                graph_batch_list.append(torch.stack(aaa, dim=0))
            # 将列表中的每个张量移至GPU
            Xs_list = torch.stack(Xs_list, dim=0).cuda()
            graph_batch_list = [torch.tensor(x).cuda() for x in graph_batch_list]

            mask = torch.tensor(dataset.mask[this_batch_indexes]).cuda()

            attr_reconstruction_outputs = at_ae(graph_batch_list,Xs_list,mask,len(this_batch_indexes))

            optimizer.zero_grad()

            loss=0.0
            mask[:, 0] = False # 除了每一个属性的起始字符之外,其他重建误差
            for i in range(len(dataset.attribute_dims)):
                #--------------
                # 除了每一个属性的起始字符之外,其他重建误差
                #---------------
                pred=attr_reconstruction_outputs[i][mask]
                true=Xs_list[i][mask]
                loss+=loss_func(pred,true)

            train_loss += loss.item()
            train_num += 1
            #optimizer.zero_grad()
            loss.backward()
            #optimizer.step()

            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch=train_loss / train_num
        print(f"[Epoch {epoch+1:{len(str(n_epochs))}}/{n_epochs}] "
              f"[loss: {train_loss_epoch:3f}]")
        at_ae.train()
        torch.save(at_ae.state_dict(), f"large_3_4-{epoch + 1}-model.pth")
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()

    #return gat_ae,ind_test

