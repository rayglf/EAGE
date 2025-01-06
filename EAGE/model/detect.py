import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

from model import device

# Cell
def calculate_precision_recall(predictions, labels):
    true_positives = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
    false_positives = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
    false_negatives = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def get_th_df(ths,abnormal,y_true):
    res = []
    for t in ths:
        y_pred = ((abnormal > t).sum(axis=(1, 2)) >= 1).astype('int64')
        anomaly_ratio = sum(x == 1 for x in y_pred) / len(y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        res.append([f1, precision, recall, anomaly_ratio])

    columns='F1 Score','Precision','Recall', 'Anomaly Ratio'
    th_df = pd.DataFrame(res,columns=columns,index=ths)
    return th_df

# Cell
def get_best_threshhold(th_df): return th_df.iloc[th_df['F1 Score'].argmax()]

# Cell
def get_ratio_th(ratio,th_df):return th_df.iloc[th_df['Anomaly Ratio'].sub(ratio).abs().argmin()]

# Cell
def elbow_heuristic(th_df):
    taus = th_df.index.to_numpy()
    r = th_df['Anomaly Ratio'].to_numpy()
    step = taus[1:] - taus[:-1]
    r_prime_prime = (r[2:] - 2 * r[1:-1] + r[:-2]) / (step[1:] * step[:-1])
    ellbow_down = th_df.iloc[np.argmax(r_prime_prime) + 1]
    ellbow_up = th_df.iloc[np.argmin(r_prime_prime) + 1]
    return ellbow_down,ellbow_up

# Cell
def get_fixed_heuristic(fixed,th_df): return th_df.iloc[np.absolute(th_df.index.to_numpy()-fixed).argmin()]
# Cell
def get_lowest_plateau_heuristic(th_df):
    taus = th_df.index.to_numpy()
    r = th_df['Anomaly Ratio'].to_numpy()
    r_prime = (r[1:] - r[:-1]) / (taus[1:] - taus[:-1])
    #r_prime = np.pad(r_prime, (0, 1), mode='constant')
    stable_region = r_prime > np.mean(r_prime) / 2
    regions = np.split(np.arange(len(stable_region)), np.where(~stable_region)[0])
    regions = [taus[idx[1:]] for idx in regions if len(idx) > 1]
    if len(regions) == 0:
        regions = [taus[-2:]]
    #print("regions[-1]:",regions[-1])
    lp_min = get_fixed_heuristic(regions[-1].min(),th_df)
    lp_mean = get_fixed_heuristic(regions[-1].mean(),th_df)
    lp_max = get_fixed_heuristic(regions[-1].max(),th_df)
    return lp_min,lp_mean,lp_max


def detect(at_ae, dataset, batch_size,ind_test,threshold):
    at_ae.eval()
    at_ae.cuda()
    with torch.no_grad():
        final_res = []
        attribute_dims=dataset.attribute_dims

        Xs = []
        for i, dim in enumerate(dataset.attribute_dims):
            Xs.append(torch.LongTensor(dataset.features[i]))

        print("*" * 10 + "detecting" + "*" * 10)

        pre = 0

        for bathc_i in tqdm(range(batch_size, len(ind_test)+batch_size, batch_size)):
            if bathc_i <= len(ind_test):
                #this_batch_indexes = list(range(pre, bathc_i))
                this_batch_indexes = ind_test[pre:bathc_i]
            else:
                #this_batch_indexes = list(range(pre,  len(ind_test)))
                this_batch_indexes = ind_test[pre: len(ind_test)]

            nodes_list = [dataset.node_xs[i] for i in this_batch_indexes]
            Xs_list = []
            # graph_batch_list = []
            # for i in range(len(dataset.attribute_dims)):
            #     Xs_list.append(Xs[i][this_batch_indexes].to(device))
            #     graph_batch = Batch.from_data_list([Data(x=nodes_list[b][i], edge_index=edge_indexs_list[b])
            #                                         for b in range(len(nodes_list))])
            #     graph_batch_list.append(graph_batch.to(device))

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
            # 将列表中的每个张量移至GPU
            #graph_batch_list = [x.cuda() for x in graph_batch_list]

            # 将列表中的每个对象转换为张量并移至GPU
            #graph_batch_list = torch.stack(graph_batch_list, dim=0)
            graph_batch_list = [torch.tensor(x).cuda() for x in graph_batch_list]


            mask = torch.tensor(dataset.mask[this_batch_indexes]).cuda()

            attr_reconstruction_outputs = at_ae(graph_batch_list, Xs_list, mask, len(this_batch_indexes))

            for attr_index in range(len(attribute_dims)):
                attr_reconstruction_outputs[attr_index] = torch.softmax(attr_reconstruction_outputs[attr_index], dim=2)

            this_res = []
            for attr_index in range(len(attribute_dims)):
                # 取比实际出现的属性值大的其他属性值的概率之和
                # temp = attr_reconstruction_outputs[attr_index]
                # index = Xs_list[attr_index].unsqueeze(2)
                # probs = temp.gather(2, index)
                # temp[(temp <= probs)] = 0
                # res = temp.sum(2)
                # res = res * (mask)
                # this_res.append(res)

#################异常分数=(最大值-当前值)/最大值########################################################
                # if attr_index==3 or attr_index==4 or attr_index==5:
                #     continue
#################
                temp = attr_reconstruction_outputs[attr_index]
                index = Xs_list[attr_index].unsqueeze(2)
                probs = temp.gather(2, index)
                #max_values = np.amax(np.array(temp), axis=2)

                # 将CUDA设备上的张量复制到主机内存中，然后转换为NumPy数组
                max_values = np.amax(np.array(temp.cpu()), axis=2)

                res=[]
                for i in range(len(temp)):
                    aa=[]
                    for j in range(len(temp[0])):
                        bb=(max_values[i][j]-probs[i][j][0])/max_values[i][j]
                        aa.append(bb.item())
                    res.append(aa)
                mask = mask.to(device)
                # 将列表中的每个对象转换为张量并移至指定设备
                res = [torch.tensor(x).to(device) for x in res]

                res=torch.stack(res,dim=0)
                res = res * (mask)
                this_res.append(res)
                ###############################################################################################

            final_res.append(torch.stack(this_res, 2))
            pre = bathc_i

        abnormal = np.array(torch.cat(final_res, 0).detach().cpu())
        ##########################################################################
        # anomaly_labels_list = [dataset.anomaly_labels[i] for i in ind_test]
        # ths = np.array((range(10000))) * 0.0001 + 0.0
        # ths=ths[-1000:]
        # tot = []
        # for i in range(len(dataset.attr_keys)):
        #     a = i + 1
        #     arr = abnormal[:, :, i:a]
        #     th_df = get_th_df(ths, arr, anomaly_labels_list)
        #     a, b, c = get_lowest_plateau_heuristic(th_df)
        #
        #     trace_level_detection2 = ((arr > b.name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        #     tot.append(trace_level_detection2)
        # tot = np.array(tot).transpose()
        # f1 = f1_score(anomaly_labels_list, np.max(tot, axis=1))
        # print(f"LP-Mean-F1分数:", f1)



        ###########################################################################


        anomaly_labels_list = [dataset.anomaly_labels[i] for i in ind_test]
        ths = np.array((range(1000))) * 0.001 + 0.0
        # ths = np.array((range(500))) * 0.002 + 0.0

        # ths = np.array((range(10000))) * 0.0001 + 0.0
        # ths = ths[-1000:]

        # ths = np.array((range(5000))) * 0.0002 + 0.0
        # ths = ths[-1000:]


        th_df = get_th_df(ths,abnormal,anomaly_labels_list)
        print("*"*30)
        # ratio_heuristic = get_ratio_th(0.5,th_df)
        # ratio_heuristic_trace_level_detection = ((abnormal > ratio_heuristic.name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        # f1 = f1_score(anomaly_labels_list, ratio_heuristic_trace_level_detection)
        # print(f"AR-0.5-{ratio_heuristic.name}-F1分数:", f1)
        #
        # print("--" * 15)
        # ellbow_down,ellbow_up = elbow_heuristic(th_df)
        # ellbow_down_trace_level_detection1= ((abnormal > ellbow_down.name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        # f1 = f1_score(anomaly_labels_list, ellbow_down_trace_level_detection1)
        # print(f"ellbow_down-{ellbow_down.name}-F1分数:", f1)
        # ellbow_up_trace_level_detection = ((abnormal > ellbow_up.name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        # f1 = f1_score(anomaly_labels_list, ellbow_up_trace_level_detection)
        # print(f"ellbow_up-{ellbow_up.name}-F1分数:", f1)


        #print("--" * 15)
        a,b,c=get_lowest_plateau_heuristic(th_df)
        trace_level_detection1 = ((abnormal > a.name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        f1 = f1_score(anomaly_labels_list, trace_level_detection1)
        print(f"LP-Min-{a.name}-F1分数:", f1)
        trace_level_detection2 = ((abnormal > b.name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        f1 = f1_score(anomaly_labels_list, trace_level_detection2)
        print(f"LP-Mean-{b.name}-F1分数:", f1)
        trace_level_detection3 = ((abnormal > c.name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        f1 = f1_score(anomaly_labels_list, trace_level_detection3)
        print(f"LP-Max-{c.name}-F1分数:", f1)
        print("--"*15)
        best_trace_level_detection = ((abnormal > get_best_threshhold(th_df).name.item()).sum(axis=(1, 2)) >= 1).astype('int64')
        f1 = f1_score(anomaly_labels_list, best_trace_level_detection)
        print(f"best_threshhold-{get_best_threshhold(th_df).name.item()}-F1分数:",f1)
        print("--" * 15)
        precision, recall = calculate_precision_recall(trace_level_detection2, anomaly_labels_list)

        print("精确度 (Precision):", precision)
        print("召回率 (Recall):", recall)