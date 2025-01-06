import os
from collections import defaultdict, Counter
import pickle
import pandas as pd
import scipy.sparse as sq
import networkx as nx
import numpy as np
#from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import LabelEncoder
import torch
from model import feature_maps_d as fm

# Cell
def df_preproc (df,cols=['activity'],start_marker='▶',end_marker='■'):
#def df_preproc(df, cols=['activity'], start_marker='###start###', end_marker='###end###'):
    # Add event log column
    df['event_id']=df.groupby('trace_id').cumcount()+1
    # Work with numpy for performance boost
    eid_col = df.columns.to_list().index('event_id')
    # Get col idx
    col_idx=[df.columns.to_list().index(c) for c in cols]
    data= df.values
    # Add Start Events
    idx= np.where(data[:,eid_col]==1)[0] # start idx
    new = data[idx].copy()

    for c in col_idx: new[:,c]=start_marker
    new[:,eid_col]=0
    data = np.insert(data,idx,new, axis=0)
    # Add End Events
    idx= np.where(data[:,eid_col]==0)[0][1:] # start idx without the first
    new = data[idx-1].copy() # get data from current last idx
    for c in col_idx: new[:,c]=end_marker

    new[:,eid_col]=new[:,eid_col]+1
    data = np.insert(data,idx,new, axis=0)
    # take care of final last event
    last= data[-1].copy()
    for c in col_idx: last[c]=end_marker

    last[eid_col]+=1
    data = np.insert(data,len(data),last, axis=0)
    df = pd.DataFrame(data,columns=df.columns)


    return df
# Cell
def import_log(log_path,cols=['activity']):
    df=pd.read_csv(log_path)
    df.rename({'name':'activity','org:resource':'resource','action_code':'action'},axis=1,inplace=True)
    if not 'activity' in df.columns:
        df.rename({'concept:name':'activity'},axis=1,inplace=True)
    df.rename({'case:pdc:isPos':'normal'},axis=1,inplace=True)
    df = df_preproc(df,cols)
    df.index=df.trace_id
    return df
# Cell

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class Dataset(object):
    #def __init__(self, dataset_Path, attr_keys, beta=555):
    def __init__(self, dataset_Path, attr_keys,em_size, beta=1):
        # Public properties
        self.dataset_name = dataset_Path
        self.attr_keys=attr_keys
        self.edge_indexs = []
        self.nodes = []
        self.graphs=[]
        self.g_labels=[]
        self.anomaly_labels=[]
        self.edges = []
        #self.trace_xs = []
        self.em_size=em_size
        self.node_xs = []
        self.case_dims = []
        self.attribute_embedding_dims=[]
        self.case_ids=[]
        self.beta = beta

        event_log = import_log(self.dataset_name,  self.attr_keys)
        traceid = np.array(event_log['trace_id'])

        self.case_lens=[]
        for i in event_log['trace_id'].unique():
            self.case_ids.append(i)
        for i in event_log['trace_id'].unique():
            self.case_lens.append(Counter(traceid)[i])

        anomaly = []
        for i in event_log['anomaly']:
            anomaly.append(i)
        it = 1
        for i in range(len(self.case_lens)):
            if anomaly[it] == 'normal':
                self.anomaly_labels.append(0)
            else:
                self.anomaly_labels.append(1)
            it = it + self.case_lens[i]

        feature_columns = defaultdict(list)
        for c in  self.attr_keys:
            for i in event_log[c]:
                feature_columns[c].append(i)
        for key in feature_columns.keys():
            encoder = LabelEncoder()
            feature_columns[key] = encoder.fit_transform(feature_columns[key]) + 1
        #print(feature_columns['activity'][0:50])

        # Transform back into sequences
        case_lens = np.array(self.case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        self.features = [np.zeros((case_lens.shape[0], case_lens.max()), dtype=int) for _ in range(len(feature_columns))]

        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            #print("(offset, case_len):",i, offset, case_len)
            for k, key in enumerate(feature_columns):
                # print("k, key:",k, key)
                x = feature_columns[key]
                self.features[k][i, :case_len] = x[offset: offset + case_len]

        # 获得图节点
        for i in range(len(self.attr_keys)):
            node=[]
            for j, (offset, case_len) in enumerate(zip(offsets, case_lens)):
                x = feature_columns[self.attr_keys[i]]
                node.append(np.array(x[offset: offset + case_len]))
            self.nodes.append(node)

        self._graphs_edges_()

        for i in range(len(self.attr_keys)):
            g_l=[]
            gr=[]
            for j in range(self.num_cases):
                g = nx.Graph()
                g.add_nodes_from(np.array(self.nodes[i][j]))
                g.add_edges_from(np.array(self.edges[i][j]))
                g_l.append(g.nodes)
                gr.append(sq.lil_matrix(nx.adjacency_matrix(g)).todense())
            self.g_labels.append(g_l)
            self.graphs.append(gr)

        self._gen_trace_nodes()

        for i,dim in enumerate(self.attribute_dims):
            self.attribute_embedding_dims.append(np.array(self.node_xs[0][i]).shape[1])


    def _graphs_edges_(self):
        for i in range(len(self.attr_keys)):
            graph_relation = np.zeros((self.attribute_dims[i] + 1, self.attribute_dims[i] + 1), dtype='int32')
            # print("graph_relation:", graph_relation.shape)
            for case_index in range(self.num_cases):
                if self.case_lens[case_index] > 1:
                    for activity_index in range(1, self.case_lens[case_index]):
                        graph_relation[self.features[i][case_index][activity_index - 1], self.features[i][case_index][
                            activity_index]] += 1
            # print("graph_relation:",graph_relation.shape)
            dims_temp = []
            dims_temp.append(self.attribute_dims[i])
            for j in range(1, len(self.attribute_dims)):
                dims_temp.append(dims_temp[j - 1] + self.attribute_dims[j])
            dims_temp.insert(0, 0)
            dims_range = [(dims_temp[i - 1], dims_temp[i]) for i in range(1, len(dims_temp))]

            graph_relation = np.array(graph_relation >= self.beta * self.num_cases, dtype='int32')
            # print("dims_range:",dims_range)
            # print("graph_relation:", graph_relation)

            onehot_features = self.flat_onehot_features
            ed = []
            for case_index in range(self.num_cases):  # 生成图
                edge = []

                # xs = []
                # ##构造顶点信息
                # for attr_index in range(self.num_attributes):
                #     xs.append(
                #         torch.tensor(onehot_features[case_index, :, dims_range[attr_index][0]:dims_range[attr_index][1]]))

                if self.case_lens[case_index] > 1:
                    ##构造边信息
                    node = self.features[i][case_index, :self.case_lens[case_index]]

                    for activity_index in range(0, self.case_lens[case_index]):
                        out = np.argwhere(graph_relation[self.features[i][case_index, activity_index]] == 1).flatten()
                        a = set(node)
                        b = set(out)

                        if activity_index + 1 < self.case_lens[case_index]:
                            # 保证trace中相连的activity一定有边。
                            edge.append([node[activity_index], node[activity_index + 1]])

                        for node_name in a.intersection(b):
                            for node_index in np.argwhere(node == node_name).flatten():
                                if activity_index + 1 != node_index:
                                    edge.append([node[activity_index], node[node_index]])  # 添加有向边

                edge_index = torch.tensor(edge, dtype=torch.long)
                ed.append(edge_index)
            self.edges.append(ed)

    def __len__(self):
        return self.num_cases

    def _gen_trace_nodes(self):
        sa=self.node_em
        for case_index in range(self.num_cases):
            xs = []
            for attr_index in range(self.num_attributes):
                xs.append(torch.tensor(sa[attr_index][case_index, :, :]))
            self.node_xs.append(xs)
        self.case_dims = self.attribute_dims.copy()
    @property
    def node_em(self):
        feature=[]
        for i in range(len(self.attr_keys)):

            file_path = f'em_result_{i}.pkl'

            # 检查文件是否存在
            if os.path.exists(file_path):
                print(f"发现 em_result_{i}.pkl")
                with open(f'em_result_{i}.pkl', 'rb') as f:
                    sa = pickle.load(f)
            else:
                print(f"没有发现 em_result_{i}.pkl")
                sa = fm.wl_subtree_feature_map(self.num_cases, self.graphs[i], self.g_labels[i], self.em_size[i][0],
                                               self.em_size[i][1])
                with open(f'em_result_{i}.pkl', 'wb') as f:
                    pickle.dump(sa, f)

            feature0 = np.zeros(
                [self.onehot_features[i].shape[0], self.onehot_features[i].shape[1],
                 self.em_size[i][0] * self.em_size[i][1]])

            for num in range(self.num_cases):
                for nn in range(self.onehot_features[i].shape[1]):
                    if self.features[i][num][nn] == 0:
                        break
                    else:
                        aa=np.where(self.g_labels[i][num]==self.features[i][num][nn])
                        feature0[num][nn]=np.array(sa[num][aa[0][0]])
            feature.append(np.array(feature0))

        return feature

    @property
    def num_cases(self):
        """Return number of cases in the event log, i.e., the number of examples in the dataset."""
        return len(self.features[0])

    @property
    def num_events(self):
        """Return the total number of events in the event log."""
        return sum(self.case_lens)

    @property
    def max_len(self):
        """Return the length of the case with the most events."""
        return self.features[0].shape[1]

    @property
    def mask(self):
        #self._mask = np.zeros([self.max_h*self.feature_size,self.features[0].shape[1],self.features[0].shape[2]], dtype=bool)
        self._mask = np.zeros(self.features[0].shape,dtype=bool)
        for m, j in zip(self._mask, self.case_lens):
            m[:j] = True
        return self._mask

    @property
    def attribute_dims(self):
        # attribute_dims=np.asarray([int(f.max()) for f in self.features])
        # attribute_dims[0]=self.max_h*self.feature_size
        # return attribute_dims
        return np.asarray([int(f.max()) for f in self.features])

    @property
    def num_attributes(self):
        """Return the number of attributes in the event log."""
        return len(self.features)

    @property
    def onehot_features(self):
        """
        Return one-hot encoding of integer encoded features

        As `features` this will return one tensor for each attribute. Shape of tensor for each attribute will be
        (number_of_cases, max_case_length, attribute_dimension). The attribute dimension refers to the number of unique
        values of the respective attribute encountered in the event log.

        :return:
        """
        return [to_categorical(f)[:, :, 1:] for f in self.features]

    @property
    def flat_onehot_features(self):
        """
        Return combined one-hot features in one single tensor.

        One-hot vectors for each attribute in each event will be concatenated. Resulting shape of tensor will be
        (number_of_cases, max_case_length, attribute_dimension[0] + attribute_dimension[1] + ... + attribute_dimension[n]).

        :return:
        """
        return np.concatenate(self.onehot_features, axis=2)
