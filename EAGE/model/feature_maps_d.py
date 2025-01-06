import networkx as nx
from gensim import corpora
import gensim
from model import breadth_first_search as bfs
from collections import defaultdict
import numpy as np
import copy, pickle
import random
#import pynauty
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix


from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# def get_graphlet(window, nsize):
    # """
    # This function takes the upper triangle of a nxn matrix and computes its canonical map
    # """
    # adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i!=idx] for idx, edge in enumerate(window)}

    # g = pynauty.Graph(number_of_vertices=nsize, directed=False, adjacency_dict = adj_mat)
    # cert = pynauty.certificate(g)
    # return cert

def compute_centrality(adj):
    n = adj.shape[0]
    adj = adj + np.eye(n)
    cen = np.zeros(n)
    G = nx.from_numpy_matrix(adj)
    nodes = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-4)
    for i in range(len(nodes)):
        cen[i] = nodes[i]

    return cen
def get_maps(n):
    # canonical_map -> {canonical string id: {"graph", "idx", "n"}}
    file_counter = open("canonical_maps/canonical_map_n%s.p"%n, "rb")
    canonical_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    # weight map -> {parent id: {child1: weight1, ...}}
    file_counter = open("graphlet_counter_maps/graphlet_counter_nodebased_n%s.p"%n, "rb")
    weight_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    weight_map = {parent: {child: weight/float(sum(children.values())) for child, weight in children.items()} for parent, children in weight_map.items()}
    child_map = {}
    for parent, children in weight_map.items():
        for k,v in children.items():
            if k not in child_map:
                child_map[k] = {}
            child_map[k][parent] = v
    weight_map = child_map
    return canonical_map, weight_map


def adj_wrapper(g):
    am_ = g["al"]
    size = max(np.shape(am_))
    am = np.zeros((size, size))
    for idx, i in enumerate(am_):
        for j in i:
            am[idx][j-1] = 1
    return am


# def graphlet_feature_map(num_graphs, graphs, num_graphlets, samplesize):
#     # if no graphlet is found in a graph, we will fall back to 0th graphlet of size k
#     fallback_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
#     canonical_map, weight_map = get_maps(num_graphlets)
#     canonical_map1, weight_map1 = get_maps(2)
#     # randomly sample graphlets
#     graph_map = {}
#     graphlet_graph = []
#     for gidx in range(num_graphs):
#         #print(gidx)
#         am = graphs[gidx]
#         m = len(am)
#         for node in range(m):
#             graphlet_node = []
#             for j in range(samplesize):
#                 rand = np.random.permutation(range(m))
#                 r = []
#                 r.append(node)
#                 for ele in rand:
#                     if ele != node:
#                         r.append(ele)
#
#                 for n in [num_graphlets]:
#                 #for n in range(3,6):
#                     if m >= num_graphlets:
#                         window = am[np.ix_(r[0:n], r[0:n])]
#                         g_type = canonical_map[get_graphlet(window, n)]
#                         #for key, value in g_type.items():
#                         #    print(key.decode("utf-8"))
#                         #    print(value)
#                         graphlet_idx = str(g_type["idx".encode()])
#                     else:
#                         window = am[np.ix_(r[0:2], r[0:2])]
#                         g_type = canonical_map1[get_graphlet(window, 2)]
#                         graphlet_idx = str(g_type["idx".encode()])
#
#                     graphlet_node.append(graphlet_idx)
#
#             graphlet_graph.append(graphlet_node)
#
#     dictionary = corpora.Dictionary(graphlet_graph)
#     corpus = [dictionary.doc2bow(graphlet_node) for graphlet_node in graphlet_graph]
#     M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
#     M = normalize(M, norm='l1', axis=0)
#     M = M.T
#
#
#
#     allFeatures = {}
#     index = 0
#     for gidx in range(num_graphs):
#         adj = graphs[gidx]
#         n = adj.shape[0]
#         graphlet_feature = M[index:index + n, :]
#         index += n
#         allFeatures[gidx] = graphlet_feature
#
#     return allFeatures


def assign_id(cp):
    cp_with_ids = []
    for idx, val in enumerate(cp):
        line_id = "g_{}".format(idx)
        cp_with_ids.append(TaggedDocument(val, [line_id]))
    return cp_with_ids


def wl_subtree_feature_map(num_graphs, graphs, labels, max_h,feature_size):
    #print("graphs:",graphs)

    alllabels = {}
    label_lookup = {}
    label_counter = 0
    #wl_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in range(num_graphs)} for it in range(-1,5)}
    num_sample = 0

    #alllabels[0] = labels
    new_labels = {}
    # initial labeling
    biaoqianshu=set()
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        #print("=====",adj)
        n = adj.shape[0]

        if n >= num_sample:
            num_sample = n
        new_labels[gidx] = np.zeros(n, dtype=np.int32)

        try:
            label=[]
            for i in range(n):
                label.append(int(labels[gidx][i][0]))
        except:
            label = labels[gidx]


        #计算标签数
        for i in label:
            biaoqianshu.add(i)
        #print("biaoqianshu:", biaoqianshu)

        label=np.array(label)

        #print("label:", label)
        for node in range(n):
            la = label[node]
            if la not in label_lookup:
                label_lookup[la] = label_counter
                new_labels[gidx][node] = label_counter
                label_counter += 1
            else:
                new_labels[gidx][node] = label_lookup[la]
            #wl_graph_map[-1][gidx][label_lookup[la]] = wl_graph_map[-1][gidx].get(label_lookup[la], 0) + 1
    compressed_labels = copy.deepcopy(new_labels)
    alllabels[0]=new_labels
    # WL iterations started
    for it in range(5):
        print('it:',it)
        label_lookup = {}
        label_counter = 0
        for gidx in range(num_graphs):
            #if gidx%100==0:
            #    print(gidx)
            adj = graphs[gidx]
            n = adj.shape[0]
            nx_G = nx.from_numpy_matrix(adj)
            #nx_G = nx.from_numpy_array(adj)

            for node in range(n):
                node_label = tuple([new_labels[gidx][node]])
                neighbors = []
                edges = list(bfs.bfs_edges(nx_G, np.zeros(n), source=node, depth_limit=1))
                for u, v in edges:
                    neighbors.append(v)

                if len(neighbors) > 0:
                    neighbors_label = tuple([new_labels[gidx][i] for i in neighbors])
                    node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                if node_label not in label_lookup:
                    label_lookup[node_label] = str(label_counter)
                    compressed_labels[gidx][node] = str(label_counter)
                    label_counter += 1
                else:
                    compressed_labels[gidx][node] = label_lookup[node_label]
                #wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label],0) + 1
        # print("Number of compressed labels at iteration %s: %s"%(it, len(label_lookup)))
        new_labels = copy.deepcopy(compressed_labels)
        # print("labels")
        # print(labels)
        alllabels[it + 1] = new_labels
        #print('wl_graph_map[it][gidx]',wl_graph_map[it][0])


    print("标签数："+str(len(biaoqianshu)))
    print(biaoqianshu)


    # subtrees_graph = []
    #
    # for gidx in range(num_graphs):
    #     adj = graphs[gidx]
    #     n = adj.shape[0]
    #     #gcp=[]
    #     for node in range(n):
    #         subtrees_node = []
    #         #gcp.append(alllabels[1][gidx][node])
    #         for it in range(5):
    #             graph_label = alllabels[it]
    #             label = graph_label[gidx]
    #             subtrees_node.append(str(label[node]))
    #
    #         subtrees_graph.append(subtrees_node)
        
        #sgcp=sorted(gcp)
        #ssgcp=[str(lb) for lb in sgcp]
        #cp.append(ssgcp)
    
    cps={}
    cp0=[]
    cp1=[]
    cp2=[]
    cp3=[]
    cp4=[]
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = adj.shape[0]
        gcp0=[]
        gcp1=[]
        gcp2=[]
        gcp3=[]
        gcp4=[]
        sgcp0=[]
        sgcp1=[]
        sgcp2=[]
        sgcp3=[]
        sgcp4=[]
        for node in range(n):#节点对应子树模式
            gcp0.append(alllabels[1][gidx][node])
            gcp1.append(alllabels[2][gidx][node])
            gcp2.append(alllabels[3][gidx][node])
            gcp3.append(alllabels[4][gidx][node])
            gcp4.append(alllabels[5][gidx][node])


        #排序
        cen = compute_centrality(adj)
        sub = np.argsort(-cen)
        #print(cen)
        #print(sub)
        for i in range(n):
            sgcp0.append(gcp0[sub[i]])
            sgcp1.append(gcp1[sub[i]])
            sgcp2.append(gcp2[sub[i]])
            sgcp3.append(gcp3[sub[i]])
            sgcp4.append(gcp4[sub[i]])
        # sgcp0=sorted(gcp0)
        # sgcp1=sorted(gcp1)
        # sgcp2=sorted(gcp2)
        # sgcp3=sorted(gcp3)
        # sgcp4=sorted(gcp4)
        
        
        # sgcp0=[]
        # sgcp1=[]
        # sgcp2=[]
        # random.shuffle(gcp0)
        # sgcp0+=gcp0
        # random.shuffle(gcp0)
        # sgcp0+=gcp0
        # random.shuffle(gcp0)
        # sgcp0+=gcp0
        # random.shuffle(gcp0)
        # sgcp0+=gcp0
        # random.shuffle(gcp0)
        # sgcp0+=gcp0
        
        # random.shuffle(gcp1)
        # sgcp1+=gcp1
        # random.shuffle(gcp1)
        # sgcp1+=gcp1
        # random.shuffle(gcp1)
        # sgcp1+=gcp1
        # random.shuffle(gcp1)
        # sgcp1+=gcp1
        # random.shuffle(gcp1)
        # sgcp1+=gcp1
        
        # random.shuffle(gcp2)
        # sgcp2+=gcp2
        # random.shuffle(gcp2)
        # sgcp2+=gcp2
        # random.shuffle(gcp2)
        # sgcp2+=gcp2
        # random.shuffle(gcp2)
        # sgcp2+=gcp2
        # random.shuffle(gcp2)
        # sgcp2+=gcp2
        
        ssgcp0=[str(lb) for lb in sgcp0]
        ssgcp1=[str(lb) for lb in sgcp1]
        ssgcp2=[str(lb) for lb in sgcp2]
        ssgcp3=[str(lb) for lb in sgcp3]
        ssgcp4=[str(lb) for lb in sgcp4]
        cp0.append(ssgcp0)
        cp1.append(ssgcp1)
        cp2.append(ssgcp2)
        cp3.append(ssgcp3)
        cp4.append(ssgcp4)
    cps[0]=cp0  #   =[[str]]
    cps[1]=cp1
    cps[2]=cp2
    cps[3]=cp3
    cps[4]=cp4
    #print(cp)
    vs=feature_size
    eps=100
    #model0=gensim.models.Word2Vec(cps[0], window=10, min_count=0, vector_size=vs, sg=1, negative=5,epochs=eps)
    #model1=gensim.models.Word2Vec(cps[1], window=10, min_count=0, vector_size=vs, sg=1, negative=5,epochs=eps)
    #model2=gensim.models.Word2Vec(cps[2], window=10, min_count=0, vector_size=vs, sg=1, negative=5,epochs=eps)
    
    cp_with_ids0 = assign_id(cps[0])
    cp_with_ids1 = assign_id(cps[1])
    cp_with_ids2 = assign_id(cps[2])
    cp_with_ids3 = assign_id(cps[3])
    cp_with_ids4 = assign_id(cps[4])
     
    d2v_dbow0 = Doc2Vec(min_count=0, vector_size=vs,  epochs=eps, dm=0, dbow_words=1, negative=10, sample=1e-3, hs=0, window=5, workers=12)
    d2v_dbow1 = Doc2Vec(min_count=0, vector_size=vs,  epochs=eps, dm=0, dbow_words=1, negative=10, sample=1e-3, hs=0, window=5, workers=12)
    d2v_dbow2 = Doc2Vec(min_count=0, vector_size=vs,  epochs=eps, dm=0, dbow_words=1, negative=10, sample=1e-3, hs=0, window=5, workers=12)
    d2v_dbow3 = Doc2Vec(min_count=0, vector_size=vs,  epochs=eps, dm=0, dbow_words=1, negative=10, sample=1e-3, hs=0, window=5, workers=12)
    d2v_dbow4 = Doc2Vec(min_count=0, vector_size=vs,  epochs=eps, dm=0, dbow_words=1, negative=10, sample=1e-3, hs=0, window=5, workers=12)
    
    
    d2v_dbow0.build_vocab(cp_with_ids0)
    d2v_dbow1.build_vocab(cp_with_ids1)
    d2v_dbow2.build_vocab(cp_with_ids2)
    d2v_dbow3.build_vocab(cp_with_ids3)
    d2v_dbow4.build_vocab(cp_with_ids4)
    
    for run in range(2):
        #print("run={}".format(run))

        all_reviews0 = cp_with_ids0[:]
        all_reviews1 = cp_with_ids1[:]
        all_reviews2 = cp_with_ids2[:]
        all_reviews3 = cp_with_ids3[:]
        all_reviews4 = cp_with_ids4[:]

        random.shuffle(all_reviews0)
        random.shuffle(all_reviews1)
        random.shuffle(all_reviews2)
        random.shuffle(all_reviews3)
        random.shuffle(all_reviews4)


        d2v_dbow0.train(all_reviews0, total_examples=d2v_dbow0.corpus_count, epochs=100)
        d2v_dbow1.train(all_reviews1, total_examples=d2v_dbow1.corpus_count, epochs=100)
        d2v_dbow2.train(all_reviews2, total_examples=d2v_dbow2.corpus_count, epochs=100)
        d2v_dbow3.train(all_reviews3, total_examples=d2v_dbow3.corpus_count, epochs=100)
        d2v_dbow4.train(all_reviews4, total_examples=d2v_dbow4.corpus_count, epochs=100)

    
    #vecs = [np.array(d2v_dbow0.docvecs[z.tags[0]]).reshape((1, 10)) for z in cp_with_ids0]
    
    
    #print(dir(d2v_dbow0))
    
    
    #print(len(d2v_dbow0.wv))
    #bb=input("PPP")
    
    

    # dictionary = corpora.Dictionary(subtrees_graph)
    # #print(dictionary.token2id)
    # corpus = [dictionary.doc2bow(subtrees_node) for subtrees_node in subtrees_graph]
    # # one corpora for a node
    #
    #
    # print(len(corpus))
    # print('corpus:0')
    # print(corpus[0])
    # print(subtrees_graph[0])
    # print(print('corpus:1'))
    # print(corpus[1])
    # print(subtrees_graph[1])
    # print(print('corpus:2'))
    # print(corpus[2])
    # print(subtrees_graph[2])
    # print()
    # M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    # M = M.T
    #one csr in M for a node，eg. M[0][0] for node 0 of graph 0, M[1][0] for node 1 of graph 0
    #len(M[0])==1,M[0][0]
    # print(len(graphs[0]))
    # print(len(graphs[1]))
    
    #aaa=input("pause")

    #print(d2v_dbow4.wv.key_to_index)
    #print(d2v_dbow0.wv['61745'])
    #print(d2v_dbow0.wv.vectors[0,:])

    allFeatures = {}
    allFeatures2 = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = adj.shape[0]
        #print(len(alllabels))
        
        
        #subtree_feature = M[index:index + n, :]
        
        node_lbs0=[alllabels[1][gidx][node] for node in range(n)]
        node_lbs1=[alllabels[2][gidx][node] for node in range(n)]
        node_lbs2=[alllabels[3][gidx][node] for node in range(n)]
        node_lbs3=[alllabels[4][gidx][node] for node in range(n)]
        node_lbs4=[alllabels[5][gidx][node] for node in range(n)]
        
        
        #temp subtree feature of i-hop
        tsf0=[d2v_dbow0.wv.get_vector(ii) for ii in node_lbs0]
        tsf1=[d2v_dbow1.wv.get_vector(ii) for ii in node_lbs1]
        tsf2=[d2v_dbow2.wv.get_vector(ii) for ii in node_lbs2]
        tsf3=[d2v_dbow3.wv.get_vector(ii) for ii in node_lbs3]
        tsf4=[d2v_dbow4.wv.get_vector(ii) for ii in node_lbs4]
        
        tfs=np.zeros((num_sample, max_h*vs))####################################注意修改
        for i1 in range(n):
            if max_h==1:
                tfs[i1, 0:vs] = tsf0[i1][0:vs]
            if max_h == 2:
                tfs[i1, 0:vs] = tsf0[i1][0:vs]
                tfs[i1, vs:2 * vs] = tsf1[i1][0:vs]
            if max_h == 3:
                tfs[i1, 0:vs] = tsf0[i1][0:vs]
                tfs[i1, vs:2 * vs] = tsf1[i1][0:vs]
                tfs[i1, 2 * vs:3 * vs] = tsf2[i1][0:vs]
            if max_h == 4:
                tfs[i1, 0:vs] = tsf0[i1][0:vs]
                tfs[i1, vs:2 * vs] = tsf1[i1][0:vs]
                tfs[i1, 2 * vs:3 * vs] = tsf2[i1][0:vs]
                tfs[i1, 3 * vs:4 * vs] = tsf3[i1][0:vs]
            if max_h == 5:
                tfs[i1, 0:vs] = tsf0[i1][0:vs]
                tfs[i1, vs:2 * vs] = tsf1[i1][0:vs]
                tfs[i1, 2 * vs:3 * vs] = tsf2[i1][0:vs]
                tfs[i1, 3 * vs:4 * vs] = tsf3[i1][0:vs]
                tfs[i1, 4 * vs:5 * vs] = tsf4[i1][0:vs]


        
        
        index += n
        #allFeatures[gidx] = subtree_feature
        tfs=np.array(tfs)
        allFeatures[gidx] = tfs
        
        
        
        
    #print('allFeatures 0')
    #print(allFeatures[0].shape)
    #allFeatures[i] csr matrix for graph i
    #allFeatures[i].shape (r,c),r:|G|,c:features
    #bb=input("pause2")

    return allFeatures


def shortest_path_feature_map(num_graphs, graphs, labels):
    sp_graph = []
    sp_graph2 = []
    
    label_lookup = {}
    label_counter = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = adj.shape[0]
        label = labels[gidx]
        nx_G = nx.from_numpy_matrix(adj)
        
        for i in range(n):
            sp_node = []
            sp_node2=[]
            for j in range(n):
                if i != j:
                    try:
                        path = list(nx.shortest_path(nx_G, i, j))
                    except nx.exception.NetworkXNoPath:
                        continue

                    if not path:
                        continue
                    if label[i] <=label[j]:
                        sp_label = str(int(label[i])) + ',' + str(int(label[j])) + ',' + str(len(path))
                    else:
                        sp_label = str(int(label[j])) + ',' + str(int(label[i])) + ',' + str(len(path))
                    sp_node.append(sp_label)
                    if sp_label not in label_lookup:
                        label_lookup[sp_label]=label_counter
                        sp_node2.append(label_counter)
                        label_counter+=1
                    else:
                        sp_node2.append(label_lookup[sp_label])
                    
            sp_graph.append(sp_node)
            sp_graph2.append(sp_node2)

    dictionary = corpora.Dictionary(sp_graph)
    corpus = [dictionary.doc2bow(sp_node) for sp_node in sp_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = adj.shape[0]
        sp_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = sp_feature
    
    
    # cp=[]
    # for gidx in range(1):
        # adj = graphs[300]
        # n = len(adj)
        # print(adj)
        # tcp=[]
        # for i in range(n):
            # print(sp_graph2[100])
            # #tcp+=sp_graph2[gidx]
        # cp.append(tcp)
    
    # print(cp[0])

    # bbb=input('pause')

    return allFeatures