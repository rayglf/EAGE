import os
from model.train import train
from dataset_sabpad import Dataset

def main(dataset,n_epochs=30,batch_size =64,lr=0.0002 ,b1=0.5 ,b2=0.999 ,seed=None,hidden_dim = 64 , GAT_heads = 4,decoder_num_layers=2,threshold= 0.98,TF_styles:str='FAP'):
    '''
    :param dataset: instance of Dataset
    :param n_epochs:  number of epochs of training
    :param batch_size:
    :param lr: adam: learning rate
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :param seed: value of Pytorch random seed
    :param enc_hidden_dim: hidden dimensional of encoder_GAT
    :param GAT_heads: heads of first layer of GATs
    :param decoder_num_layers: numberf of layers  of decoder_GRU
    :param dec_hidden_dim: hidden dimensional of decoder_GRU
    :param TF_styles: teacher forcing styles
    :return:
    '''
    if TF_styles not in ['AN','PAV', 'FAP']:
        raise Exception('"TF_styles" must be a value in ["AN","PAV", "FAP"]')

    train(dataset, n_epochs, batch_size, lr, b1, b2, seed, hidden_dim, GAT_heads, decoder_num_layers,threshold, TF_styles)

if __name__ == '__main__':
    attr_keys = ['activity']
    em_size = [[3, 15]]
    file = os.path.abspath('../../../data/csv/binet_logs/bpic12-0.3-1.csv.gz')
    dataset = Dataset(file, attr_keys,em_size)
    main(dataset, n_epochs=30, lr=0.0006, decoder_num_layers=2, batch_size=32, hidden_dim=64,threshold= 0.98, TF_styles='FAP')




