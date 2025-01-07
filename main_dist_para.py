import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import argparse
import yaml
import torch
from lib.utils import load_graph_data

from model.pytorch.dist_para import ModelsDist
from model.pytorch.dist_train_teacher import ModelsTearcher
def main(args):

    with open(args.config_filename,encoding='utf-8') as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)
        print('config_filename:',args.config_filename)
        #supervisor_config is the config.yaml information
        #print(adj_mx)
        #adj_mx改变DCRNN节点结构，交通流数据是输入与输出

        if args.load_pretrained == True:
            supervisor = ModelsDist( args.models, args.load_pretrained, args.config_filename,args.cuda,**supervisor_config)
        else: 
            
            supervisor = ModelsTearcher( args.models, args.load_pretrained, args.config_filename,args.cuda,**supervisor_config)
        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=None, type=str)
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')

    parser.add_argument('--load_pretrained', action='store_true', help='Whether to load a pretrained model.')
    parser.add_argument('--cuda', default='cuda:0', type = str)
    args = parser.parse_args()
    main(args)
