import torch
import numpy as np
import time
import os
import util
from engine import trainer

class arg:
    def __init__(self):
        self.device= "2"
        self.data ='data/PEMS-BAY'
        self.adjdata ='data/PEMS-BAY/adj_mx_bay.pkl'
        self.seq_length = 12
        self.nhid=40
        self.in_dim=2
        self.num_nodes = 325
        self.batch_size = 64
        self.learning_rate =0.001
        self.dropout=0.3
        self.weight_decay =0.0001
        self.clip=3
        self.lr_decay_rate=0.97
        self.epochs=200
        self.top_k=4
        self.print_every=100
        self.save ='./garage/metr-la'
        self.seed=530302
'''
python train.py --data "data/METR-LA" --adjdata "data/METR-LA/adj_mx.pkl" --in_dim 2 --num_nodes 207

'''
args = arg()
dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)

def setup_seed(seed):
    np.random.seed(seed) # Numpy module
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU
    
def get_engine():

    setup_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    adj_mx = util.load_adj(args.adjdata)
    supports = [torch.tensor(i).cuda() for i in adj_mx]
    H_a, H_b, H_T_new, lwjl, G0, G1, indices, G0_all, G1_all = util.load_hadj(args.adjdata, args.top_k)
    
    scaler = dataloader['scaler']

    lwjl = (((lwjl.t()).unsqueeze(0)).unsqueeze(3)).repeat(args.batch_size, 1, 1, 1)

    H_a = H_a.cuda()
    H_b = H_b.cuda()
    G0 = torch.tensor(G0).cuda()
    G1 = torch.tensor(G1).cuda()
    H_T_new = torch.tensor(H_T_new).cuda()
    lwjl = lwjl.cuda()
    indices = indices.cuda()

    G0_all = torch.tensor(G0_all).cuda()
    G1_all = torch.tensor(G1_all).cuda()

    engine = trainer(args.batch_size, scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, supports, H_a, H_b, G0, G1, indices,
                     G0_all, G1_all, H_T_new, lwjl, args.clip, args.lr_decay_rate) 
                     
    print("engine get!")                 
    return engine   
    
    
def get_time_T(engine):
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()
    realy = realy.transpose(1,3)[:,0,:,:]
    #print("really.shape:")
    #print(realy.shape)
    
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())
        

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]
    #print(yhat.shape)
    
    return yhat
   
    
    
    
    
    
    
    
    
    