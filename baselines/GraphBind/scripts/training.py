import sys
import os
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
import prettytable as pt
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torchnet import meter
import pickle
import argparse

from torch_geometric.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from data_io import NeighResidue3DPoint
from ModelCode.GN_model_gru import MetaBind_MultiEdges
from valid_metrices import *



def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", help="A ligand type. It can be chosen from DNA,RNA,CA,MG,MN,ATP,HEME.")
    parser.add_argument("--psepos", dest="psepos", default='SC',
                        help="Pseudo position of residues. SC, CA, C stand for centroid of side chain, alpha-C atom and centroid of residue, respectively.")
    parser.add_argument("--features", dest="features", default='PSSM,HMM,SS,AF',
                        help="Feature groups. Multiple features should be separated by commas. You can combine features from PSSM, HMM, SS(secondary structure) and AF(atom features).")
    parser.add_argument("--context_radius", dest="context_radius",type=int,help="Radius of structure context.")
    parser.add_argument("--trans_anno", dest="trans_anno",type=bool, default=True,
                        help="Transfer binding annotations for DNA-(RNA-)binding protein training data sets or not.")
    parser.add_argument("--edge_radius", dest='edge_radius',type=int, default=10,
                        help='Radius of the neighborhood of a node. It should be smaller than radius of structure context.')
    parser.add_argument("--use_GRU", dest='use_GRU',type=bool, default=True, help='Use GRU or not.')
    parser.add_argument("--apply_edgeattr", dest='apply_edgeattr',type=bool, default=True, help='Use the edge feature vectors or not. ')
    parser.add_argument("--apply_posemb", dest='apply_posemb',type=bool, default=True,
                        help='Use the relative distance from every node to the central node as position embedding of nodes or not')
    parser.add_argument("--hidden_size", dest='hidden_size',type=int, default=64,
                        help='The dimension of encoded edge, node and graph feature vector.')
    parser.add_argument("--aggr", dest='aggr', default='sum',
                        help='The aggregation operation in node update module and graph update module. You can choose frpm sum and max.')
    parser.add_argument("--gru_steps", dest='gru_steps',type=int, default=4, help='The number of GNN-blocks')
    parser.add_argument("--lr", dest='lr',type=float, default=0.00005, help='Learning rate for training the deep model.')
    parser.add_argument("--batch_size", dest='batch_size',type=int, default=64, help='Batch size for training deep model.')
    parser.add_argument("--epoch", dest='epoch',type=int, default=30, help='Training epochs.')
    return parser.parse_args()

def checkargs(args):
    if args.ligand is None:
        print('ERROR: please input ligand type!')
        raise ValueError
    if args.context_radius is None:
        print('ERROR: please input context_radius!')
        raise ValueError


    if args.ligand not in ['DNA','RNA','CA','MN','MG','ATP','HEME']:
        print('ERROR: ligand "{}" is not supported by GraphBind!'.format(args.ligand))
        raise ValueError
    if args.psepos not in ['SC','CA','C']:
        print('ERROR: pseudo position of a residue "{}" is not supported by GraphBind!'.format(args.psepos))
        raise ValueError
    features = args.features.strip().split(',')
    for feature in features:
        if feature not in ['PSSM','HMM','SS','AF']:
            print('ERROR: feature "{}" is not supported by GraphBind!'.format(feature))
            raise ValueError
    if args.context_radius<=0:
        print('ERROR: radius of structure context should be positive!')
        raise ValueError

    if args.edge_radius<=0:
        print('ERROR: radius of structure context should be positive!')
        raise ValueError
    elif args.edge_radius>=args.context_radius:
        print('ERROR: radius of structure context should be smaller than radius of structure context!')
        raise ValueError

    if args.aggr not in ['sum','max']:
        print('ERROR: aggregation operation "{}" is not supported by GraphBind!'.format(args.aggr))
        raise ValueError


    return

class Config():
    def __init__(self,args):

        self.ligand = 'P'+args.ligand if args.ligand != 'HEME' else 'PHEM'
        self.Dataset_dir = os.path.abspath('..') + '/Datasets/' + self.ligand
        self.psepos = args.psepos
        self.feature_combine = ''
        if 'PSSM' in args.features:
            self.feature_combine+='P'
        if 'HMM' in args.features:
            self.feature_combine+='H'
        if 'SS' in args.features:
            self.feature_combine+='S'
        if 'AF' in args.features:
            self.feature_combine+='A'

        self.dist = args.context_radius
        self.data_root_dir = '{}/{}_{}_dist{}_{}'.format(self.Dataset_dir, self.ligand, self.psepos,
                                                         self.dist, self.feature_combine)

        self.str_dataio = NeighResidue3DPoint
        self.trans_anno = args.trans_anno

        self.edge_method = 'radius'
        self.radius_list = [args.edge_radius]
        self.max_nn = 40
        self.str_model = MetaBind_MultiEdges
        self.stack_method = 'GRU' if args.use_GRU else 'Composed'
        self.apply_edgeattr = args.apply_edgeattr
        self.apply_nodeposemb = args.apply_posemb
        self.e_hs = args.hidden_size
        self.x_hs = args.hidden_size
        self.u_hs = args.hidden_size
        self.edge_aggr = ['add' if args.aggr == 'sum' else args.aggr]
        self.node_aggr = ['add' if args.aggr == 'sum' else args.aggr]
        self.gru_steps = args.gru_steps
        self.gru_heads = 1

        self.str_lr = args.lr
        self.bias = True
        self.dropratio=0.5
        self.L2_weight = 0
        self.max_metric = 'MCC'
        self.batch_size = args.batch_size
        self.test_batchsize = self.batch_size
        self.epoch = args.epoch
        self.num_workers = 0
        self.early_stop_epochs = 10
        self.saved_model_num = 1

        self.model_time = None
        self.train = True
        if self.model_time is not None:
            self.model_path = self.Dataset_dir+'/checkpoints/' + self.model_time
        else:
            localtime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
            self.model_path = self.Dataset_dir+'/checkpoints/' + localtime
            os.makedirs(self.model_path)
        self.submodel_path = None
        self.sublog_path = None

    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab',buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    else:
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


def train(opt,device,model,learning_rate,train_data,valid_data,test_data):



    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.num_workers,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=opt.L2_weight)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5,last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=10,
                                                           min_lr=1e-6)
    criterion = nn.BCELoss()
    epoch_begin = 0

    # print('** loss function: {}'.format(criterion))

    if opt.model_time is not None:
        model_path = '{}/model0.pth'.format(opt.submodel_path)
        if os.path.exists(model_path):
            print('Continue train model...')
            model_path = '{}/model0.pth'.format(opt.submodel_path)
            model, criterion, optimizer, _, epoch_begin= torch.load(model_path)
            print('epoch_begin:',epoch_begin)

    model.to(device)
    criterion.to(device)

    loss_meter = meter.AverageValueMeter()

    early_stop_iter = 0
    max_metric_val = -1
    nsave_model = 0
    begintime = datetime.datetime.now()
    print('Time:', begintime)
    for epoch in range(epoch_begin,opt.epoch):
        nstep = len(train_dataloader)
        for ii, data in enumerate(train_dataloader):
            model.train()
            data = data.to(device)
            # target = data.y.half()
            target = data.y
            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            if ii % (nstep-1) == 0 and ii != 0:
                nowtime = datetime.datetime.now()

                print('|| Epoch{} step{} || lr={:.6f} | train_loss={:.5f}'.format(epoch, ii,
                                                                                  optimizer.param_groups[0]['lr'],
                                                                                  loss_meter.mean))
                print('Time:',nowtime)
                print('Timedelta: %s seconds' % (nowtime - begintime).seconds)
                val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc = val(opt,device,model, valid_data, 'valid',val_th=None)
                _ = val(opt,device,model, test_data,'test',val_th)

                if opt.max_metric == 'AUC':
                    metrice_val = val_auc
                elif opt.max_metric == 'MCC':
                    metrice_val = val_mcc
                elif opt.max_metric == 'F1':
                    metrice_val = val_F1
                else:
                    print('ERROR: opt.max_metric.')
                    raise ValueError

                if metrice_val>max_metric_val:
                    max_metric_val = metrice_val
                    if nsave_model<opt.saved_model_num:
                        save_path = '{}/model{}.pth'.format(opt.submodel_path,nsave_model)
                        print('save net: ', save_path)
                        torch.save([model, criterion, optimizer,val_th,epoch], save_path)
                        nsave_model+=1
                    else:
                        save_path = '{}/model{}.pth'.format(opt.submodel_path, nsave_model-1)
                        print('save net: ', save_path)
                        for model_i in range(1,opt.saved_model_num):
                            os.system('mv {}/model{}.pth {}/model{}.pth'.format(opt.submodel_path, model_i,opt.submodel_path,model_i-1))
                        torch.save([model, criterion, optimizer,val_th,epoch], save_path)

                    early_stop_iter = 0
                else:
                    early_stop_iter += 1
                    if early_stop_iter == opt.early_stop_epochs:
                        break

                scheduler.step(metrice_val)
                loss_meter.reset()

    return

def val(opt,device,model, valid_data, dataset_type,val_th=None):
    valid_dataloader = DataLoader(valid_data, batch_size=opt.test_batchsize, shuffle=False, num_workers=opt.num_workers,pin_memory=True)

    model.eval()
    if val_th is not None:
        AUC_meter = meter.AUCMeter()
        Confusion_meter = meter.ConfusionMeter(k=2)
        with torch.no_grad():
            for ii, data in enumerate(valid_dataloader):
                data = data.to(device)
                target = data.y
                score = model(data).float()
                AUC_meter.add(score,target)
                pred_bi = target.data.new(score.shape).fill_(0)
                pred_bi[score > val_th] = 1
                Confusion_meter.add(pred_bi,target)

        val_auc = AUC_meter.value()[0]
        cfm = Confusion_meter.value()
        val_rec, val_pre, val_F1, val_spe, val_mcc = CFM_eval_metrics(cfm)
        print('{} result: '
              'th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f}'
              .format(dataset_type, val_th,val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc))
    else:
        AUC_meter = meter.AUCMeter()
        for j in range(2,100,2):
            th = j/100.0
            locals()['Confusion_meter_' + str(th)] = meter.ConfusionMeter(k=2)
        with torch.no_grad():
            for ii, data in enumerate(valid_dataloader):
                data = data.to(device)
                target = data.y
                score = model(data).float()
                AUC_meter.add(score, target)
                for j in range(2, 100, 2):
                    th = j / 100.0
                    pred_bi = target.data.new(score.shape).fill_(0)
                    pred_bi[score > th] = 1
                    locals()['Confusion_meter_' + str(th)].add(pred_bi,target)
        val_auc = AUC_meter.value()[0]
        val_rec, val_pre, val_F1, val_spe, val_mcc = -1,-1,-1,-1,-1
        for j in range(2, 100, 2):
            th = j / 100.0
            cfm = locals()['Confusion_meter_' + str(th)].value()
            rec, pre, F1, spe, mcc = CFM_eval_metrics(cfm)
            if mcc>val_mcc:
                val_rec, val_pre, val_F1, val_spe, val_mcc = rec, pre, F1, spe, mcc
                val_th = th
        print('{} result: '
              'th={:.2f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f}'
              .format(dataset_type, val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc))

    return val_th, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc

def test(opt,device,test_data):
    # for k, v in kwargs.items():
    #     setattr(opt, k, v)

    avg_test_probs = []
    avg_test_targets = []

    for model_i in range(opt.saved_model_num):
        model_path = '{}/model{}.pth'.format(opt.submodel_path,model_i)
        model, criterion, optimizer,th,_ = torch.load(model_path)

        model.to(device)
        model.eval()


        test_dataloader = DataLoader(test_data, batch_size=opt.test_batchsize, shuffle=False, num_workers=opt.num_workers,pin_memory=True)
        test_probs = []
        test_targets = []
        with torch.no_grad():
            for ii, data in enumerate(test_dataloader):
                data = data.to(device)
                target = data.y
                score = model(data).float()
                test_probs += score.tolist()
                test_targets += target.tolist()
        test_probs = np.array(test_probs)
        test_targets = np.array(test_targets)
        avg_test_probs.append(test_probs.reshape(-1,1))
        avg_test_targets.append(test_targets.reshape(-1,1))

    avg_test_probs = np.concatenate(avg_test_probs,axis=1)
    avg_test_probs = np.average(avg_test_probs,axis=1)

    avg_test_targets = np.concatenate(avg_test_targets,axis=1)
    avg_test_targets = np.average(avg_test_targets,axis=1)

    return avg_test_probs,avg_test_targets



def main(opt,device):
    print('=' * 89)
    print('||parameter||')
    opt.print_config()

    opt.submodel_path = opt.model_path + '/model'
    opt.sublog_path = opt.model_path + '/log'
    if not os.path.exists(opt.submodel_path): os.makedirs(opt.submodel_path)
    if not os.path.exists(opt.sublog_path): os.makedirs(opt.sublog_path)


    print('='*40 +'structure'+'='*40)

    train_data = opt.str_dataio(root=opt.data_root_dir, dataset='train')
    valid_data = opt.str_dataio(root=opt.data_root_dir, dataset='valid')
    test_data = opt.str_dataio(root=opt.data_root_dir, dataset='test')

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'NumRes', 'Pos', 'Neg', 'PNratio']
    tb.float_format = "2.3"

    Numres = train_data.data.y.shape[0]
    pos = torch.sum(train_data.data.y).item()
    neg = train_data.data.y.shape[0] - pos
    tb.add_row(['train', Numres, int(pos), int(neg), pos / float(neg)])

    Numres = valid_data.data.y.shape[0]
    pos = torch.sum(valid_data.data.y).item()
    neg = valid_data.data.y.shape[0] - pos
    tb.add_row(['valid', Numres, int(pos), int(neg), pos / float(neg)])
    if (Numres - 1) % opt.test_batchsize == 0:
        opt.test_batchsize += 1
        print('test_batchsize=', opt.test_batchsize)

    Numres = test_data.data.y.shape[0]
    pos = torch.sum(test_data.data.y).item()
    neg = test_data.data.y.shape[0] - pos
    tb.add_row(['test', Numres, int(pos), int(neg), pos / float(neg)])
    if (Numres - 1) % opt.test_batchsize == 0:
        opt.test_batchsize += 1
        print('test_batchsize=', opt.test_batchsize)

    print(tb)

    # multiedges
    model = opt.str_model(edge_aggr=opt.edge_aggr,node_aggr=opt.node_aggr,
                          gru_steps=opt.gru_steps,
                          x_ind=train_data.data.x.shape[1] + 1,
                          edge_ind=2,
                          x_hs=opt.x_hs,e_hs=opt.e_hs,u_hs=opt.u_hs,
                          dropratio=opt.dropratio, bias=opt.bias,
                          edge_method=opt.edge_method, r_list=opt.radius_list,dist=opt.dist,max_nn=opt.max_nn,
                          stack_method=opt.stack_method,
                          apply_edgeattr=opt.apply_edgeattr,apply_nodeposemb=opt.apply_nodeposemb)


    learning_rate = opt.str_lr


    if opt.train:
        print('=====train=====')
        train(opt,device,model,learning_rate,train_data,valid_data,test_data)
    print('=====test=====')

    valid_probs, valid_labels = test(opt,device,valid_data)
    test_probs, test_labels = test(opt,device,test_data)

    th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class = eval_metrics(valid_probs, valid_labels)
    valid_matrices = th_, rec_, pre_, f1_, spe_, mcc_, auc_

    th_, rec_, pre_, f1_, spe_, mcc_, auc_, pred_class = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, rec_, pre_, f1_, spe_, mcc_, auc_

    print_results(valid_matrices, test_matrices)

    results = {'valid_probs':valid_probs,'valid_labels':valid_labels,'test_probs':test_probs,'test_labels':test_labels}
    with open(opt.sublog_path+'/results.pkl','wb') as f:
        pickle.dump(results,f)

    return


if __name__ == '__main__':
    args = parse_args()
    checkargs(args)
    opt = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sys.stdout = Logger(opt.model_path + '/training.log')
    main(opt,device)
    sys.stdout.log.close()
