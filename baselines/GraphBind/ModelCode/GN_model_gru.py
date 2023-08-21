import torch
from torch import nn
# from torch_scatter import *
from torch_cluster import knn_graph,radius_graph
from torch_geometric.nn import MetaLayer,voxel_grid,global_max_pool,max_pool,avg_pool
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, scatter_
from torch_scatter import scatter_add,scatter_mean,scatter_max
from math import pi
from torch_geometric.data import Batch
import numpy as np


class EdgeModel(torch.nn.Module):
    def __init__(self,x_ind,edge_ind,u_ind,hs,dropratio,bias=True):
        super(EdgeModel, self).__init__()
        self.mlp = nn.Sequential(
                              nn.Conv1d(x_ind*2+edge_ind+u_ind, hs,kernel_size=1,bias=bias),
                              nn.BatchNorm1d(hs),
                              nn.ReLU(inplace=True),
                              nn.Dropout(dropratio),
                              nn.Conv1d(hs,hs,kernel_size=1,bias=bias),
                              nn.BatchNorm1d(hs))

    def forward(self, src, dst, edge_attr,u,batch):
        out = [src,dst,edge_attr]
        if u is not None:
            out.append(u[batch])
        out = torch.cat(out,dim=-1)
        out = out.permute(1,0).unsqueeze(0)
        out = self.mlp(out).squeeze().permute(1,0)
        return out

class NodeModel(torch.nn.Module):
    def __init__(self,x_ind,edge_ind,u_ind,hs,dropratio,aggr,bias=True,edge_attr_num=1):
        super(NodeModel, self).__init__()
        self.aggr = aggr
        if edge_ind != 0:
            self.mlp1 = nn.Sequential(
                                  nn.Conv1d(edge_attr_num*edge_ind*len(aggr), hs*len(aggr),kernel_size=1,bias=bias),
                                  nn.BatchNorm1d(hs*len(aggr)),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropratio),
                                  nn.Conv1d(hs*len(aggr),hs*len(aggr),kernel_size=1,bias=bias),
                                  nn.BatchNorm1d(hs*len(aggr)))

        self.mlp2 = nn.Sequential(
            nn.Conv1d(x_ind + edge_ind*len(aggr) + u_ind, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Conv1d(hs, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs))

    def forward(self, x, edge_index, edge_attr,u, batch):
        out = [x]
        if u is not None:
            out.append(u[batch])

        if isinstance(edge_attr, torch.Tensor):
            row, col = edge_index
            if edge_attr is not None:
                if 'max' in self.aggr:
                    out.append(scatter_max(edge_attr, col, dim=0, dim_size=x.size(0))[0])
                if 'mean' in self.aggr:
                    out.append(scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0)))
                if 'add' in self.aggr:
                    out.append(scatter_add(edge_attr, col, dim=0, dim_size=x.size(0)))
            else:
                if 'max' in self.aggr:
                    out.append(scatter_max(x[row], col, dim=0, dim_size=x.size(0))[0])
                if 'mean' in self.aggr:
                    out.append(scatter_mean(x[row], col, dim=0, dim_size=x.size(0)))
                if 'add' in self.aggr:
                    out.append(scatter_add(x[row], col, dim=0, dim_size=x.size(0)))
        elif isinstance(edge_attr, list):
            out_edge = []
            for i in range(len(edge_index)):
                edge_index_i = edge_index[i]
                row, col = edge_index_i
                if edge_attr is not None:
                    edge_attr_i = edge_attr[i]
                    if 'max' in self.aggr:
                        out_edge.append(scatter_max(edge_attr_i, col, dim=0, dim_size=x.size(0))[0])
                    if 'mean' in self.aggr:
                        out_edge.append(scatter_mean(edge_attr_i, col, dim=0, dim_size=x.size(0)))
                    if 'add' in self.aggr:
                        out_edge.append(scatter_add(edge_attr_i, col, dim=0, dim_size=x.size(0)))
                else:
                    if 'max' in self.aggr:
                        out_edge.append(scatter_max(x[row], col, dim=0, dim_size=x.size(0))[0])
                    if 'mean' in self.aggr:
                        out_edge.append(scatter_mean(x[row], col, dim=0, dim_size=x.size(0)))
                    if 'add' in self.aggr:
                        out_edge.append(scatter_add(x[row], col, dim=0, dim_size=x.size(0)))

            out_edge = torch.cat(out_edge,dim=1).unsqueeze(-1)
            out_edge = self.mlp1(out_edge).squeeze(-1)
            out.append(out_edge.squeeze(-1))

        out = torch.cat(out,dim=1).permute(1,0).unsqueeze(0)
        out = self.mlp2(out).squeeze().permute(1,0)
        return out

class GlobalModel(torch.nn.Module):
    def __init__(self,x_ind,u_ind,hs,dropratio,aggr,bias=True,poolratio=1):
        super(GlobalModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(x_ind*len(aggr)*poolratio + u_ind, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Conv1d(hs, hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hs))

        self.hs = hs
        self.aggr = aggr
        self.poolratio = poolratio

    def forward(self, x,u, batch,polar_pos=None):
        batchsize = max(batch)+1
        out = []
        if u is not None:
            out.append(u)

        if polar_pos is None:
            if 'max' in self.aggr:
                out.append(scatter_max(x, batch, dim=0)[0])
            if 'mean' in self.aggr:
                out.append(scatter_mean(x, batch, dim=0))
            if 'add' in self.aggr:
                out.append(scatter_add(x, batch, dim=0))
        else:
            if 'max' in self.aggr:
                out.append(ring_pool(x, polar_pos, batch, self.poolratio, 'max').reshape(batchsize,-1))
            if 'mean' in self.aggr:
                out.append(ring_pool(x, polar_pos, batch, self.poolratio, 'mean').reshape(batchsize,-1))
            if 'add' in self.aggr:
                out.append(ring_pool(x, polar_pos, batch, self.poolratio, 'add').reshape(batchsize,-1))

        out = torch.cat(out, dim=1).unsqueeze(-1)
        out = self.mlp(out).squeeze(-1) # batch * hs
        return out

def ring_pool(x, pos, batch, ratio, method='max'):
    nbatch = batch.max().item() + 1
    cluster = voxel_grid(pos, batch, size=[1 / ratio, 2 * pi, 2 * pi], start=[0.0001, 0, -pi / 2],
                         end=[0.999, pi, pi / 2])
    out = []
    for b in range(nbatch):
        b_cluster = cluster[batch == b]
        b_x = x[batch == b]
        b_cluster -= ratio * b
        b_out = scatter_(method, b_x, b_cluster, dim_size=ratio)
        # i = b_out.cpu().detach().numpy()

        out.append(b_out.unsqueeze(0))
    out = torch.cat(out, dim=0)
    # out_ = out.cpu().detach().numpy()

    return out

class MetaEncoder(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MetaEncoder, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""
        if edge_attr is not None:
            if isinstance(edge_attr, torch.Tensor):
                # row, col = edge_index
                edge_attr = self.edge_model(x[edge_index[0]], x[edge_index[1]], edge_attr, u, batch[edge_index[0]])
            elif isinstance(edge_attr, list):
                edge_attr2 = []
                for edge_index_i, edge_attr_i in zip(edge_index, edge_attr):
                    # row, col = edge_index_i
                    edge_attr_i = self.edge_model(x[edge_index_i[0]], x[edge_index_i[1]], edge_attr_i, u, batch[edge_index_i[0]])
                    edge_attr2.append(edge_attr_i)
                edge_attr = edge_attr2
            else:
                raise TypeError

        x = self.node_model(x, edge_index, edge_attr, u, batch)
        u = self.global_model(x, u, batch,polar_pos)

        return x, edge_attr, u

class MetaGRU(torch.nn.Module):
    def __init__(self, gru_steps,e_hs,x_hs,u_hs,bias=True,edge_model=None, node_model=None, global_model=None):
        super(MetaGRU, self).__init__()
        self.gru_steps = gru_steps

        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        if edge_model is not None:
            self.edge_rnn = torch.nn.GRUCell(e_hs,e_hs,bias=bias)
        self.node_rnn = torch.nn.GRUCell(x_hs,x_hs,bias=bias)
        self.global_rnn = torch.nn.GRUCell(u_hs,u_hs,bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""
        global_out = []
        for i in range(self.gru_steps):
            if edge_attr is not None:
                if isinstance(edge_attr, torch.Tensor):
                    # row, col = edge_index
                    edge_attr_out = self.edge_model(x[edge_index[0]], x[edge_index[1]], edge_attr, u, batch[edge_index[0]])
                    edge_attr = self.edge_rnn(edge_attr_out,edge_attr)
                elif isinstance(edge_attr, list):
                    edge_attr2 = []
                    for edge_index_i, edge_attr_i in zip(edge_index, edge_attr):
                        # row, col = edge_index_i
                        edge_attr_out = self.edge_model(x[edge_index_i[0]], x[edge_index_i[1]], edge_attr_i, u, batch[edge_index_i[0]])
                        edge_attr_i = self.edge_rnn(edge_attr_out, edge_attr_i)
                        edge_attr2.append(edge_attr_i)
                    edge_attr = edge_attr2

            x_out = self.node_model(x, edge_index, edge_attr, u, batch)
            x = self.node_rnn(x_out,x)

            u_out = self.global_model(x, u, batch,polar_pos)
            u = self.global_rnn(u_out,u)
            global_out.append(u.unsqueeze(1))

        global_out = torch.cat(global_out,dim=1)
        return global_out

class MetaGRUNoShareW(torch.nn.Module):
    def __init__(self, gru_steps,e_hs,x_hs,u_hs,bias=True,edge_model=None, node_model=None, global_model=None):
        super(MetaGRUNoShareW, self).__init__()
        self.gru_steps = gru_steps

        self.edge_model = nn.ModuleList([edge_model] * gru_steps)
        self.node_model = nn.ModuleList([node_model] * gru_steps)
        self.global_model = nn.ModuleList([global_model] * gru_steps)

        if edge_model is not None:
            self.edge_rnn = torch.nn.GRUCell(e_hs,e_hs,bias=bias)
        self.node_rnn = torch.nn.GRUCell(x_hs,x_hs,bias=bias)
        self.global_rnn = torch.nn.GRUCell(u_hs,u_hs,bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""
        global_out = []
        for i in range(self.gru_steps):
            if edge_attr is not None:
                if isinstance(edge_attr, torch.Tensor):
                    # row, col = edge_index
                    edge_attr_out = self.edge_model[i](x[edge_index[0]], x[edge_index[1]], edge_attr, u, batch[edge_index[0]])
                    edge_attr = self.edge_rnn(edge_attr_out,edge_attr)
                elif isinstance(edge_attr, list):
                    edge_attr2 = []
                    for edge_index_i, edge_attr_i in zip(edge_index, edge_attr):
                        # row, col = edge_index_i
                        edge_attr_out = self.edge_model[i](x[edge_index_i[0]], x[edge_index_i[1]], edge_attr_i, u, batch[edge_index_i[0]])
                        edge_attr_i = self.edge_rnn(edge_attr_out, edge_attr_i)
                        edge_attr2.append(edge_attr_i)
                    edge_attr = edge_attr2

            x_out = self.node_model[i](x, edge_index, edge_attr, u, batch)
            x = self.node_rnn(x_out,x)

            u_out = self.global_model[i](x, u, batch,polar_pos)
            u = self.global_rnn(u_out,u)
            global_out.append(u.unsqueeze(1))

        global_out = torch.cat(global_out,dim=1)
        return global_out

class MetaComposed(torch.nn.Module):
    def __init__(self, gru_steps,edge_model=None, node_model=None, global_model=None):
        super(MetaComposed, self).__init__()
        self.gru_steps = gru_steps

        self.edge_model = nn.ModuleList([edge_model]*gru_steps)
        self.node_model = nn.ModuleList([node_model]*gru_steps)
        self.global_model = nn.ModuleList([global_model]*gru_steps)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""
        global_out = []
        for i in range(self.gru_steps):
            if isinstance(edge_attr, torch.Tensor):
                # row, col = edge_index
                edge_attr = self.edge_model[i](x[edge_index[0]], x[edge_index[1]], edge_attr, u, batch[edge_index[0]])
            elif isinstance(edge_attr, list):
                edge_attr2 = []
                for edge_index_i, edge_attr_i in zip(edge_index, edge_attr):
                    # row, col = edge_index_i
                    edge_attr_out = self.edge_model[i](x[edge_index_i[0]], x[edge_index_i[1]], edge_attr_i, u, batch[edge_index_i[0]])
                    edge_attr2.append(edge_attr_out)
                edge_attr = edge_attr2

            x = self.node_model[i](x, edge_index, edge_attr, u, batch)
            u = self.global_model[i](x, u, batch,polar_pos)
            global_out.append(u.unsqueeze(1))

        global_out = torch.cat(global_out,dim=1)
        return global_out


class MetaMLP(torch.nn.Module):
    def __init__(self, gru_steps,hs,bias=True,edge_model=None, node_model=None, global_model=None):
        super(MetaMLP, self).__init__()
        self.gru_steps = gru_steps

        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.edge_rnn = torch.nn.GRUCell(hs,hs,bias=bias)
        self.node_rnn = torch.nn.GRUCell(hs,hs,bias=bias)
        self.global_rnn = torch.nn.GRUCell(hs,hs,bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None,polar_pos=None):
        """"""

        global_out = []
        for i in range(self.gru_steps):
            if isinstance(edge_attr, torch.Tensor):
                row, col = edge_index
                edge_attr2 = self.edge_model(x[row], x[col], edge_attr, u, batch[row])
                # edge_attr2 = self.edge_rnn(edge_attr_out,edge_attr)
            elif isinstance(edge_attr, list):
                edge_attr2 = []
                for edge_index_i, edge_attr_i in zip(edge_index, edge_attr):
                    row, col = edge_index_i
                    edge_attr_out = self.edge_model(x[row], x[col], edge_attr_i, u, batch[row])
                    # edge_attr_i = self.edge_rnn(edge_attr_out, edge_attr_i)
                    edge_attr2.append(edge_attr_out)
            else:
                raise TypeError

            x = self.node_model(x, edge_index, edge_attr2, u, batch)
            # x = self.node_rnn(x_out,x)

            u = self.global_model(x, u, batch,polar_pos)
            # u = self.global_rnn(u_out,u)
            global_out.append(u.unsqueeze(1))

        global_out = torch.cat(global_out,dim=1)
        return global_out

class MetaBind_MultiEdges(torch.nn.Module):
    def __init__(self,gru_steps,x_ind,edge_ind,x_hs,e_hs,u_hs,dropratio,bias,edge_method,r_list,edge_aggr,node_aggr,dist,max_nn,
                 stack_method,apply_edgeattr,apply_nodeposemb):
        super(MetaBind_MultiEdges, self).__init__()
        self.dist = dist
        self.max_nn = max_nn
        self.u_ind = 0
        self.edge_method = edge_method
        if apply_nodeposemb is False:
            x_ind -= 1
        if apply_edgeattr:
            self.bn = nn.ModuleList([nn.BatchNorm1d(x_ind),
                                     nn.BatchNorm1d(2)])
            self.encoder = MetaEncoder(edge_model=EdgeModel(x_ind=x_ind,edge_ind=edge_ind,u_ind=self.u_ind,hs=e_hs,dropratio=dropratio,bias=bias),
                                       node_model=NodeModel(x_ind=x_ind,edge_ind=e_hs,u_ind=self.u_ind,hs=x_hs,dropratio=dropratio,bias=bias,aggr=edge_aggr,edge_attr_num=len(r_list)),
                                       global_model=GlobalModel(x_ind=x_hs,u_ind=self.u_ind,hs=u_hs,dropratio=dropratio,bias=bias,aggr=node_aggr))
            if stack_method == 'GRU':
                self.meta_GN_gru = MetaGRU(gru_steps=gru_steps,e_hs=e_hs,x_hs=x_hs,u_hs=u_hs,bias=True,
                                      edge_model=EdgeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=e_hs,dropratio=dropratio,bias=bias),
                                      node_model=NodeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=x_hs,dropratio=dropratio,bias=bias,aggr=edge_aggr,edge_attr_num=len(r_list)),
                                      global_model=GlobalModel(x_ind=x_hs,u_ind=u_hs,hs=u_hs,dropratio=dropratio,bias=bias,aggr=node_aggr))
                self.stacked_GN = self.meta_GN_gru
            elif stack_method == 'Composed':
                self.meta_GN_Composed = MetaComposed(gru_steps=gru_steps,
                                      edge_model=EdgeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=e_hs,dropratio=dropratio,bias=bias),
                                      node_model=NodeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=x_hs,dropratio=dropratio,bias=bias,aggr=edge_aggr,edge_attr_num=len(r_list)),
                                      global_model=GlobalModel(x_ind=x_hs,u_ind=u_hs,hs=u_hs,dropratio=dropratio,bias=bias,aggr=node_aggr))
                self.stacked_GN = self.meta_GN_Composed
            elif stack_method == 'GRUNoShareW':
                self.meta_GN_gru = MetaGRUNoShareW(gru_steps=gru_steps,e_hs=e_hs,x_hs=x_hs,u_hs=u_hs,bias=True,
                                      edge_model=EdgeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=e_hs,dropratio=dropratio,bias=bias),
                                      node_model=NodeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=x_hs,dropratio=dropratio,bias=bias,aggr=edge_aggr,edge_attr_num=len(r_list)),
                                      global_model=GlobalModel(x_ind=x_hs,u_ind=u_hs,hs=u_hs,dropratio=dropratio,bias=bias,aggr=node_aggr))
                self.stacked_GN = self.meta_GN_gru
            else:
                raise ValueError
        else:
            e_hs = 0
            self.bn = nn.ModuleList([nn.BatchNorm1d(x_ind)])
            self.encoder = MetaEncoder(
                node_model=NodeModel(x_ind=x_ind, edge_ind=e_hs, u_ind=self.u_ind, hs=x_hs, dropratio=dropratio,bias=bias, aggr=edge_aggr, edge_attr_num=len(r_list)),
                global_model=GlobalModel(x_ind=x_hs, u_ind=self.u_ind, hs=u_hs, dropratio=dropratio, bias=bias,aggr=node_aggr))
            if stack_method == 'GRU':
                self.meta_GN_gru = MetaGRU(gru_steps=gru_steps, e_hs=e_hs, x_hs=x_hs, u_hs=u_hs, bias=True,
                                           node_model=NodeModel(x_ind=x_hs, edge_ind=e_hs, u_ind=u_hs, hs=x_hs,dropratio=dropratio, bias=bias, aggr=edge_aggr,edge_attr_num=len(r_list)),
                                           global_model=GlobalModel(x_ind=x_hs, u_ind=u_hs, hs=u_hs,dropratio=dropratio, bias=bias, aggr=node_aggr))
                self.stacked_GN = self.meta_GN_gru
            elif stack_method == 'Composed':
                self.meta_GN_Composed = MetaComposed(gru_steps=gru_steps,
                                                     node_model=NodeModel(x_ind=x_hs, edge_ind=e_hs, u_ind=u_hs,hs=x_hs, dropratio=dropratio, bias=bias, aggr=edge_aggr, edge_attr_num=len(r_list)),
                                                     global_model=GlobalModel(x_ind=x_hs, u_ind=u_hs, hs=u_hs, dropratio=dropratio, bias=bias, aggr=node_aggr))
                self.stacked_GN = self.meta_GN_Composed
            elif stack_method == 'GRUNoShareW':
                self.meta_GN_gru = MetaGRUNoShareW(gru_steps=gru_steps,e_hs=e_hs,x_hs=x_hs,u_hs=u_hs,bias=True,
                                      edge_model=EdgeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=e_hs,dropratio=dropratio,bias=bias),
                                      node_model=NodeModel(x_ind=x_hs,edge_ind=e_hs,u_ind=u_hs,hs=x_hs,dropratio=dropratio,bias=bias,aggr=edge_aggr,edge_attr_num=len(r_list)),
                                      global_model=GlobalModel(x_ind=x_hs,u_ind=u_hs,hs=u_hs,dropratio=dropratio,bias=bias,aggr=node_aggr))
                self.stacked_GN = self.meta_GN_gru
            else:
                raise ValueError

        self.clf = nn.Sequential(
            nn.Linear(u_hs*gru_steps, u_hs//2),
            nn.BatchNorm1d(u_hs//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropratio),
            nn.Linear(u_hs//2, 2))

        self.pdist = nn.PairwiseDistance(p=2,keepdim=True)
        self.cossim = nn.CosineSimilarity(dim=1)

        self.r_list = r_list
        self.apply_edgeattr = apply_edgeattr
        self.apply_nodeposemb = apply_nodeposemb

    def forward(self, data):
        """"""
        x,pos, batch = data.x,data.pos, data.batch

        # batchsize = max(batch)+1
        # polar_pos = self.Cartesian2Polar(pos)

        if len(self.r_list)>1:
            radius_index_list = []
            for r in self.r_list:
                pos1 = pos.cpu()
                batch1 = batch.cpu()
                radius_index_list_i = radius_graph(pos1, r=r, batch=batch1, loop=True, max_num_neighbors=self.max_nn)
                radius_index_list_i = radius_index_list_i.to(x.device)
                radius_index_list.append(radius_index_list_i)
            # radius_index_list = self.cal_difference_index(radius_index_list)
            if self.apply_edgeattr is True:
                radius_attr_list = self.cal_edge_attr(radius_index_list,pos)
            else:
                radius_attr_list = None
        else:
            pos1 = pos.cpu()
            batch1 = batch.cpu()
            radius_index_list = radius_graph(pos1, r=self.r_list[0], batch=batch1, loop=True,max_num_neighbors=self.max_nn)
            radius_index_list = radius_index_list.to(x.device)
            if self.apply_edgeattr is True:
                radius_attr_list = self.cal_edge_attr(radius_index_list, pos)
            else:
                radius_attr_list = None



        if self.apply_nodeposemb is True:
            x = torch.cat([x,torch.sqrt(torch.sum(pos*pos,dim=1)).unsqueeze(-1)/self.dist],dim=-1)
        # x = x.half()
        x = self.bn[0](x.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)

        x, radius_attr_list, u = self.encoder(x=x, edge_index=radius_index_list, edge_attr=radius_attr_list, u=None, batch=batch)
        global_output = self.stacked_GN(x=x, edge_index=radius_index_list, edge_attr=radius_attr_list, u=u, batch=batch)
        global_output = global_output.reshape(max(batch)+1,-1)
        out = self.clf(global_output)
        out = F.softmax(out, -1)
        return out[:, 1]

    def gaussfunc(self,distance,sd):
        gauss_dis = torch.exp(-distance**2/(2*sd**2))
        return gauss_dis

    def cal_difference_index(self,index_list):
        for i in range(len(self.r_list)-1):
            index1 = set(tuple(zip(index_list[i][0].cpu().numpy(),index_list[i][1].cpu().numpy())))
            index2 = set(tuple(zip(index_list[i+1][0].cpu().numpy(),index_list[i+1][1].cpu().numpy())))
            a = np.array(list(index1.difference(index2)))
            # index_list[i] = torch.from_numpy(a).permute(1,0)
            index_list[i] = torch.tensor(a).permute(1,0)
        return index_list

    def cal_edge_attr(self,index_list,pos):
        if len(self.r_list) == 1:
            # radius_distance = self.gaussfunc(self.pdist(pos[index_list[0]], pos[index_list[1]]), sd=2)

            # radius_distance = self.pdist(pos[index_list[0]], pos[index_list[1]])/self.r_list[0]
            # radius_arccos = (self.cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2
            radius_attr_list = torch.cat([self.pdist(pos[index_list[0]], pos[index_list[1]])/self.r_list[0],
                                          (self.cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2], dim=1)
            radius_attr_list = self.bn[1](radius_attr_list.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        else:
            radius_attr_list = []
            for i,radius_index in enumerate(index_list):
                # radius_distance = self.gaussfunc(self.pdist(pos[radius_index[0]], pos[radius_index[1]]), sd=2)
                # radius_distance = self.pdist(pos[radius_index[0]], pos[radius_index[1]])/self.r_list[i]
                # radius_arccos = (self.cossim(pos[radius_index[0]], pos[radius_index[1]]).unsqueeze(-1) + 1) / 2
                radius_attr = torch.cat([self.pdist(pos[radius_index[0]], pos[radius_index[1]])/self.r_list[i],
                                         (self.cossim(pos[radius_index[0]], pos[radius_index[1]]).unsqueeze(-1) + 1) / 2], dim=1)
                radius_attr = self.bn[1](radius_attr.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
                radius_attr_list.append(radius_attr)

        return radius_attr_list

    def Cartesian2LogPolar(self,src):
        inf = 1e-6
        src[src==0] = inf
        r = torch.sqrt(torch.sum(src*src,dim=1))
        log_r = torch.log10(r+1).unsqueeze(-1)  # or log10(r+1)
        log_r[log_r>1] = 1
        # a = (src[:,2]/r).cpu().numpy()
        # b = (src[:,1]/src[:,0]).cpu().numpy()
        theta = torch.acos(src[:,2]/r).unsqueeze(-1)  # [0,pi]
        phi = torch.atan(src[:,1]/src[:,0]).unsqueeze(-1)  # (-pi/2,pi/2)
        dst = torch.cat([log_r,theta,phi],dim=-1)

        # r_ = r.cpu().numpy()
        # logr_ = log_r.cpu().numpy()
        # the_ = theta.cpu().numpy()
        # phi_ = phi.cpu().numpy()

        return dst

    def Cartesian2Polar(self,src):
        inf = 1e-6
        src[src==0] = inf
        r = torch.sqrt(torch.sum(src*src,dim=1))
        r = r/10
        r[r>1]=1
        theta = torch.acos(src[:,2]/r).unsqueeze(-1)  # [0,pi]
        phi = torch.atan(src[:,1]/src[:,0]).unsqueeze(-1)  # (-pi/2,pi/2)
        dst = torch.cat([r.unsqueeze(-1),theta,phi],dim=-1)

        return dst
