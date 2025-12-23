import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from e3nn.o3 import Irreps, spherical_harmonics,FullyConnectedTensorProduct, Linear

# class SimpleVBNet(nn.Module):
#     def __init__(self, in_channels, hidden_channels=64):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.lin = nn.Linear(hidden_channels, 1)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = self.conv1(x, edge_index) # 每个节点被都映射成64维向量
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index) # 每个节点的表示都感知到“更远一级”的信息
#         x = torch.relu(x)

#         x = global_mean_pool(x, batch) # 节点->图，把一张图里所有节点的表示压缩成整体。[number_of_graph, 64]
#         out = self.lin(x) # [batch, 64] -> [batch , 1]

#         return out.view(-1) # [bacth ,1]->batch 


class SimpleEquivariantConv(MessagePassing):
    def __init__(self, irreps_in, irreps_sh, irreps_out):
        super().__init__(aggr="add")
        
        self.tp = FullyConnectedTensorProduct(
            irreps_in,
            irreps_sh,
            irreps_out
        )
    
    def forward(self, x, edge_index, edge_sh):
        return self.propagate(edge_index, x=x, edge_sh=edge_sh)
    
    def message(self, x_j, edge_sh):
        return self.tp(x_j, edge_sh)
    
class E3nnVBnet(nn.Module):
    def __init__(self):
        super().__init__()    

        # ========== irreps定义 ==========
        self.irreps_node_input = Irreps("3x0e")    # Z, 活性e, 非活性e
        self.irreps_edge = Irreps("1o")            # 方向信息-相对坐标
        self.irreps_hidden = Irreps("16x0e")       # 隐藏标量

        self.embed=Linear("3x0e", self.irreps_node_input)

        # ========== 等变卷积 ==========
        self.conv = SimpleEquivariantConv(
            irreps_in=self.irreps_node_input,
            irreps_sh=self.irreps_edge,
            irreps_out=self.irreps_hidden
        )

        # ========== readout ==========
        self.lin=nn.Linear(self.irreps_hidden.dim,1)

    def forward(self, data):
        x = self.embed(data.x)   # [N, 3]
        edge_index = data.edge_index
        r_ij = data.edge_attr[:, 2:5]   # 相对坐标
        batch = data.batch

        edge_sh=spherical_harmonics(
            self.irreps_edge,
            r_ij,
            normalize=True,
            normalization="component"
        )

        x=self.conv(x,edge_index,edge_sh)

        x=global_mean_pool(x,batch)

        out=self.lin(x)

        return out.view(-1)

