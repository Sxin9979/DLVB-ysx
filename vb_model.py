import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, MessagePassing
from e3nn.o3 import Irreps, spherical_harmonics,FullyConnectedTensorProduct, Linear

class SimpleEquivariantConv(MessagePassing):
    def __init__(self, irreps_in, irreps_sh, irreps_out):
        super().__init__(aggr="add")
        
        self.tp = FullyConnectedTensorProduct(
            irreps_in,
            irreps_sh,
            irreps_out
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(3, 16), # 2 + r^2
            nn.SiLU(),
            nn.Linear(16, irreps_out.dim)
        )
    
    def forward(self, x, edge_index, edge_sh, edge_scalar):
        return self.propagate(edge_index, x=x, edge_sh=edge_sh, edge_scalar=edge_scalar)
    
    def message(self, x_j, edge_sh, edge_scalar):
        gate = self.edge_mlp(edge_scalar)
        msg = gate * self.tp(x_j, edge_sh)
        return msg
    
class E3nnVBnet(nn.Module):
    def __init__(
        self, 
        hidden_irreps="16x1o",
        layers = 4,
        residual = True, 
        ):
        super().__init__()    

        self.residual = residual

        # ========== irreps定义 ==========
        self.irreps_node_input = Irreps("3x0e")    # Z, 活性e, 非活性e
        self.irreps_edge = Irreps("1o")            # 方向信息-相对坐标
        self.irreps_hidden = Irreps(hidden_irreps)       # 隐藏标量

        self.embed=Linear(self.irreps_node_input, self.irreps_hidden) 

        # ========== 等变卷积 ==========
        self.convs = nn.ModuleList([
            SimpleEquivariantConv(
                irreps_in=self.irreps_hidden,
                irreps_sh=self.irreps_edge,
                irreps_out=self.irreps_hidden
            )
            for _ in range(layers)
        ] )

        # ========== readout ==========
        self.lin=nn.Linear(self.irreps_hidden.dim, 1)

    def forward(self, data):
        x = self.embed(data.x)   # [N, 3]
        edge_index = data.edge_index
        r_ij = data.edge_attr[:, 2:5]   # 相对坐标
        r_2 = (r_ij**2).sum(dim=-1, keepdim=True)
        edge_scalar = torch.cat(
            [data.edge_attr[:,0:2],r_2], dim=-1
        )
        batch = data.batch

        edge_sh=spherical_harmonics(
            self.irreps_edge,
            r_ij,
            normalize=True,
            normalization="component"
        )

        for conv in self.convs:
            dx = conv(x,edge_index,edge_sh,edge_scalar)
            if self.residual: 
                x = x + dx
            else:
                x = dx

        x = global_mean_pool(x,batch)
        out = self.lin(x)
        return out.view(-1)

