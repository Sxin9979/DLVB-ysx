import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from e3nn.o3 import Irreps, spherical_harmonics,FullyConnectedTensorProduct, Linear
from e3nn.nn import BatchNorm
from LapPE import SignNet

class EquivariantConv(MessagePassing):
    """
    Equivariant message passing block:
    Δx_i = Σ_j gate_ij * TP(x_j, sh_ij)
    - TP provides O(3)-equivariant geometric mixing using spherical harmonics.
    - gate_ij(0e) is produced from scalar edge features by MLP.
    """
    def __init__(self, irreps_hidden: Irreps, irreps_sh: Irreps, edge_mlp_in: int, edge_mlp_hidden: int):
        super().__init__(aggr="add")
        self.irreps_hidden=Irreps(irreps_hidden)
        self.irreps_sh=Irreps(irreps_sh)
        
        self.tp = FullyConnectedTensorProduct(
            irreps_hidden,
            irreps_sh,
            irreps_hidden
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, edge_mlp_hidden),
            nn.SiLU(),
            nn.Linear(edge_mlp_hidden, 1)
        )
    
    def forward(self, x, edge_index, edge_sh, edge_scalar):
        return self.propagate(edge_index, x=x, edge_sh=edge_sh, edge_scalar=edge_scalar)
    
    def message(self, x_j, edge_sh, edge_scalar):
        gate = self.edge_mlp(edge_scalar)
        msg = gate * self.tp(x_j, edge_sh)
        return msg
    
class E3nnVBnet(nn.Module):
    """
    Expects:
    data.x: [N,irreps_in.dim] (node scalars)
    data.edge_attr:[E,K] last 3 columns are r_ij (relative vectors)
    data.edge_index, data.batch

    - build edge_sh via spherical harmonics up to lmax_attr
    - build node_attr by aggregating edge_sh onto nodes
    - embedding: TP(x, node_attr) -> hidden_irreps  (allows hidden irreps like 0e+1o+2e)
    - multi-layer equivariant conv: hidden->hidden, with residual and optional norm
    - readout: Linear(hidden->out_irreps) then pool to graph
    """
    def __init__(self, cfg:dict):
        super().__init__()    
        mcfg = cfg.get("model",{})

        # ========== irreps定义 ==========
        self.irreps_in = Irreps(mcfg.get("irreps_in"))  # Z, 活性e, 非活性e
        self.hidden_irreps = Irreps(mcfg.get("hidden_irreps"))
        self.irreps_out = Irreps(mcfg.get("irreps_out"))

        self.lmax_attr = mcfg.get("lmax_attr")
        self.irreps_sh = Irreps.spherical_harmonics(self.lmax_attr)  # e.g., lmax=2 -> 0e+1o+2e
        self.use_r2 = mcfg.get("use_r2")
        self.edge_scalar_dim = mcfg.get("edge_scalar_dim")

        self.num_layers = mcfg.get("layers")
        self.norm_type = mcfg.get("norm", "none").lower()
        self.pool = mcfg.get("pool", "mean").lower()
        self.residual = True
        edge_mlp_hidden = mcfg.get("edge_mlp_hidden")
        
        self.embedding_tp = FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.hidden_irreps)

        self.signnet = SignNet(
            k=mcfg.get("lap_pe_k"),
            phi_hidden=mcfg.get("signnet").get("phi_hidden"),
            phi_out=mcfg.get("signnet").get("phi_out"),
            phi_layers=mcfg.get("signnet").get("phi_layers"),
            rho_hidden=mcfg.get("signnet").get("rho_hidden"),
            out_dim=mcfg.get("signnet").get("out_dim"),
            rho_layers=mcfg.get("signnet").get("rho_layers")            
        )
        self.signnet_out_dim=mcfg.get("signnet").get("out_dim")
        
        # ========== convs ==========
        self.convs = nn.ModuleList([
            EquivariantConv(
                irreps_hidden = self.hidden_irreps,
                irreps_sh=self.irreps_sh,
                edge_mlp_in = self.edge_scalar_dim,
                edge_mlp_hidden = edge_mlp_hidden
            )
            for _ in range(self.num_layers)
        ] )

        # ========== norms ============
        if self.norm_type == "batch":
            self.norms = nn.ModuleList([BatchNorm(self.hidden_irreps) for _ in range (self.num_layers)])
        else: 
            self.norms = nn.ModuleList([nn.Identity() for _ in range(self.num_layers)])

        # ========== readout: hidden -> out irreps ==========
        self.to_out = Linear(self.hidden_irreps, self.irreps_out)

    def pool_nodes(self, x, batch, mode = "mean") :
        mode = (mode or "mean").lower()
        if mode in ["sum", "add"]:
            return scatter(x, batch, dim=0, reduce="sum")
        else:
            return scatter(x, batch, dim=0, reduce="mean")

    def forward(self, data):
        edge_index = data.edge_index
        batch = data.batch

        r_ij = data.edge_attr[:, -3:]
        edge_sh = spherical_harmonics(
            self.irreps_sh,
            r_ij,
            normalize=True,
            normalization="component"
        )

        node_attr = scatter(edge_sh, edge_index[1], dim=0, reduce="mean", dim_size=data.num_nodes)
        V = self.signnet(data.lap_evecs, data.lap_evals)
        x_in = torch.cat([data.x,V],dim=-1)

        assert x_in.shape[1] == Irreps(self.irreps_in).dim

        x = self.embedding_tp(x_in, node_attr)

        edge_scalar = data.edge_attr[:,:-3]

        if self.use_r2:
            r = torch.norm(r_ij, dim=-1,keepdim=True) # r = bond lenth
            edge_scalar = torch.cat([edge_scalar, r], dim=-1)        

        # enforce scalar dim to match (edge_scalar_dim==3)
        if edge_scalar.shape[1] > self.edge_scalar_dim:
            edge_scalar = edge_scalar[:, :self.edge_scalar_dim]
        elif edge_scalar.shape[1] < self.edge_scalar_dim:
            # pad zeros if short 
            pad = torch.zeros(edge_scalar.shape[0], self.edge_scalar_dim - edge_scalar.shape[1], device=edge_scalar.device)
            edge_scalar = torch.cat([edge_scalar, pad], dim=-1)

        # --- multi-layer residual + (optional) norm ---
        for conv, norm in zip(self.convs, self.norms):
            dx = conv(x, edge_index, edge_sh, edge_scalar)
            x = x + dx if self.residual else dx
            x = norm(x)
    
        # --- node -> out_irreps then pool to graph ---
        out_node = self.to_out(x)  # [N, out_irreps.dim(1) ]
        out_graph = self.pool_nodes(out_node, batch, self.pool)  # [B, out_irreps.dim]
        return out_graph.view(-1)
    

