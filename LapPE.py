import torch
import torch.nn as nn

def lap_pe( A, num_nodes, k, eps, add_self_loops):
    """
    Compute normalized Laplacian eigenpairs.
    L_sym = I - D^{-1/2} A D^{-1/2}

    Args: 
    A: [N,N] adjacency matrix
    num_nodes: number of nodes N
    k: number of non-trivial eigenvectors
    eps: numerical stability
    add_self_loops: whether to add self-loops to adjacency

    Returns:
    evals_k: FloatTensor [k]
    evecs_k: FloatTensor [N, k]    
    """
    if k<= 0 :
        return (
            torch.zeros((0,),dtype=torch.float32),
            torch.zeros((num_nodes,0),dtype=torch.float32)
        )
    A = A.clone()
    if add_self_loops:
        A.fill_diagonal_(1.0)

    deg = A.sum(dim=1)
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    I = torch.eye(num_nodes)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt

    evals, evecs = torch.linalg.eigh(L) # Eigen-decomposition
    nontrivial = torch.where(evals > eps )[0]
    if nontrivial.numel() == 0:
        return (
            torch.zeros(k,),
            torch.zeros(num_nodes, k),
        )

    take = nontrivial[:k]
    evals_k = evals[take].float()
    evecs_k = evecs[:, take].float()

    if evecs_k.shape[1] < k:
        pad = k - evecs_k.shape[1]
        evecs_k = torch.cat(
            [evecs_k, torch.zeros(num_nodes, pad)], dim=1
        )
        evals_k = torch.cat(
            [evals_k, torch.zeros(pad,)], dim=0
        )
    
    return evals_k, evecs_k

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        assert num_layers >= 1
        layers=[]
        d=in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)       
    
    def forward(self,x):
        return self.net(x)
    
class SignNet(nn.Module):
    """
    Sign-invariant encoder for Laplacian eigenvectors.
    For each eigenvector v_j with eigenvalue λ_j:
    h_j = ϕ([v_j, λ_j]) + ϕ([-v_j, λ_j])

    Output: V[N,out_dim]
    """

    def __init__(self, k, phi_hidden, phi_out, phi_layers, rho_hidden, out_dim, rho_layers):
        super().__init__()
        self.k = k 

        self.phi=MLP(
            in_dim=2,
            hidden_dim=phi_hidden,
            out_dim=phi_out,
            num_layers=phi_layers
        )

        self.rho=MLP(
            in_dim=k*phi_out,
            hidden_dim=rho_hidden,
            out_dim=out_dim,
            num_layers=rho_layers,
        )

    def forward(self, evecs, evals):
        N, _ = evecs.shape
        blocks = []
        for j in range(self.k):
            v = evecs[:,j:j+1] # [N,1]
            lam = evals[j].expand(N,1) # [N,1]

            x_pos = torch.cat([v, lam], dim=1)    # [N,2]
            x_neg = torch.cat([-v, lam], dim=1)   # [N,2]

            h = self.phi(x_pos) + self.phi(x_neg)  # [N, phi_out]
            blocks.append(h)

        H = torch.cat(blocks, dim=1)  # [N, k*phi_out]
        V = self.rho(H)               # [N, out_dim]
        return V              