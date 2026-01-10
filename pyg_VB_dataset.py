import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.serialization import safe_globals
from tqdm import tqdm
from VB_types import VBinformation
from torch_geometric.data.data import DataEdgeAttr
import os.path as osp
from LapPE import lap_pe

class VBGraphDataset(InMemoryDataset):
    def __init__(self, cfg, root, task,
                 transform=None, pre_transform=None, pre_filter=None):
        self.folder = osp.join(root, task)
        self.cfg=cfg
        self.lap_pe_k = cfg["model"]["lap_pe_k"]
        super().__init__(root, transform, pre_transform, pre_filter)

        # 加载 processed 数据
        processed_file = osp.join(self.processed_dir, self.processed_file_names[0])
        if osp.exists(processed_file):
            # safe_globals 告诉 PyTorch 这些自定义类是安全的
            with safe_globals([VBinformation, DataEdgeAttr]):
                self.data, self.slices = torch.load(processed_file, map_location='cpu', weights_only=False)
            print(f"Loaded processed data from: {processed_file}")
        else:
            print("Processed data not found, please run process() first.")

    @property
    def raw_file_names(self):
        return ['vb_data_collection.pt']

    @property
    def processed_file_names(self):
        return ['train_data.pt']

    @staticmethod
    def load_vb_data(file_path):
        vbdata = torch.load(file_path, map_location='cpu', weights_only=False)
        return vbdata

    def process(self):
        """transfer to Graph """
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        vbdata = self.load_vb_data(raw_path)

        data_list = []

        for mol in tqdm(vbdata['molecules'], desc="Processing molecules"):
            nodes = int(mol.nodes)
            num_str = len(mol.str)
            weights=torch.tensor([float(w) for w in mol.LowdinWeights], dtype=torch.float)
            weight_max = torch.max(weights)
            weights = weights / weight_max # Relative weight
            pos = mol.coor.clone().detach().float()
            atom_nums = mol.atom_nums.clone().detach().long()
            edge_cursor = 0

            for i in range(num_str):
                # 1. 节点特征
                x = mol.X[:, 3*i:3*(i+1)].clone().detach().float()
                # 2. 当前VB结构的边数
                Amat_i = mol.A_mat[i]
                num_edge_i = int(Amat_i.sum().item())
                # 3. 分割edge_index
                source = mol.A_list[0][edge_cursor : edge_cursor + num_edge_i]
                target = mol.A_list[1][edge_cursor : edge_cursor + num_edge_i]
                edge_index = torch.tensor([source, target], dtype=torch.long) - 1
                # 4. 分割edge_attr
                edge_attr = torch.tensor(mol.E[edge_cursor : edge_cursor + num_edge_i], dtype=torch.float)
                edge_cursor += num_edge_i
                # 5. 标签 
                y = weights[i].view(1)
                # 6. 构造Data
                k = self.lap_pe_k
                evals_k, evecs_k = lap_pe(Amat_i.float(), nodes, k, eps=1e-12, add_self_loops=False )

                data = Data(
                    x=x, # [node,3]
                    edge_index=edge_index, # [2,num_edge_i]
                    edge_attr=edge_attr, # [num_edge_i, 5]
                    pos=pos, # [nodes, 3] 
                    y=y
                )

                data.lap_evals = evals_k
                data.lap_evecs = evecs_k
                data.molecule_id = mol.molecule_id
                data.vb_index = i
                data_list.append(data)

        # Collate & save
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print("Saved processed data to:", self.processed_paths[0])
        print("Number of graphs:", len(data_list))


if __name__ == '__main__':
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).with_name("config.yaml")
    with open (cfg_path,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = VBGraphDataset(
        cfg=cfg,
        root=cfg["dataset"]["root"],
        task=cfg["dataset"]["task"]
    )
    print(dataset)  # Dataset 对象信息
    print("Number of graphs:", len(dataset))  # 图数量

    d = dataset[0]
    print(d)
    print("x:", d.x.shape)
    print("edge_index:", d.edge_index.shape)
    print("edge_attr:", d.edge_attr.shape)
    print("pos:", d.pos.shape)
    print("y:", d.y)
