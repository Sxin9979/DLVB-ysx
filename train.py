import torch
from torch_geometric.loader import DataLoader
from pyg_VB_dataset import VBGraphDataset
from vb_model import E3nnVBnet

def train():
    dataset = VBGraphDataset(
        root='/pool1/home/ysxin/E3nnVB/VB',
        task='regression'
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # model = SimpleVBNet(in_channels=dataset[0].x.shape[1]) # 读取每个节点的特征数
    model = E3nnVBnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print(dataset[0].molecule_id, dataset[0].vb_index)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:03d}, Loss {total_loss / len(loader):.6f}")

if __name__ == "__main__":
    train()