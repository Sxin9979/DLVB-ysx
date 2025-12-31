import random
import torch
import yaml
import numpy as np
from torch_geometric.loader import DataLoader
from pyg_VB_dataset import VBGraphDataset
from vb_model import E3nnVBnet

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed",0)
    set_seed(seed)

    dataset = VBGraphDataset(
        root=cfg["dataset"]["root"],
        task=cfg["dataset"]["task"]
    )

    # device
    device_str = cfg.get("training", {}).get("device", "cpu")
    if device_str.startswith("cuda") and (not torch.cuda.is_available()):
        print("[Warn] CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

    loader = DataLoader(
        dataset, 
        batch_size = cfg["training"]["batch_size"],
        shuffle = False,
        num_workers = cfg["training"].get("num_workers",0),
        pin_memory = (device.type == "cuda")
        )

    # model
    # try:
    #     model = E3nnVBnet(cfg)
    # except TypeError:
    model = E3nnVBnet( hidden_irreps=cfg["model"]["hidden_irreps"]).to(device)

    optimizer = torch.optim.Adam ( 
        model.parameters(), 
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    loss_fn = torch.nn.MSELoss()

    print(dataset[0].molecule_id, dataset[0].vb_index)

    sch = cfg["training"].get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sch.get("mode","min"),
        factor=float(sch.get("factor",0.5)),
        patience=int(sch.get("patience",10)),
        min_lr=float(sch.get("min_lr",1e-6)),
        verbose=True
    )
    
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        total_loss = 0.0
        for data in loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, data.y.view(-1))
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss) # 传入监控的指标

        print(f"Epoch {epoch:03d}, Loss {total_loss / len(loader):.6f}")

if __name__ == "__main__":
    train()