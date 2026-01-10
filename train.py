import random
import torch
import yaml
import numpy as np
import os
from datetime import datetime
from torch_geometric.loader import DataLoader
from pyg_VB_dataset import VBGraphDataset
from vb_model import E3nnVBnet

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_print(msg, log_file=None):
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()

def train():
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed",0)
    set_seed(seed)

    dataset = VBGraphDataset(
        cfg=cfg,
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
        shuffle = True,
        num_workers = cfg["training"].get("num_workers",0),
        pin_memory = (device.type == "cuda")
    )

    model = E3nnVBnet(cfg).to(device)

    optimizer = torch.optim.Adam ( 
        model.parameters(), 
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0))
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
        total_abs_error = 0.0
        total_sq_error = 0.0
        total_samples = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            pred = model(data)
            out_act = cfg.get("training", {}).get("output_activation", "none").lower()
            if out_act == "sigmoid":
                pred = torch.sigmoid(pred)
            y = data.y.view(-1)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()

            batch_size = y.numel()
            total_abs_error += torch.sum(torch.abs(pred - y)).item()
            total_sq_error += torch.sum((pred - y) ** 2).item()
            total_samples += batch_size
        
        avg_mse = total_sq_error / total_samples
        mae = total_abs_error / total_samples
        rmse = avg_mse ** 0.5
        scheduler.step(avg_mse) # 传入监控的指标

        print(
            f"Epoch {epoch:03d} | "
            f"Loss(MSE)={avg_mse:.6f} | "
            f"MAE={mae:.6f} | "
            f"RMSE={rmse:.6f}"
        )

if __name__ == "__main__":
    train()