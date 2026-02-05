import os 
import torch
import torch.nn as nn

def save_checkpoint(
    path: str,
    round_num: int,
    global_model: nn.Module,
    metrics: dict
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "round": round_num,
        "global_model": global_model.state_dict(),
        "metrics": metrics
    }, path)

    print(f"[Checkpoint] Saved at round {round_num}")

def load_checkpoint(
    path: str,
    global_model: nn.Module,
    device    
) : 
    if not os.path.exists(path):
        return None

    checkpoint  = torch.load(path, map_location=device)  

    global_model.load_state_dict(checkpoint["global_model"])

    print(f"[Checkpoint] Resumed from round {checkpoint['round']}") 

    return {
        "round": checkpoint["round"],
        "metrics" : checkpoint["metrics"]
    }
