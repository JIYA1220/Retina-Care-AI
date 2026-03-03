"""
temperature_scaling.py
Calibrate the model confidence using post-hoc temperature scaling.
Learns a single scalar T (temperature) on the validation set logits.
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import load_model
from utils.dataset import APTOSDataset, get_val_transforms

class ModelWithTemperature(nn.Module):
    """
    A helper module that divides logits by a learned temperature (T).
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize slightly > 1

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits.
        """
        # Expand temperature to match batch size
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Optimize the temperature (T) on the validation set using L-BFGS.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        # 1. Collect all logits and labels from the validation set
        logits_list = []
        labels_list = []
        
        device = torch.device("cpu")
        print("Collecting validation logits...")
        with torch.no_grad():
            for input, label in tqdm(valid_loader):
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        # 2. Optimize temperature T using L-BFGS
        # We only optimize 'self.temperature'
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        print(f"Initial Temperature: {self.temperature.item():.4f}")
        optimizer.step(eval_loss)
        print(f"Optimized Temperature: {self.temperature.item():.4f}")

        # 3. Save the learned temperature (model-specific)
        temp_save_path = f"model/{self.model_name}_temp.pt"
        torch.save(self.temperature, temp_save_path)
        print(f"Saved temperature -> {temp_save_path}")

def calibrate_model(model_name="efficientnet_b0", weights_path="model/efficientnet_b0_dr.pth", data_dir="data"):
    """
    Full calibration pipeline:
    1. Load model and validation data
    2. Run temperature optimization
    """
    device = torch.device("cpu")
    model = load_model(weights_path, model_name, device)
    
    val_csv = os.path.join(data_dir, "_val_split.csv")
    img_dir = os.path.join(data_dir, "train_images")
    
    if not os.path.exists(val_csv):
        print(f"Error: Validation split CSV not found at {val_csv}. Run train.py first.")
        return

    val_ds = APTOSDataset(val_csv, img_dir, img_size=160, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    scaled_model = ModelWithTemperature(model)
    scaled_model.model_name = model_name # Pass name to class
    scaled_model.set_temperature(val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()
    
    # Auto-set weights path if not provided
    weights = args.weights if args.weights else f"model/{args.model}_dr.pth"
    
    calibrate_model(args.model, weights, args.data_dir)
