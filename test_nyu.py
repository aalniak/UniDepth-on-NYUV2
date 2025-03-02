"""
Script Name: my_script.py
Author: Your Name
Date: YYYY-MM-DD

This script is built on top of 'UniDepth & UniDepthV2' (https://github.com/lpiccinelli-eth/UniDepth).
All credit for the base functionality goes to the original developers.
"""
import argparse
import time
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from unidepth.models import UniDepthV1, UniDepthV2
import os

home_dir = os.environ["HOME"] # to save the nyu_cache
print(home_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Depth Estimation using UniDepthV1 or UniDepthV2")
    parser.add_argument("--model", type=str, choices=["v1", "v2"], required=True,
                        help="Select UniDepth model version: 'v1' or 'v2'")
    return parser.parse_args()


def load_model(model_version):
    """Load the specified UniDepth model."""
    if model_version == "v1":
        model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")  # or "lpiccinelli/unidepth-v1-cnvnxtl"
    else:
        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")  # or "lpiccinelli/unidepth-v2-cnvnxtl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def compute_depth_metrics(pred_depth, gt_depth):
    """Compute depth estimation error metrics: RMSE, log RMSE, AbsRel, SqRel, and SI Log Error."""
    valid_mask = gt_depth > 0  # Avoid invalid depth values
    pred_depth = pred_depth.squeeze(0).squeeze(0)
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    # RMSE
    rmse = np.sqrt(mean_squared_error(gt_depth, pred_depth))
    
    # Log RMSE
    log_rmse = np.sqrt(mean_squared_error(np.log(gt_depth), np.log(pred_depth)))
    
    # Absolute Relative Difference
    absrel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
    
    # Squared Relative Difference
    sqrel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)
    
    # Scale Invariant Log Error
    log_diff = np.log(pred_depth) - np.log(gt_depth)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
    
    return {
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "AbsRel": absrel,
        "SqRel": sqrel,
        "SI Log Error": silog
    }


def process_dataset(dataset, model):
    """Process dataset images and compute depth metrics."""
    total_metrics = {"RMSE": 0, "Log RMSE": 0, "AbsRel": 0, "SqRel": 0, "SI Log Error": 0}
    num_samples = 0
    
    for idx, sample in enumerate(list(dataset)):
        print(f"Processing Sample {idx}")
        
        rgb_image = sample['image']  # RGB image
        rgb_image = np.array(rgb_image)
        rgb_image = torch.from_numpy(rgb_image).float()
        rgb_image = rgb_image.permute(2, 0, 1)  # Change (H, W, C) -> (C, H, W)
        
        gt_depth = np.array(sample['depth_map'])  # Ground truth depth
        
        start = time.time()
        inference = model.infer(rgb_image)
        end = time.time()
        
        inferred_depth = inference["depth"].cpu().numpy()
        
        # Compute depth metrics
        metrics = compute_depth_metrics(inferred_depth, gt_depth)
        
        # Accumulate metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        num_samples += 1
        #print(f"Unique ID (Scene Name): {sample.get('scene', 'Unknown')}")
        #print(f"Depth Map Hash: {hash(gt_depth.tobytes())}")  # Ensure unique depth values
        print(f"Metrics: {metrics}")
        print(f"Inference Time for the sample: {end - start:.4f} seconds")
    
    # Compute average metrics
    avg_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}
    print(f"Average Metrics: {avg_metrics}")
    return avg_metrics


if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"Loading model: UniDepth {args.model}")
    model = load_model(args.model)
    
    dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train[:40000]", cache_dir=home_dir+"/nyu_cache")
    #dataset = dataset.select(range(0, 40000, 40))  # Sample dataset
    
    process_dataset(dataset, model)
