"""
Convert a single RGB-D frame with CLIP features to a 3D point cloud.

This script takes:
- RGB image
- Depth image
- Pre-extracted CLIP features (from extract_conceptfusion_features.py)
- Camera intrinsics

And outputs:
- 3D point cloud with CLIP embeddings attached to each point
- Saved in .h5 format compatible with demo scripts
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import torch
import tyro
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages


@dataclass
class ProgramArgs:
    """Command-line arguments for single-frame point cloud generation"""

    # Path to RGB image
    rgb_path: Union[str, Path] = "data/droid_single/images/left/00000.png"

    # Path to depth image (16-bit PNG in millimeters)
    depth_path: Union[str, Path] = "data/droid_single/images/depth_mm/00000.png"

    # Path to extracted CLIP features (.pt file from Phase 1)
    features_path: Union[str, Path] = "saved-feat-droid-single/00000.pt"

    # Path to camera intrinsics JSON
    intrinsics_path: Union[str, Path] = (
        "data/droid_single/images/camera_intrinsics.json"
    )

    # Depth scale (millimeters to meters)
    depth_scale: float = 1000.0

    # Directory to save output point cloud
    output_dir: Union[str, Path] = "saved-map-droid-single"

    # Device
    device: str = "cuda:0"


def load_camera_intrinsics(intrinsics_path: Path) -> torch.Tensor:
    with open(intrinsics_path, "r") as f:
        data = json.load(f)

    K = torch.eye(4, dtype=torch.float32)
    camera_matrix = data["camera_matrix"]
    K[0, 0] = camera_matrix[0][0]  # fx
    K[1, 1] = camera_matrix[1][1]  # fy
    K[0, 2] = camera_matrix[0][2]  # cx
    K[1, 2] = camera_matrix[1][2]  # cy

    return K


def main():
    args = tyro.cli(ProgramArgs)

    print("Loading data...")

    # Load RGB image
    rgb = cv2.imread(str(args.rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = torch.from_numpy(rgb).float() / 255.0  # Normalize to [0, 1]
    H, W = rgb.shape[:2]
    print(f"  RGB: {rgb.shape}")

    # Load depth image
    depth = cv2.imread(str(args.depth_path), cv2.IMREAD_UNCHANGED)
    depth = torch.from_numpy(depth).float() / args.depth_scale  # Convert mm to meters
    depth = depth.unsqueeze(-1)  # Add channel dimension
    print(
        f"  Depth: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}] meters"
    )

    # Load CLIP features
    features = torch.load(str(args.features_path))  # (H, W, feature_dim)
    feat_H, feat_W, feat_dim = features.shape
    print(f"  Features: {features.shape}")

    # Resize features to match RGB/depth if needed
    if feat_H != H or feat_W != W:
        print(f"  Resizing features from ({feat_H}, {feat_W}) to ({H}, {W})")
        features = features.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        features = torch.nn.functional.interpolate(
            features, size=(H, W), mode="bilinear", align_corners=False
        )
        features = features.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        # Re-normalize after interpolation (bilinear changes norms)
        features = torch.nn.functional.normalize(features, dim=-1)

    # Load camera intrinsics
    K = load_camera_intrinsics(args.intrinsics_path)
    print(
        f"  Intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}"
    )

    # # Create identity pose (static camera at origin)
    # pose = torch.eye(4, dtype=torch.float32)

    print("\nCreating RGBDImages object...")
    # Add batch and sequence dimensions: (H, W, C) -> (1, 1, H, W, C)
    rgbd = RGBDImages(
        rgb_image=rgb.unsqueeze(0).unsqueeze(0),
        depth_image=depth.unsqueeze(0).unsqueeze(0),
        intrinsics=K.unsqueeze(0).unsqueeze(0),
        # poses=pose.unsqueeze(0).unsqueeze(0),
        embeddings=features.unsqueeze(0).unsqueeze(0),
        has_embeddings=True,
        embeddings_dim=feat_dim,
    ).to(args.device)

    print("Back-projecting to 3D...")
    # Get 3D vertex map (back-projects depth to 3D coordinates)
    vertex_map = rgbd.global_vertex_map  # (1, 1, H, W, 3)
    normal_map = rgbd.global_normal_map  # (1, 1, H, W, 3)

    # Filter valid depths
    valid_mask = rgbd.valid_depth_mask.squeeze(-1)[0, 0]  # (H, W)
    num_valid = valid_mask.sum().item()
    print(f"  Valid points: {num_valid} / {H * W} ({100 * num_valid / (H * W):.1f}%)")

    # Extract valid points, colors, normals, embeddings
    valid_mask_cpu = valid_mask.cpu()
    points = vertex_map[0, 0][valid_mask]
    colors = rgb[valid_mask_cpu]
    normals = normal_map[0, 0][valid_mask]
    embeddings = features[valid_mask_cpu]

    print(f"  Points shape: {points.shape}")
    print(f"  Colors shape: {colors.shape}")
    print(f"  Normals shape: {normals.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")

    print("\nCreating Pointclouds object...")
    # Create point cloud with features
    pointcloud = Pointclouds(
        points=points.unsqueeze(0).to(args.device),  # (1, N, 3)
        colors=colors.unsqueeze(0).to(args.device),  # (1, N, 3)
        normals=normals.unsqueeze(0).to(args.device),  # (1, N, 3)
        embeddings=embeddings.unsqueeze(0).to(args.device),  # (1, N, feat_dim)
    )

    print(f"Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save point cloud
    pointcloud.save_to_h5(args.output_dir, index=0)

    print(f"\nsaved point cloud to {args.output_dir}/")


if __name__ == "__main__":
    main()
