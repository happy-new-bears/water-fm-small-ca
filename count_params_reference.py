"""
Calculate parameter count for the MultimodalFoundationModel
"""

import sys
sys.path.insert(0, '/Users/transformer/Desktop/water_code/riverflow_prediction-main')

import torch
from foundation.modules import MultimodalFoundationModel


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    print("=" * 60)
    print("MultimodalFoundationModel Parameter Count")
    print("=" * 60)

    # Model configuration from water-model-ds.py
    config = {
        'img_height': 160,  # Estimated based on typical ERA5 data
        'img_width': 160,   # Estimated
        'patch_size': 16,
        'in_chans': 1,
        'embed_dim': 512,
        'encoder_depth': 12,
        'encoder_num_heads': 8,
        'mlp_ratio': 4.0,
        'catchment_size': 604,  # From CAMELS-GB dataset
        'feature_size': 3,      # precipitation, pet, temperature
        'temporal_encoder_depth': 4,
        'temporal_encoder_num_heads': 8,
        'decoder_embed_dim': 512 * 2,  # 1024
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'image_mask_ratio': 0.75,
        'tabular_mask_ratio': 0.5,
        'max_timesteps': 15,  # lookback=15 from the script
    }

    print("\nModel Configuration:")
    print(f"  Image size: {config['img_height']}x{config['img_width']}")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Encoder depth: {config['encoder_depth']}")
    print(f"  Encoder heads: {config['encoder_num_heads']}")
    print(f"  Decoder embed dim: {config['decoder_embed_dim']}")
    print(f"  Decoder depth: {config['decoder_depth']}")
    print(f"  Decoder heads: {config['decoder_num_heads']}")
    print(f"  Temporal encoder depth: {config['temporal_encoder_depth']}")
    print(f"  Temporal encoder heads: {config['temporal_encoder_num_heads']}")

    print("\nCreating model...")
    model = MultimodalFoundationModel(**config)

    print("✓ Model created successfully")

    # Count parameters
    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 60)
    print("Parameter Count:")
    print("=" * 60)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Model size (MB):       {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32

    # Count parameters by submodule
    print("\n" + "=" * 60)
    print("Parameters by Submodule:")
    print("=" * 60)

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:40s}: {params:>12,} ({params/total_params*100:>5.2f}%)")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
