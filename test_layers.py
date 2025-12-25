"""
Unit tests for shared layers
"""

import torch
import sys
sys.path.insert(0, '/Users/transformer/Desktop/water_code/water_fm')

from models.layers import (
    PositionalEncoding,
    FiLMLayerNorm,
    FiLMEncoderLayer,
    patchify,
    unpatchify,
)


def test_positional_encoding():
    print("=" * 60)
    print("TEST 1: PositionalEncoding")
    print("=" * 60)

    pe = PositionalEncoding(d_model=256, max_len=90)
    x = torch.randn(4, 90, 256)  # [B, T, d_model]

    out = pe(x)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (4, 90, 256), "Shape mismatch!"
    print(f"✓ PositionalEncoding test passed!\n")


def test_film_layer_norm():
    print("=" * 60)
    print("TEST 2: FiLMLayerNorm")
    print("=" * 60)

    film_ln = FiLMLayerNorm(d_model=256)
    x = torch.randn(4, 90, 256)  # [B, T, d_model]
    gamma = torch.randn(4, 1, 256)  # [B, 1, d_model]
    beta = torch.randn(4, 1, 256)  # [B, 1, d_model]

    out = film_ln(x, gamma, beta)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Gamma shape: {gamma.shape}")
    print(f"✓ Beta shape: {beta.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (4, 90, 256), "Shape mismatch!"

    # Test modulation effect
    x_normalized = film_ln.ln(x)
    expected = gamma * x_normalized + beta
    assert torch.allclose(out, expected, atol=1e-6), "FiLM computation incorrect!"
    print(f"✓ FiLM modulation verified!\n")


def test_film_encoder_layer():
    print("=" * 60)
    print("TEST 3: FiLMEncoderLayer")
    print("=" * 60)

    layer = FiLMEncoderLayer(d_model=256, nhead=8, dropout=0.1)
    x = torch.randn(4, 90, 256)  # [B, T, d_model]
    gamma = torch.randn(4, 1, 256)  # [B, 1, d_model]
    beta = torch.randn(4, 1, 256)  # [B, 1, d_model]

    out = layer(x, gamma, beta)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (4, 90, 256), "Shape mismatch!"
    print(f"✓ FiLMEncoderLayer test passed!\n")


def test_patchify_unpatchify():
    print("=" * 60)
    print("TEST 4: Patchify and Unpatchify")
    print("=" * 60)

    # Create test image
    B, T, H, W = 2, 90, 290, 180
    x_img = torch.randn(B, T, H, W)

    # Patchify
    patches = patchify(x_img, patch_size=10)

    print(f"✓ Input image shape: {x_img.shape}")
    print(f"✓ Patches shape: {patches.shape}")

    expected_num_patches = (290 // 10) * (180 // 10)  # 29 * 18 = 522
    expected_patch_dim = 10 * 10  # 100
    assert patches.shape == (B, T, expected_num_patches, expected_patch_dim), \
        f"Expected shape ({B}, {T}, {expected_num_patches}, {expected_patch_dim}), got {patches.shape}"

    # Unpatchify
    x_recon = unpatchify(patches, patch_size=10, image_hw=(290, 180))

    print(f"✓ Reconstructed image shape: {x_recon.shape}")
    assert x_recon.shape == (B, T, H, W), "Shape mismatch after unpatchify!"

    # Check reconstruction accuracy
    assert torch.allclose(x_img, x_recon, atol=1e-6), "Reconstruction not exact!"
    print(f"✓ Perfect reconstruction verified!")
    print(f"✓ Patchify/Unpatchify test passed!\n")


def test_with_padding_mask():
    print("=" * 60)
    print("TEST 5: FiLMEncoderLayer with Padding Mask")
    print("=" * 60)

    layer = FiLMEncoderLayer(d_model=256, nhead=8, dropout=0.0)

    # Create padded sequence [B=2, T=20, d_model=256]
    # First sample: 15 valid tokens
    # Second sample: 10 valid tokens
    B, max_T, d_model = 2, 20, 256
    x = torch.randn(B, max_T, d_model)
    gamma = torch.randn(B, 1, d_model)
    beta = torch.randn(B, 1, d_model)

    # Create padding mask (True = padding)
    padding_mask = torch.zeros(B, max_T, dtype=torch.bool)
    padding_mask[0, 15:] = True  # Last 5 positions are padding
    padding_mask[1, 10:] = True  # Last 10 positions are padding

    out = layer(x, gamma, beta, key_padding_mask=padding_mask)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Padding mask shape: {padding_mask.shape}")
    print(f"✓ Output shape: {out.shape}")
    assert out.shape == (B, max_T, d_model), "Shape mismatch!"
    print(f"✓ Padding mask test passed!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TESTING SHARED LAYERS")
    print("=" * 60 + "\n")

    test_positional_encoding()
    test_film_layer_norm()
    test_film_encoder_layer()
    test_patchify_unpatchify()
    test_with_padding_mask()

    print("=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60 + "\n")
