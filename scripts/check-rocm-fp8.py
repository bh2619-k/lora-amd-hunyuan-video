#!/usr/bin/env python3
"""
ROCm Validation Script for HunyuanVideo Training
Checks PyTorch ROCm setup, GPU capabilities, bf16/fp8 support.
"""

import sys
import time


def check_pytorch():
    """Check PyTorch installation and ROCm availability."""
    print("=" * 60)
    print("ROCm / PyTorch Validation for HunyuanVideo Training")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("\n❌ PyTorch not installed!")
        print("   Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2")
        return False

    print(f"\n📦 PyTorch version: {torch.__version__}")
    print(f"   CUDA/ROCm available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n❌ No GPU detected via torch.cuda!")
        print("   Possible causes:")
        print("   - ROCm not installed or not in PATH")
        print("   - Wrong PyTorch build (CPU-only or wrong ROCm version)")
        print("   - GPU not supported by installed ROCm version")
        return False

    device_count = torch.cuda.device_count()
    print(f"   GPU count: {device_count}")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        print(f"\n🖥️  GPU {i}: {name}")
        print(f"   VRAM: {mem:.1f} GB")

        if mem < 24:
            print(f"   ⚠️  Warning: {mem:.0f}GB may be insufficient. 32GB+ recommended.")
        elif mem >= 32:
            print(f"   ✅ Sufficient VRAM for HV1.5 training with blocks_to_swap=20")
        else:
            print(f"   ✅ Sufficient VRAM (may need higher blocks_to_swap)")

    return True


def check_bf16():
    """Check bf16 support."""
    import torch

    print("\n" + "-" * 40)
    print("🔬 bf16 Support Check")
    print("-" * 40)

    try:
        a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        c = torch.matmul(a, b)
        assert c.dtype == torch.bfloat16
        print("✅ bf16 matmul: WORKING")
        del a, b, c
        return True
    except Exception as e:
        print(f"❌ bf16 matmul: FAILED ({e})")
        return False


def check_fp8():
    """Check fp8 _scaled_mm support (for inference optimization)."""
    import torch

    print("\n" + "-" * 40)
    print("🔬 fp8 Support Check (for inference)")
    print("-" * 40)

    if not hasattr(torch, "_scaled_mm"):
        print("❌ torch._scaled_mm not available (PyTorch too old?)")
        return False

    # Test e4m3fn (NVIDIA format, used by FLUX/HV1.5 fp8 checkpoints)
    test_sizes = [(256, 256), (1024, 1024), (4096, 4096)]
    all_pass = True

    for m, n in test_sizes:
        try:
            a = torch.randn(m, n, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(n, m, device="cuda").to(torch.float8_e4m3fn).t()
            scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            result = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            print(f"   ✅ e4m3fn {m}×{n}: PASS")
            del a, b, scale_a, scale_b, result
        except Exception as e:
            print(f"   ❌ e4m3fn {m}×{n}: FAIL ({e})")
            all_pass = False

    # Test e4m3fnuz (AMD native format)
    try:
        a = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fnuz)
        b = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fnuz).t()
        scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        result = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        print(f"   ✅ e4m3fnuz: PASS")
        del a, b, scale_a, scale_b, result
    except Exception as e:
        print(f"   ⚠️  e4m3fnuz: FAIL (expected — hipBLAS unsupported)")

    if all_pass:
        print("\n✅ fp8 e4m3fn: FULLY WORKING — fp8 inference will work!")
    else:
        print("\n⚠️  fp8 e4m3fn: PARTIAL/BROKEN — use bf16 for inference")

    return all_pass


def check_gradient_clip():
    """Quick test to verify gradient clipping works (critical for AMD bf16 training)."""
    import torch

    print("\n" + "-" * 40)
    print("🔬 Gradient Clipping Check")
    print("-" * 40)

    try:
        model = torch.nn.Linear(64, 64, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        loss = model(x).sum()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"   ✅ Gradient clipping: WORKING (norm: {grad_norm:.4f})")
        del model, x, loss
        return True
    except Exception as e:
        print(f"   ❌ Gradient clipping: FAILED ({e})")
        return False


def run_quick_benchmark():
    """Quick matmul benchmark to sanity-check GPU performance."""
    import torch

    print("\n" + "-" * 40)
    print("⚡ Quick Performance Check")
    print("-" * 40)

    # Warmup
    a = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(100):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    tflops = (2 * 2048**3 * 100) / elapsed / 1e12
    print(f"   bf16 matmul 2048×2048 (100 iters): {elapsed:.2f}s")
    print(f"   Estimated TFLOPS: {tflops:.1f}")

    del a, b


def main():
    if not check_pytorch():
        sys.exit(1)

    bf16_ok = check_bf16()
    fp8_ok = check_fp8()
    grad_ok = check_gradient_clip()

    if bf16_ok:
        run_quick_benchmark()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if bf16_ok and grad_ok:
        print("✅ Ready for HunyuanVideo 1.5 LoRA training (bf16)")
        if fp8_ok:
            print("✅ fp8 inference available (halves VRAM for generation)")
        else:
            print("⚠️  fp8 not available — use bf16 for inference (needs more VRAM)")
    else:
        print("❌ Issues detected — see above for details")
        sys.exit(1)

    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
