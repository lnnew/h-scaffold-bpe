#!/usr/bin/env python
"""Run this FIRST on the target machine and send the JSON back.  Collects everything that decides
precision, parallelism, batch size and data access.  Makes no changes to the system."""
import json, os, platform, shutil, subprocess, sys, time


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERR {e}"


def main():
    r = dict(platform=platform.platform(), python=sys.version.split()[0], cpus=os.cpu_count(),
             ram_gb=None, disk_free_gb=round(shutil.disk_usage(".").free / 1e9, 1))
    try:
        r["ram_gb"] = round(int(open("/proc/meminfo").readline().split()[1]) / 1e6, 1)
    except Exception:
        pass
    r["nvidia_smi"] = sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv")
    r["topology"] = sh("nvidia-smi topo -m")
    r["nvlink"] = sh("nvidia-smi nvlink --status")
    pk = {}
    for m in ("torch", "numpy", "scipy", "sklearn", "regex", "datasets", "tokenizers", "triton", "transformers"):
        try:
            mod = __import__(m)
            pk[m] = getattr(mod, "__version__", "?")
        except Exception as e:
            pk[m] = f"MISSING ({type(e).__name__})"
    r["packages"] = pk
    try:
        import torch
        r["torch"] = dict(version=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                          available=torch.cuda.is_available(), n_gpu=torch.cuda.device_count())
        if torch.cuda.is_available():
            gp = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                g = dict(name=p.name, mem_gb=round(p.total_memory / 1e9, 1), capability=f"{p.major}.{p.minor}", sm=p.multi_processor_count)
                try:
                    g["bf16_supported_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                except Exception as e:
                    g["bf16_supported_native"] = f"? {e}"
                torch.cuda.set_device(i)
                for dt in (torch.float16, torch.float32):
                    n = 8192
                    a = torch.randn(n, n, device=f"cuda:{i}", dtype=dt); b = torch.randn(n, n, device=f"cuda:{i}", dtype=dt)
                    for _ in range(3): a @ b
                    torch.cuda.synchronize(); t = time.time()
                    for _ in range(20): a @ b
                    torch.cuda.synchronize()
                    g[f"matmul_{str(dt).split('.')[-1]}_TFLOPs"] = round(20 * 2 * n ** 3 / (time.time() - t) / 1e12, 1)
                gp.append(g)
            r["gpus"] = gp
            if torch.cuda.device_count() >= 2:
                r["p2p_access_0_1"] = bool(torch.cuda.can_device_access_peer(0, 1))
                x = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
                y = torch.empty_like(x, device="cuda:1")
                for _ in range(2): y.copy_(x)
                torch.cuda.synchronize(); t = time.time()
                for _ in range(10): y.copy_(x)
                torch.cuda.synchronize()
                r["copy_0_to_1_GBps"] = round(10 * x.numel() / (time.time() - t) / 1e9, 1)
            # which SDPA backends actually run on this GPU (flash needs sm80+, mem-efficient works on sm75)
            import torch.nn.functional as F
            from torch.nn.attention import SDPBackend, sdpa_kernel
            q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
            sd = {}
            for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
                try:
                    with sdpa_kernel(be):
                        F.scaled_dot_product_attention(q, q, q, is_causal=True)
                    sd[be.name] = "ok"
                except Exception as e:
                    sd[be.name] = "unavailable"
            r["sdpa_backends"] = sd
    except Exception as e:
        r["torch_error"] = repr(e)
    r["network"] = {u: sh(f"curl -s -o /dev/null -m 8 -w '%{{http_code}}' {u}") for u in
                    ("https://huggingface.co", "https://pypi.org", "https://github.com")}
    r["env_vars"] = {k: os.environ.get(k, "") for k in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "HTTPS_PROXY")}
    out = sys.argv[1] if len(sys.argv) > 1 else "env_report.json"
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
