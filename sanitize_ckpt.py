# sanitize_ckpt.py  (versi robust)
import sys, os, torch

IN  = sys.argv[1] if len(sys.argv) > 1 else "m_MobileViG_ti_50_epoch_0.5_d.pth"
OUT = sys.argv[2] if len(sys.argv) > 2 else "mobilevig_state_cpu_only.pth"

def try_load_cpu(path):
    # Coba varian standar lebih dulu
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        # Monkey-patch untuk abaikan device PrivateUse1 saat rebuild dari numpy
        import torch._utils as _u
        orig = _u._rebuild_device_tensor_from_numpy

        def _cpu_rebuild(*args):
            """
            Handle dua kemungkinan signature:
              (data, dtype, device, requires_grad)
              (data, dtype, device, requires_grad, strides, offset)
            Abaikan 'device' PrivateUse1, paksa tensor dibuat di CPU.
            """
            if len(args) == 4:
                data, dtype, device, requires_grad = args
                t = torch.from_numpy(data).to(dtype=dtype)
                t.requires_grad = requires_grad
                return t
            elif len(args) == 6:
                data, dtype, device, requires_grad, strides, offset = args
                t = torch.from_numpy(data).to(dtype=dtype)
                if strides is not None and offset is not None:
                    t = t.as_strided(data.shape, strides, offset)
                t.requires_grad = requires_grad
                return t
            # fallback ke implementasi asli kalau formatnya lain
            return orig(*args)

        _u._rebuild_device_tensor_from_numpy = _cpu_rebuild
        try:
            obj = torch.load(path, map_location="cpu")
        finally:
            _u._rebuild_device_tensor_from_numpy = orig
        return obj

def clean_state_dict(obj):
    # Ambil state_dict dari checkpoint; kalau bukan dict, anggap sudah state_dict
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    clean = {}
    for k, v in state.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("ema."):    k = k[4:]
        clean[k] = v.cpu()
    return clean

def main():
    ap_in  = os.path.abspath(IN)
    ap_out = os.path.abspath(OUT)
    if not os.path.exists(ap_in):
        raise FileNotFoundError(f"Tidak ditemukan: {ap_in}")

    print(f"[INFO] membaca: {ap_in}")
    obj = try_load_cpu(ap_in)
    sd  = clean_state_dict(obj)
    torch.save(sd, ap_out)
    print(f"[OK] tulis CPU-only state_dict → {ap_out} ({len(sd)} tensors)")
    print("Contoh keys:", list(sd)[:8])

if __name__ == "__main__":
    main()
