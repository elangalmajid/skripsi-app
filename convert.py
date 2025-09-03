# extract_to_cpu.py
import os
import torch

OLD_CKPT = "m_MobileViG_s_50_epoch_0_d.pth"
NEW_CKPT = "mobilevig_cpu_clean.pth"

def move_tensors_to_cpu(obj):
    """ Rekursif pindahkan semua tensor ke CPU dari struktur nested dict/list/tuple. """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [move_tensors_to_cpu(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj

def main():
    try:
        import torch_directml as tdm
        dml = tdm.device()
        print("✅ torch-directml terdeteksi. Menggunakan device DML untuk membuka checkpoint.")
    except Exception as e:
        print("❌ torch-directml belum terpasang atau gagal import.")
        print("   Jalankan:  pip install torch-directml")
        raise e

    if not os.path.exists(OLD_CKPT):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {os.path.abspath(OLD_CKPT)}")

    print(f"🔄 Memuat checkpoint DML: {OLD_CKPT}")
    ckpt = torch.load(OLD_CKPT, map_location=dml)

    # Beberapa checkpoint menaruh bobot di 'state_dict'
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    print("🧹 Memindahkan semua tensor ke CPU...")
    state_cpu = move_tensors_to_cpu(state)

    # 🔎 Cek layer classifier terakhir
    head_keys = [k for k in state_cpu.keys() if "classifier" in k or "head" in k or "fc" in k]
    print("📊 Key output layer yang terdeteksi:")
    for k in head_keys:
        print(f"  {k}: {state_cpu[k].shape}")

    print(f"💾 Menyimpan ulang CPU-only state_dict: {NEW_CKPT}")
    torch.save(state_cpu, NEW_CKPT)
    print("✅ Selesai. Gunakan file NEW_CKPT ini di Streamlit.")

if __name__ == "__main__":
    main()
