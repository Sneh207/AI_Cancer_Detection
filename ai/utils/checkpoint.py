import os
try:
    import torch
except Exception:
def load_checkpoint_if_available(model, ckpt_path, map_location=None, strict=True):
    """
    Try to load a checkpoint. If missing, return False (caller can use pretrained weights).
    ckpt_path may be a local path or a URL.
    """
    if not ckpt_path:
        return False

    if torch is None:
        print("PyTorch (torch) is not installed; skipping checkpoint load.")
        return False

    # download if URL
    if ckpt_path.startswith("http://") or ckpt_path.startswith("https://"):
        os.makedirs("ai/checkpoints", exist_ok=True)
        local = os.path.join("ai", "checkpoints", os.path.basename(ckpt_path))
        if not os.path.exists(local):
            try:
                urllib.request.urlretrieve(ckpt_path, local)
            except Exception as e:
                print(f"Failed to download checkpoint: {e}")
                return False
        ckpt_path = local

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return False

    try:
        state = torch.load(ckpt_path, map_location=map_location or torch.device("cpu"))
        if "state_dict" in state:
            state = state["state_dict"]
        # optional compatibility sanitizing can be done here
        model.load_state_dict(state, strict=strict)
        print(f"Loaded checkpoint: {ckpt_path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        return False
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        return False