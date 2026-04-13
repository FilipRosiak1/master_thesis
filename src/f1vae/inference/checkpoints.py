from __future__ import annotations

import torch


def load_checkpoint(path: str, device: torch.device) -> tuple[dict, dict]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("meta", {})

    return payload, {}
