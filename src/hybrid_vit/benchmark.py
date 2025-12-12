from __future__ import annotations

import time
import numpy as np
import torch


@torch.no_grad()
def measure_inference_time(model, loader, device, num_batches: int = 50):
    model.eval()
    times = []
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        t0 = time.time()
        _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        times.append(time.time() - t0)

    times = np.array(times)
    return times.mean(), times.std()