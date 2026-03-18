import time

import torch


def benchmark_inference_passes(model, x, runs=50):
    model.cuda()
    x = x.cuda()
    model.eval()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return sum(times) / len(times)


def compare_models(models, x):
    x = x.cuda()
    results = {}
    for name, model in models.items():
        start = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        results[name] = time.time() - start
    return results
