import time

import torch


def benchmark_latency(model, input_tensor, warmup_runs=20, measure_runs=100):
    model.cuda()
    model.eval()
    input_tensor = input_tensor.cuda()

    with torch.no_grad():
        for _ in range(warmup_runs):
            model(input_tensor)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(measure_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(input_tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    avg_ms = sum(times) / len(times) * 1000
    return avg_ms


def benchmark_multiple_inputs(model, input_sizes, warmup_runs=10, measure_runs=50):
    model.cuda()
    model.eval()
    results = {}

    for size in input_sizes:
        x = torch.randn(1, 3, size, size, device="cuda")

        with torch.no_grad():
            for _ in range(warmup_runs):
                model(x)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(measure_runs):
                model(x)
        torch.cuda.synchronize()
        total = time.perf_counter() - start

        results[size] = total / measure_runs * 1000

    return results
