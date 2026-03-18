import statistics

import torch


class LatencyTracker:
    def __init__(self, device_id=0):
        self.device = torch.device(f"cuda:{device_id}")
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def _run_passes(self, net, tensor, count):
        for _ in range(count):
            net(tensor)

    def collect_samples(self, net, tensor, n_warmup=20, n_samples=200):
        net.to(self.device)
        net.eval()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            self._run_passes(net, tensor, n_warmup)

        torch.cuda.synchronize(self.device)
        measurements = []

        with torch.no_grad():
            for _ in range(n_samples):
                self.start_event.record()
                net(tensor)
                self.end_event.record()
                torch.cuda.synchronize(self.device)
                measurements.append(self.start_event.elapsed_time(self.end_event))

        return measurements


def report_percentiles(net, resolution=(1, 3, 512, 512), reps=300):
    tracker = LatencyTracker()
    sample_input = torch.randn(*resolution, device="cuda")

    ms_values = tracker.collect_samples(net, sample_input, n_warmup=50, n_samples=reps)

    sorted_ms = sorted(ms_values)
    p50 = sorted_ms[len(sorted_ms) // 2]
    p95 = sorted_ms[int(len(sorted_ms) * 0.95)]
    p99 = sorted_ms[int(len(sorted_ms) * 0.99)]
    mean = statistics.mean(ms_values)
    std = statistics.stdev(ms_values)

    return {
        "mean_ms": mean,
        "std_ms": std,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "num_samples": len(ms_values),
    }
