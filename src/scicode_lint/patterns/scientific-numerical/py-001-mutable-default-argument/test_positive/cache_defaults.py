import numpy as np


class DataProcessor:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def register(self, fn):
        self.transforms.append(fn)

    def run(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class MetricTracker:
    def __init__(self, window_values=[]):
        self.window = window_values

    def update(self, value):
        self.window.append(value)
        if len(self.window) > 50:
            self.window.pop(0)
        return np.mean(self.window)


class ExperimentRegistry:
    def __init__(self, completed_ids=set()):
        self.completed = completed_ids

    def mark_done(self, exp_id):
        self.completed.add(exp_id)

    def is_done(self, exp_id):
        return exp_id in self.completed


proc1 = DataProcessor()
proc1.register(lambda x: x * 2)
proc2 = DataProcessor()
