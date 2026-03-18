from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def process_sensor_reading(sensor_id):
    rng = np.random.default_rng(seed=sensor_id)
    readings = rng.normal(0, 1, size=256)
    return readings


def build_sensor_matrix(sensor_ids):
    rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_sensor_reading, sid) for sid in sensor_ids]
        for future in as_completed(futures):
            rows.append(future.result())
    matrix = np.vstack(rows)
    return matrix


sensor_ids = list(range(20))
sensor_matrix = build_sensor_matrix(sensor_ids)
baseline = sensor_matrix[0]
diffs = sensor_matrix - baseline
per_sensor_drift = np.mean(diffs, axis=1)
