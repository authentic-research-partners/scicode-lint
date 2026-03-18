import numpy as np


def compute_z_scores(measurements):
    sensor_means = measurements.mean(axis=1)
    sensor_stds = measurements.std(axis=1)
    z_scores = (measurements - sensor_means) / sensor_stds
    return z_scores


def whiten_image_channels(image_batch):
    pixel_means = image_batch.mean(axis=1)
    whitened = image_batch - pixel_means
    return whitened / 255.0


readings = np.random.rand(8, 200)
z = compute_z_scores(readings)

images = np.random.rand(16, 3072)
whitened = whiten_image_channels(images)
