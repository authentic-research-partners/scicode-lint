import time
from datetime import datetime


def record_experiment(experiment_name, results):
    timestamp = datetime.now()
    return {"name": experiment_name, "timestamp": timestamp.isoformat(), "results": results}


def record_event(event_type):
    event = {
        "type": event_type,
        "time": datetime.now(),
        "unix_ts": datetime.fromtimestamp(time.time()),
    }
    return event


def save_measurement(sensor_data):
    return {
        "data": sensor_data,
        "recorded_at": datetime.now(),
    }
