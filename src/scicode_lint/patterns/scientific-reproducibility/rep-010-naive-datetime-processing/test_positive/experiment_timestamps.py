from datetime import datetime


class ExperimentTracker:

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.start_time = datetime.now()
        self.results = []

    def record_measurement(self, sample_id: int, value: float) -> None:
        self.results.append(
            {
                "sample_id": sample_id,
                "value": value,
                "measured_at": datetime.now(),
            }
        )

    def filter_results_by_time(self, cutoff_hour: int):
        return [
            r for r in self.results
            if r["measured_at"].hour >= cutoff_hour
        ]

    def save_dataset(self) -> list:
        return [
            {
                "experiment": self.name,
                "start": self.start_time,
                "measurements": self.results,
            }
        ]
