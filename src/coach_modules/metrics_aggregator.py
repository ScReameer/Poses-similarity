class MetricsAggregator:
    def __init__(self, metrics_dict: dict) -> None:
        self.metrics_dict = metrics_dict
        
    def set_cumulative_zeros(self) -> None:
        self.cumulative_metrics = {metric_name: .0 for metric_name in self.metrics_dict.keys()}
    
    def get_metrics_per_batch(self, nn_output: dict) -> dict:
        return {metric_name: metric(**nn_output).cpu().item() for metric_name, metric in self.metrics_dict.items()}
    
    def accumulate_per_batch(self, metrics_per_batch: dict) -> None:
        for metric_name, value in metrics_per_batch.items():
            self.cumulative_metrics[metric_name] += value
            
    def get_averaged_metrics(self, n_batches: int) -> dict:
        for metric_name in self.cumulative_metrics.keys():
            self.cumulative_metrics[metric_name] /= n_batches
        return self.cumulative_metrics
            
    