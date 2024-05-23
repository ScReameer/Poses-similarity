class MetricsAggregator:
    def __init__(self, metrics_dict: dict) -> None:
        """Auxiliary class for working with metrics

        Args:
            metrics_dict (`dict`): dictionary with metrics as objects
        """
        self.metrics_dict = metrics_dict
        
    def set_cumulative_zeros(self) -> None:
        """Initializes cumulative variables of each metric for further averaging
        """
        self.cumulative_metrics = {
            metric_name: .0 
            for metric_name in self.metrics_dict.keys()
        }
    
    def get_metrics_per_batch(self, nn_output: dict) -> dict:
        """Returns a dictionary of calculated metrics for a single batch

        Args:
            `nn_output` (`dict`): 2 outputs from keypoint detector in dictionary with 2 keys: `'reference_output'` and `'actual_output'`

        Returns:
            `dict`: calculated metrics for a single batch
        """
        return {
            metric_name: metric(**nn_output).cpu().item() 
            for metric_name, metric in self.metrics_dict.items()
        }
    
    def accumulate_per_batch(self, metrics_per_batch: dict) -> None:
        """Accumulates the output from `get_metrics_per_batch` function for further averaging

        Args:
            metrics_per_batch (`dict`): output from `get_metrics_per_batch`
        """
        for metric_name, value in metrics_per_batch.items():
            self.cumulative_metrics[metric_name] += value
            
    def get_averaged_metrics(self, n_batches: int) -> dict:
        """Returns the average accumulated metrics among all batches

        Args:
            `n_batches` (`int`): number of dataloader batches

        Returns:
            `dict`: average accumulated metrics among all batches
        """
        for metric_name in self.cumulative_metrics.keys():
            self.cumulative_metrics[metric_name] /= n_batches
        return self.cumulative_metrics
            
    