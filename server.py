import flwr as fl
import torch
import signal
import threading
import sys
from load_quantized import load_model

class TinyBertServer(fl.server.strategy.FedAvg):
    def __init__(self, model, device, num_rounds=5):
        self.model = model
        self.device = device
        self.num_rounds = num_rounds
        initial_parameters = fl.common.ndarrays_to_parameters(self.get_initial_parameters())
        super().__init__(
            min_fit_clients=2,  # Set minimum number of clients to fit on
            min_available_clients=2,  # Minimum number of clients needed to start training
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=self.evaluate_metrics_aggregation_fn
        )

    def get_initial_parameters(self):
        """Get initial parameters from the pre-trained global model."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit_metrics_aggregation_fn(self, metrics):
        """Aggregate fit metrics across clients."""
        aggregated_metrics = {}
        for num_samples, metric_dict in metrics:
            for key, value in metric_dict.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0.0
                aggregated_metrics[key] += value * num_samples
        total_samples = sum(num_samples for num_samples, _ in metrics)
        for key in aggregated_metrics:
            aggregated_metrics[key] /= total_samples
        return aggregated_metrics

    def evaluate_metrics_aggregation_fn(self, metrics):
        if not metrics:
            print("No evaluation metrics received from clients.")
            return {}
        aggregated_metrics = {}
        try:
            for metric_key in metrics[0][1].keys():
                aggregated_metrics[metric_key] = sum(
                    m[1][metric_key] * m[0] for m in metrics
                ) / sum(m[0] for m in metrics)
        except KeyError:
            print("Error in aggregating evaluation metrics: keys mismatch.")
        except IndexError:
            print("Error: metrics list is empty or does not contain valid elements.")
        return aggregated_metrics

    def get_parameters(self):
        """Get current model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters."""
        keys = list(self.model.state_dict().keys())
        state_dict = {keys[i]: torch.tensor(parameters[i]) for i in range(len(keys))}
        self.model.load_state_dict(state_dict, strict=True)

def run_server_thread(strategy, server_address="0.0.0.0:8080"):
    """Function to start the server in a separate thread."""
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path =  "D:\\fed_up\\Quantized"

    model, _ = load_model(model_path=model_path)
    model.to(device)

    server_strategy = TinyBertServer(model=model, device=device, num_rounds=1)
    server_thread = threading.Thread(target=run_server_thread, args=(server_strategy,))
    server_thread.start()

    try:
        while server_thread.is_alive():
            server_thread.join(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, shutting down server...")
        sys.exit(0)
