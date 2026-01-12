import wandb
import config

def start_monitor_server():
    """
    Initialize Weights & Biases project.
    This creates a dashboard in the cloud automatically.
    """
    # 'project' is your dashboard name on wandb.ai
    # 'config' stores hyperparameters for version control
    wandb.init(
        project="fraud-guard-demo", 
        name="random-forest-run",
        config={
            "model_type": "RandomForest",
            "n_estimators": config.RF_ESTIMATORS,
            "random_state": config.RANDOM_STATE
        }
    )
    print("📡 [Monitor] W&B Connected! View live dashboard at: https://wandb.ai/home")

def update_metrics(scam_prob, prediction_label, process_time, text_sample):
    """
    Log metrics to W&B cloud.
    """
    # wandb.log takes a dictionary of metrics
    wandb.log({
        "fraud_probability": scam_prob,        # Line chart
        "prediction": prediction_label,        # Text log
        "latency_seconds": process_time,       # Histogram
        "input_text": text_sample,             # Input text content
        "is_scam": 1 if prediction_label == "SCAM" else 0  # Boolean metric
    })