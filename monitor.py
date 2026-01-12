from prometheus_client import start_http_server, Gauge, Counter, Histogram
import config

# Define Global Metrics
PREDICTION_GAUGE = Gauge('fraud_prediction_probability', 'Model confidence score (0.0 - 1.0)')
REQUEST_COUNTER = Counter('fraud_requests_total', 'Total requests processed', ['result'])
LATENCY_HISTOGRAM = Histogram('fraud_processing_seconds', 'Time spent processing request')

def start_monitor_server():
    """Start the Prometheus metrics HTTP server."""
    print(f"📡 [Monitor] Metrics Server started at http://localhost:{config.MONITOR_PORT}")
    start_http_server(config.MONITOR_PORT)

def update_metrics(scam_prob, prediction_label, process_time):
    """
    Update all metrics after a prediction is made.
    """
    # Update Gauge (Current Probability)
    PREDICTION_GAUGE.set(scam_prob)
    
    # Update Counter (Total Requests by Type)
    REQUEST_COUNTER.labels(result=prediction_label).inc()
    
    # Update Histogram (Latency Distribution)
    LATENCY_HISTOGRAM.observe(process_time)