# main.py
import time
import config
import data_loader
import monitor
from model_engine import FraudModel

def run_batch_inference():
    print("🚀 FraudGuard AI Agent - Initializing System...")
    
    # 1. Initialize Monitoring (W&B) & AI Engine
    monitor.start_monitor_server()
    model = FraudModel()

    # 2. Check for existing model, otherwise retrain
    if not model.load():
        print("⚠️ [System] No saved model found. Initiating training sequence...")
        df_train = data_loader.prepare_training_data(config.TRAIN_DATA_FILE)
        print(f"📊 [System] Loaded {len(df_train)} training samples.")
        
        model.train(df_train)
        model.save()

    # 3. Load Test Data (Batch Inference)
    print(f"\n📂 [System] Reading target messages from: {config.TARGET_DATA_FILE}")
    target_msgs = data_loader.load_json_file(config.TARGET_DATA_FILE)
    print(f"📊 [System] Found {len(target_msgs)} messages. Starting analysis...\n")
    time.sleep(1)

    # 4. Main Inference Loop
    for msg in target_msgs:
        msg_id = msg['id']
        text = msg['text']
        start_time = time.time()

        # AI Inference
        is_scam, scam_prob = model.predict(text)
        
        # Calculate Latency
        process_time = time.time() - start_time
        prediction_label = "SCAM" if is_scam else "SAFE"

        # Update W&B Metrics (Pass 'text' as well!)
        monitor.update_metrics(scam_prob, prediction_label, process_time, text)

        # Display Result to Console
        color = "\033[91m" if is_scam else "\033[92m" # Red for Scam, Green for Safe
        reset = "\033[0m"
        
        print(f"📥 ID: {msg_id} | Text: {text[:30]}...")
        print(f"    🌲 RF Confidence: {scam_prob:.4f}")
        print(f"    📊 Verdict: {color}{prediction_label}{reset}")
        print("-" * 60)
        
        # Slight delay to visualize changes in W&B
        time.sleep(1.0)

    print("\n✅ Batch Analysis Finished.")
    print("📡 W&B Syncing complete.")
    print("👉 Go to https://wandb.ai/home to see your dashboard!")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    run_batch_inference()