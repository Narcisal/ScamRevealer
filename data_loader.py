import json
import sys
import pandas as pd

def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ [Error] File '{filepath}' not found.")
        sys.exit(1)

def prepare_training_data(filepath):
    data = load_json_file(filepath)
    df = pd.DataFrame(data)
    
    # Label Encoding: scam -> 1, ham -> 0
    if 'label' in df.columns:
        df['label_id'] = df['label'].apply(lambda x: 1 if x == 'scam' else 0)
    
    return df