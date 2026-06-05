import pandas as pd
from src.preprocess import preprocess_traffic_data
# Assuming you wrapped your training logic in a function called 'train_and_predict'
from src.train import train_and_predict 

def main():
    print("Initializing traffic-demand-predictor pipeline...")
    
    # 1. Load the raw data
    # Path assumes you are running main.py from the root 'traffic-demand-predictor' folder
    print("Loading datasets from data/raw/...")
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    
    # 2. Preprocess
    print("Running feature engineering...")
    train_processed = preprocess_traffic_data(train_df)
    test_processed = preprocess_traffic_data(test_df)
    
    # 3. Train and Predict
    print("Passing data to CatBoost model...")
    # This should return your final formatted pandas DataFrame
    submission = train_and_predict(train_processed, test_processed)
    
    # 4. Save Output
    output_path = 'submissions/final_submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"Success! Submission file saved to: {output_path}")

if __name__ == "__main__":
    main()