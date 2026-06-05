import pandas as pd
import pygeohash as pgh

def safe_decode(gh):
    """Safely decodes a geohash, returning (0.0, 0.0) if missing or corrupted."""
    try:
        return pgh.decode(str(gh))
    except:
        return (0.0, 0.0)

def preprocess_traffic_data(df):
    data = df.copy()
    
    # 1. Temporal & Rush Hour
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    data = data.drop(columns=['timestamp'])
    
    # 2. The TRUE Spatial Frankenstein
    print("Extracting coordinates while keeping native geohash...")
    coords = data['geohash'].apply(safe_decode)
    data['latitude'] = [c[0] for c in coords]
    data['longitude'] = [c[1] for c in coords]
    # We are explicitly keeping 'geohash' for CatBoost's categorical engine!
    
    # 3. Cross-Feature Interactions
    if 'NumberofLanes' in data.columns:
        data['lane_rush_impact'] = data['is_rush_hour'] * data['NumberofLanes']
        
    # 4. Categorical Handling
    cat_cols = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].fillna('Missing').astype(str)
            
    return data