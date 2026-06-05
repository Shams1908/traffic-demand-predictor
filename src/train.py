import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

def train_and_predict(train_processed, test_processed):
    X_train = train_processed.drop(columns=['demand', 'Index'])
    y_train = train_processed['demand']
    X_test = test_processed.drop(columns=['Index'])
    test_indices = test_processed['Index']

    categorical_features = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']

    for col in categorical_features:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # --- CATBOOST ---
    print("Initiating CatBoost training...")
    cat_model = CatBoostRegressor(
        iterations=6000,       
        learning_rate=0.01,    
        depth=8,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=1000  # Less console spam
    )
    cat_model.fit(X_train, y_train, cat_features=categorical_features)
    cat_preds = cat_model.predict(X_test)

    # --- LIGHTGBM ---
    print("Initiating LightGBM training...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=4000,     # Increased to match CatBoost precision
        learning_rate=0.01,    # Decreased to force deep learning
        max_depth=8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)

    # --- ENSEMBLE ---
    print("Ensembling predictions...")
    final_preds = (cat_preds + lgb_preds) / 2.0

    submission = pd.DataFrame({
        'Index': test_indices,
        'demand': final_preds
    })

    return submission