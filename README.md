🚦 Urban Traffic Demand Predictor
Gridlock Hackathon - Top Tier Submission
(Final Public Leaderboard Score: 91.38)

📝 Overview
This repository contains a high-performance, machine learning pipeline designed to forecast urban traffic demand. Built under a strict 24-hour hackathon time constraint, the solution completely avoids data-leak exploits, relying instead on rigorous spatial-temporal feature engineering and a robust multi-model ensemble strategy.

🧠 Methodology
The core of this pipeline is the "Spatial Frankenstein" approach combined with a mathematically stabilized ensemble:

Spatial-Temporal Feature Engineering:

Targeted Temporal Flags: Engineered an is_rush_hour boolean to explicitly feed traffic cycle knowledge to the boosting trees.

Hybrid Spatial Mapping: Extracted raw latitude and longitude coordinates from geohash strings to allow for geographical interpolation, while intentionally preserving the raw geohash as a categorical feature to leverage target-statistic memory in CatBoost.

Cross-Feature Interactions: Created bottleneck metrics like lane_rush_impact (is_rush_hour * NumberofLanes).

The Ensemble Architecture:

CatBoost Regressor (50%): Handles high-cardinality categorical variables (like geohashes and landmarks) natively without extreme memory overhead.

LightGBM Regressor (50%): Utilizes asymmetric tree building to capture nuanced, residual noise.

Averaging these models cancels out individual variance and significantly boosts out-of-sample generalization.

📂 Repository Structure
documentation/

Technical_Report.pdf: Comprehensive breakdown of architecture and engineering

src/

preprocess.py: Feature engineering, spatial decoding, and cleaning

train.py: CatBoost/LightGBM ensemble training and prediction logic

main.py: Main execution script

requirements.txt: Environment dependencies

README.md: Project documentation

⚙️ Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

pip install -r requirements.txt

Core Libraries:

pandas

catboost

lightgbm

pygeohash

scikit-learn

🚀 How to Run
Place your training and testing datasets in the root directory.

Execute the main pipeline script:
python main.py

The script will automatically trigger the preprocessing logic, train both the CatBoost and LightGBM models, calculate the ensemble average, and output the final predictions to submissions/final_submission.csv.

📊 Results
Ensemble Strategy: 50/50 CatBoost + LightGBM

Optimization: High-iteration (6000/4000), low-learning-rate (0.01) strategy to force microscopic learning steps and prevent overfitting.

Accuracy Score: ~91.38 (Public Leaderboard)