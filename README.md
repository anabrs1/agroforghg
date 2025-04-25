# AgroForGHG: Agroforestry Land Suitability and GHG Reduction Prediction

This repository contains machine learning models for predicting land areas suitable for conversion to agroforestry and estimating the potential greenhouse gas (GHG) emission reductions from such conversions.

## Models

1. **Land Conversion Classifier (Random Forest)**: Predicts if a given land area is suitable for agroforestry.
2. **GHG Reduction Regressor (Random Forest)**: Estimates the carbon sequestration potential (GHG emission reductions) of converting a given land area to agroforestry.

## Project Structure

```
agroforghg/
├── main.py                  # Main script to run the pipeline
├── README.md                # This file
└── src/                     # Source code
    ├── data/                # Data loading and preprocessing
    │   └── data_loader.py   # Data loading and preprocessing utilities
    ├── evaluation/          # Model evaluation
    │   └── model_evaluation.py  # Evaluation metrics and visualization
    ├── models/              # ML models
    │   ├── ghg_reduction_regressor.py  # GHG reduction regressor
    │   └── land_suitability_classifier.py  # Land suitability classifier
    └── utils/               # Utilities
        ├── feature_engineering.py  # Feature engineering utilities
        └── visualization.py  # Data visualization utilities
```

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

Optional dependencies for spatial data:
- geopandas
- folium

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/anabrs1/agroforghg.git
cd agroforghg
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

```bash
python main.py --data_path /path/to/your/data.csv \
               --output_dir output \
               --suitability_target is_suitable \
               --ghg_target ghg_reduction \
               --n_estimators 100 \
               --max_depth 10
```

### Command Line Arguments

- `--data_path`: Path to the input data file (required)
- `--output_dir`: Directory to save outputs (default: 'output')
- `--suitability_target`: Column name for land suitability target (required)
- `--ghg_target`: Column name for GHG reduction target (required)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--val_size`: Proportion of data to use for validation (default: 0.1)
- `--random_state`: Random seed for reproducibility (default: 42)
- `--n_estimators`: Number of trees in the random forest (default: 100)
- `--max_depth`: Maximum depth of trees in the random forest (default: None)

## Input Data Format

The input data should be a CSV, Excel, JSON, GeoJSON, or shapefile containing the following:

1. Features for predicting land suitability and GHG reduction
2. A target column for land suitability (binary: 0/1 or True/False)
3. A target column for GHG reduction (continuous value)

Example features might include:

- Soil properties (pH, texture, organic matter)
- Climate data (temperature, precipitation, aridity)
- Topographic information (slope, elevation, aspect)
- Current land use and vegetation
- Socioeconomic factors

## Output

The pipeline generates the following outputs:

1. Trained models saved in the output directory
2. Evaluation metrics for both models
3. Visualizations of model performance and feature importance
4. Predictions on test data

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
