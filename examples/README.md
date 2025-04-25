# AgroForGHG Examples

This directory contains example scripts and sample data to demonstrate how to use the AgroForGHG machine learning models.

## Sample Data

The `sample_data.csv` file in the `data` directory contains synthetic data with the following features:

- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `temperature`: Average temperature (°C)
- `precipitation`: Annual precipitation (mm)
- `soil_ph`: Soil pH value
- `soil_organic_matter`: Soil organic matter content (%)
- `elevation`: Elevation above sea level (m)
- `slope`: Terrain slope (degrees)
- `current_land_use`: Current land use type (cropland, grassland, forest, etc.)
- `vegetation_cover`: Vegetation cover fraction (0-1)
- `population_density`: Population density (people/km²)
- `distance_to_roads`: Distance to nearest road (km)
- `is_suitable`: Binary target for land suitability (0/1)
- `ghg_reduction`: Continuous target for GHG reduction potential (tons CO2e/ha/year)

## Running the Example

To run the example with sample data:

```bash
cd examples
python run_sample_data.py
```

This will:
1. Load the sample data
2. Preprocess and engineer features
3. Train both the land suitability classifier and GHG reduction regressor
4. Evaluate model performance
5. Generate visualizations
6. Save trained models and results to the `output/sample_run` directory

## Output

The example script generates:
- Feature distribution plots
- Correlation matrix
- Model evaluation metrics
- Confusion matrix for the classifier
- Actual vs predicted plots for the regressor
- Feature importance charts
- Trained model files

All outputs are saved in the `output/sample_run` directory.
