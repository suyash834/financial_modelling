# Financial Market Growth Prediction

## Project Overview
This project implements a sophisticated machine learning system for predicting market capitalization growth of companies using historical financial indicators. The system employs multiple deep learning architectures and provides comprehensive model interpretability analysis.

## Dataset Description

The `FidelFolio_Dataset.csv` contains financial indicators for companies spanning from 2001 to 2023:

### Data Structure
- **Temporal Coverage**: 2001-2023 (23 years)
- **Company Identifiers**: Unique identifiers for each company
- **Features**: 28 financial indicators and ratios capturing:
  - Profitability metrics
  - Liquidity ratios
  - Operational efficiency
  - Market performance indicators
  - Growth metrics
  
### Target Variables
Three distinct prediction horizons:
- **Target 1**: 1-year forward market cap growth (%)
- **Target 2**: 2-year forward market cap growth (%)
- **Target 3**: 3-year forward market cap growth (%)

## Project Structure

```
.
├── FidelFolio_Dataset.csv      # Raw financial dataset
├── preprocess_data_v2.py       # Data preprocessing pipeline
├── main_contents.py            # Model training and evaluation
├── results/                    # Prediction results and metrics
├── plots/                      # Visualization outputs
├── models/                     # Saved model checkpoints
└── interpretability/           # Model interpretation analysis
```

## Technical Implementation

### Data Preprocessing Strategy (`preprocess_data_v2.py`)

1. **Initial Data Cleaning**
   - Numeric value cleaning (removing commas, handling special characters)
   - Column name standardization
   - Data type validation and conversion for features and targets
   - Missing value detection and reporting for each column

2. **Feature Engineering**
   - KNN imputation (n_neighbors=5) for features only, preserving target integrity
   - Outlier handling using IQR method:
     - Q1 - 1.5*IQR to Q3 + 1.5*IQR range
     - Winsorization at 1st and 99th percentiles
   - Feature standardization using StandardScaler (μ=0, σ=1)
   - Separate processing pipelines for each target to maintain data independence

3. **Target-Specific Dataset Creation**
   - Independent datasets for each prediction horizon
   - Strict filtering based on target availability
   - Statistical analysis at each processing stage:
     - Shape monitoring
     - Summary statistics
     - Feature-target correlations
   - Data integrity checks between stages

### Model Architectures (`main_contents.py`)

1. **Multi-Layer Perceptron (MLP)**
   ```python
   model = Sequential([
       Input(shape=(input_dim,)),
       Dense(64, activation='relu'),
       BatchNormalization(),
       Dropout(0.3),
       Dense(32, activation='relu'),
       Dropout(0.2),
       Dense(1, activation='linear')
   ])
   ```
   - Designed for static feature learning
   - Batch normalization for training stability
   - Dropout layers for regularization

2. **LSTM Network**
   ```python
   model = Sequential([
       Input(shape=(sequence_length, features)),
       LSTM(32, return_sequences=True),
       Dropout(0.3),
       LSTM(16),
       Dropout(0.2),
       Dense(1)
   ])
   ```
   - Captures temporal dependencies
   - Dual LSTM layers for hierarchical feature extraction
   - Dropout for preventing overfitting

3. **Transformer**
   ```python
   model = Sequential([
       Input(shape=(sequence_length, features)),
       MultiHeadAttention(heads=4, key_dim=8),
       LayerNormalization(),
       Dense(32),
       Dense(feature_dim),
       LayerNormalization(),
       Dense(1)
   ])
   ```
   - Self-attention mechanism for temporal relationships
   - Multi-head attention for parallel feature processing
   - Layer normalization for training stability

### Training Strategy

1. **Data Organization**
   - Time-based train-test splitting
   - Strict temporal separation to prevent data leakage
   - Sequence creation with 3-year lookback windows

2. **Cross-Validation Approach**
   - Strict time-based cross-validation (5-fold)
   - Forward-chaining methodology:
     - Each fold represents a distinct time period
     - Training on past data, testing on future data
     - No temporal overlap between train and test sets
   - Example split for a dataset spanning 2001-2023:
     - Fold 1: Train [2001-2005], Test [2006]
     - Fold 2: Train [2001-2010], Test [2011]
     - Fold 3: Train [2001-2015], Test [2016]
     - Fold 4: Train [2001-2020], Test [2021]
     - Fold 5: Train [2001-2022], Test [2023]

3. **Training Configuration**
   ```python
   optimizer = Adam(learning_rate=0.001)
   loss = 'mean_squared_error'
   batch_size = 32
   epochs = 100  # with early stopping
   ```

4. **Training Optimizations**
   - Early Stopping:
     ```python
     EarlyStopping(
         monitor='val_loss',
         patience=10,
         restore_best_weights=True
     )
     ```
   - Learning Rate Scheduling:
     ```python
     ReduceLROnPlateau(
         monitor='val_loss',
         factor=0.5,
         patience=5,
         min_lr=1e-6
     )
     ```
   - Gradient Clipping:
     ```python
     clipnorm=1.0  # Prevents exploding gradients
     ```

5. **Model Ensemble Strategy**
   - Individual model training for each target
   - Cross-model performance comparison
   - Weighted averaging for final predictions

### Model Interpretability

1. **Feature Importance Analysis**
   - SHAP (SHapley Additive exPlanations) values
     - Global feature importance
     - Individual prediction explanations
     - Feature interaction analysis
   
   - Permutation Importance
     - Feature ranking by prediction impact
     - Cross-validation stability assessment

2. **Prediction Analysis**
   - Confidence Intervals:
     - Monte Carlo Dropout estimation
     - Uncertainty quantification
   
   - Error Analysis:
     - Distribution visualization
     - Temporal error patterns
     - Target-specific performance metrics

## Installation and Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide

1. Data Preprocessing:
```bash
python preprocess_data_v2.py
```
Generates:
- `processed_data_target_1.csv`
- `processed_data_target_2.csv`
- `processed_data_target_3.csv`

2. Model Training and Evaluation:
```bash
python main_contents.py
```

### Output Structure

1. **Results Directory** (`results/`)
   - Model comparison metrics
   - Prediction results by target
   - Cross-validation statistics

2. **Plots Directory** (`plots/`)
   - Learning curves
   - Error distributions
   - Feature importance visualizations
   - Model comparison charts

3. **Models Directory** (`models/`)
   - Trained model checkpoints
   - Model architecture configs
   - Training history

4. **Interpretability Directory** (`interpretability/`)
   - SHAP analysis results
   - Feature importance rankings
   - Prediction explanations

## Performance Evaluation

### Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Cross-validation Performance

### Validation Strategy
- Strict time-based splits
- Out-of-sample testing
- Rolling window validation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Citation

If you use this code in your research, please cite:
```bibtex
@software{fidelfolio_prediction,
  title = {Financial Market Growth Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/repository}
}
``` 
