import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, MultiHeadAttention, LayerNormalization, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import shap  # For SHAP values calculation
from sklearn.inspection import permutation_importance  # For feature importance

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories for saving models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('interpretability', exist_ok=True)  # For interpretability results

# ================= MODEL INTERPRETABILITY FUNCTIONS =================
def analyze_feature_importance_inline(model_type, target_name, X_test, y_test, feature_names, model=None):
    """Analyze and visualize feature importance directly without loading saved models"""
    print(f"\nAnalyzing feature importance for {model_type} on {target_name}...")
    
    if model is None:
        print(f"No model provided for {model_type} on {target_name}, skipping analysis")
        return None
    
    # Create baseline prediction
    baseline_pred = model.predict(X_test, verbose=0).flatten()
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    
    # Calculate feature importance via permutation
    feature_importances = []
    
    # For sequence data, analyze each feature
    seq_length, n_features = X_test.shape[1], X_test.shape[2]
    
    for feat_idx in range(n_features):
        # Copy the test data
        X_permuted = X_test.copy()
        
        # Permute this feature across all time steps
        for t in range(seq_length):
            X_permuted[:, t, feat_idx] = np.random.permutation(X_permuted[:, t, feat_idx])
        
        # Make predictions with permuted feature
        perm_pred = model.predict(X_permuted, verbose=0).flatten()
        perm_mse = mean_squared_error(y_test, perm_pred)
        
        # Importance is the increase in error when feature is permuted
        importance = perm_mse - baseline_mse
        feature_importances.append(importance)
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Save to CSV
    importance_df.to_csv(f"interpretability/{model_type}_{target_name.replace(' ', '_')}_importance.csv", index=False)
    
    # Visualize top features
    plt.figure(figsize=(12, 6))
    top_features = importance_df.head(10)
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title(f"Top 10 Important Features for {target_name} ({model_type.upper()})")
    plt.xlabel('Permutation Importance (higher = more important)')
    plt.tight_layout()
    plt.savefig(f"interpretability/{model_type}_{target_name.replace(' ', '_')}_importance.png")
    plt.close()
    
    return importance_df

def visualize_prediction_confidence_inline(model_type, target_name, X_test, y_test, model=None):
    """Visualize model predictions, errors, and confidence directly without loading saved models"""
    print(f"\nVisualizing prediction confidence for {model_type} on {target_name}...")
    
    if model is None:
        print(f"No model provided for {model_type} on {target_name}, skipping visualization")
        return
    
    # Generate predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate absolute error for each prediction
    abs_errors = np.abs(y_test - y_pred)
    
    # Create dataframe for visualization
    pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'AbsError': abs_errors
    })
    
    # Create scatter plot with error as color
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pred_df['Actual'], pred_df['Predicted'], 
                         c=pred_df['AbsError'], cmap='viridis', alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
    max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.colorbar(scatter, label='Absolute Error')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Prediction Confidence - {model_type.upper()} for {target_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"interpretability/{model_type}_{target_name.replace(' ', '_')}_prediction_confidence.png")
    plt.close()
    
    # Create error distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(pred_df['AbsError'], kde=True)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution - {model_type.upper()} for {target_name}')
    plt.tight_layout()
    plt.savefig(f"interpretability/{model_type}_{target_name.replace(' ', '_')}_error_distribution.png")
    plt.close()

def compare_importances(importance_data, model_type):
    """Compare feature importance across different targets"""
    # If we have data for all targets, create comparison
    if len(importance_data) >= 2:  # Need at least 2 targets to compare
        # Get top 5 features from each target
        all_top_features = []
        for target, imp_df in importance_data.items():
            all_top_features.extend(imp_df.head(5)['Feature'].tolist())
        
        # Create unique list
        all_top_features = list(set(all_top_features))
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({'Feature': all_top_features})
        
        # Add importance for each target
        for target, imp_df in importance_data.items():
            # Get importance values for these features
            target_df = imp_df[imp_df['Feature'].isin(all_top_features)]
            # Create a mapping from feature to importance
            importance_map = dict(zip(target_df['Feature'], target_df['Importance']))
            
            # Add to comparison df with the target name as column
            comparison_df[target] = comparison_df['Feature'].map(importance_map).fillna(0)
        
        # Save to CSV
        comparison_df.to_csv(f"interpretability/{model_type}_feature_importance_comparison.csv", index=False)
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 8))
        heatmap_df = comparison_df.set_index('Feature')
        sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'Feature Importance Comparison Across Targets - {model_type.upper()}')
        plt.tight_layout()
        plt.savefig(f"interpretability/{model_type}_feature_importance_heatmap.png")
        plt.close()

# Load the target-specific datasets
def load_target_dataset(target_num):
    """Load dataset for specific target"""
    filename = f'processed_data_target_{target_num}.csv'
    data = pd.read_csv(filename)
    print(f"Dataset {target_num} loaded with shape: {data.shape}")
    return data

# Define feature and target columns
feature_columns = [f'Feature{i}' for i in range(1, 29)]
target_columns = ['Target 1', 'Target 2', 'Target 3']

# Prepare sequences for time-series modeling
def prepare_sequences(df, feature_cols, target_col, seq_length=3):
    """Prepare time series sequences for each company"""
    companies = df['Company'].unique()
    X_sequences = []
    y_values = []
    companies_included = []
    years_included = []
    
    for company in companies:
        company_data = df[df['Company'] == company]
        if len(company_data) >= seq_length + 1:  # +1 because we need at least one target
            company_X = company_data[feature_cols].values
            company_years = company_data['Year'].values
            
            # Create sequences
            for i in range(len(company_data) - seq_length):
                X_sequences.append(company_X[i:i+seq_length])
                y_values.append(company_data[target_col].values[i+seq_length])
                companies_included.append(company)
                years_included.append(company_years[i+seq_length])
    
    X_sequences = np.array(X_sequences)
    y_values = np.array(y_values)
    
    # Create tracking DataFrame
    tracking_df = pd.DataFrame({
        'Company': companies_included,
        'Year': years_included
    })
    
    print(f"Created {len(X_sequences)} sequences with shape: {X_sequences.shape}")
    return X_sequences, y_values, tracking_df

def strict_time_cv_split(X, y, tracking_df, n_splits=5):
    """Create time-based cross-validation splits"""
    unique_years = sorted(tracking_df['Year'].unique())
    years_per_split = len(unique_years) // n_splits
    
    splits = []
    for i in range(n_splits - 1):
        split_year = unique_years[min((i+1) * years_per_split, len(unique_years)-1)]
        train_mask = tracking_df['Year'] < split_year
        test_mask = tracking_df['Year'] == split_year
        
        # Only include split if we have both train and test data
        if train_mask.any() and test_mask.any():
            splits.append((
                np.where(train_mask)[0],
                np.where(test_mask)[0]
            ))
    
    return splits

def build_mlp_model(input_shape, target_name):
    """Build MLP model"""
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_lstm_model(input_shape, target_name):
    """Build LSTM model"""
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_transformer_model(input_shape, target_name):
    """Build Transformer model"""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Multi-head attention layer
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=input_shape[-1]//4
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(attention_output + x)
    
    # Feed-forward network
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(input_shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_and_evaluate_cv(model_type, cv_splits, input_shape, target_name, target_num, X, y):
    """Train and evaluate model using cross-validation"""
    print(f"\nTraining {model_type.upper()} for {target_name}...")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Store results for each fold
    all_test_scores = []
    all_predictions = []
    all_actuals = []
    all_test_indices = []
    best_model = None
    best_score = float('inf')
    
    # Train and evaluate on each fold
    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        print(f"\nFold {fold}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Build model
        if model_type == 'mlp':
            model = build_mlp_model(input_shape, target_name)
        elif model_type == 'lstm':
            model = build_lstm_model(input_shape, target_name)
        else:  # transformer
            model = build_transformer_model(input_shape, target_name)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_score = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test MSE: {test_score:.4f}")
        
        # Make predictions
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Store results
        all_test_scores.append(test_score)
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        all_test_indices.extend(test_idx)
        
        # Update best model
        if test_score < best_score:
            best_score = test_score
            best_model = model
            
            # Save best model
            model_path = f"models/{model_type}_target_{target_num}.h5"
            model.save(model_path)
            print(f"Saved best model to {model_path}")
    
    # Calculate overall metrics
    mse = mean_squared_error(all_actuals, all_predictions)
    rmse = np.sqrt(mse)
    
    print(f"\nOverall Results for {target_name}:")
    print(f"Average MSE: {np.mean(all_test_scores):.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save predictions and create results DataFrame
    results_df = pd.DataFrame({
        'Actual': all_actuals,
        'Predicted': all_predictions,
        'Error': np.array(all_actuals) - np.array(all_predictions),
        'TestIndex': all_test_indices
    })
    results_df.to_csv(f"results/{model_type}_target_{target_num}_predictions.csv", index=False)
    
    # Analyze feature importance
    if best_model is not None:
        analyze_feature_importance_inline(
            model_type, target_name,
            X[all_test_indices], np.array(all_actuals),
            feature_columns, best_model
        )
        
        # Visualize predictions
        visualize_prediction_confidence_inline(
            model_type, target_name,
            X[all_test_indices], np.array(all_actuals),
            best_model
        )
    
    return best_model, results_df

def main():
    # Model types to train
    model_types = ['mlp', 'lstm', 'transformer']
    
    # Store importance data for comparison
    importance_data = {model_type: {} for model_type in model_types}
    
    # Store results for comparison
    results = []
    
    # Process each target separately
    for target_num in range(1, 4):
        target_name = f'Target {target_num}'
        print(f"\nProcessing {target_name}...")
        
        # Load target-specific dataset
        data = load_target_dataset(target_num)
        
        # Prepare sequences
        X, y, tracking_df = prepare_sequences(data, feature_columns, target_name)
        input_shape = X.shape[1:]
        
        # Create CV splits
        cv_splits = strict_time_cv_split(X, y, tracking_df)
        
        # Train and evaluate each model type
        for model_type in model_types:
            model, results_df = train_and_evaluate_cv(
                model_type, cv_splits, input_shape, target_name, target_num, X, y
            )
            
            # Calculate RMSE from Error column
            rmse = np.sqrt(np.mean(np.square(results_df['Error'])))
            results.append({
                'Model': model_type.upper(),
                'Target': target_name,
                'RMSE': rmse
            })
            
            # Load feature importance if available
            importance_file = f"interpretability/{model_type}_{target_name.replace(' ', '_')}_importance.csv"
            if os.path.exists(importance_file):
                imp_df = pd.read_csv(importance_file)
                importance_data[model_type][target_name] = imp_df
    
    # Create comparison visualization with results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison_results.csv", index=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Target', y='RMSE', hue='Model')
    plt.title("Model Performance Comparison (RMSE)")
    plt.ylabel("RMSE (lower is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png")
    plt.close()
    
    # Compare feature importance across targets for each model type
    print("\nComparing feature importance across targets...")
    for model_type in model_types:
        if importance_data[model_type]:
            compare_importances(importance_data[model_type], model_type)
    
    print("\nModel training and evaluation completed.")
    print("Results saved to:")
    print("- Model comparison plot: plots/model_comparison.png")
    print("- Detailed results: results/model_comparison_results.csv")
    print("- Feature importance comparisons: interpretability/[model_type]_feature_importance_comparison.csv")
    print("- Feature importance plots: interpretability/[model_type]_feature_importance_heatmap.png")

if __name__ == "__main__":
    main() 
