import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def clean_numeric(x):
    """Clean numeric values by removing commas and converting to float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # Remove commas and spaces
        x = x.strip().replace(',', '')
        # Handle empty strings and dashes
        if x in ['', '-', ' -   ', '  -  ']:
            return np.nan
        try:
            return float(x)
        except ValueError:
            return np.nan
    return np.nan

def load_data(file_path):
    """Load the dataset and perform initial analysis."""
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns
    feature_cols = [col for col in df.columns if col.startswith('Feature')]
    target_cols = [col for col in df.columns if col.startswith('Target')]
    
    for col in feature_cols + target_cols:
        df[col] = df[col].apply(clean_numeric)
    
    print(f"Initial shape: {df.shape}")
    print("\nMissing values:\n", df.isnull().sum())
    
    return df

def handle_missing_features(df, n_neighbors=5):
    """Handle missing values in features only using KNN Imputer."""
    # Separate features from targets and categorical columns
    features = df.filter(regex='Feature\d+')
    
    # Apply KNN imputation to features only
    imputer = KNNImputer(n_neighbors=n_neighbors)
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    # Replace features in original dataframe
    for col in features.columns:
        df[col] = features_imputed[col]
    
    return df

def handle_outliers(df, method='iqr'):
    """Handle outliers using IQR method with winsorization."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in ['Year']:  # Skip year column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Winsorization
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def normalize_features(df):
    """Normalize features using StandardScaler."""
    # Separate features
    features = df.filter(regex='Feature\d+')
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    # Replace features in original dataframe
    for col in features.columns:
        df[col] = features_normalized[col]
    
    return df

def create_target_specific_datasets(df):
    """Create three datasets, each filtered for non-null values of their respective targets."""
    # Create copies for each target
    dataset_1 = df[~df['Target 1'].isna()].copy()
    dataset_2 = df[~df['Target 2'].isna()].copy()
    dataset_3 = df[~df['Target 3'].isna()].copy()
    
    print("\nDataset shapes after filtering for each target:")
    print(f"Dataset 1 (Target 1): {dataset_1.shape}")
    print(f"Dataset 2 (Target 2): {dataset_2.shape}")
    print(f"Dataset 3 (Target 3): {dataset_3.shape}")
    
    return dataset_1, dataset_2, dataset_3

def analyze_data(df, stage, target_num=None):
    """Print analysis at different stages."""
    target_str = f" for Target {target_num}" if target_num else ""
    print(f"\n=== Analysis at {stage}{target_str} ===")
    print(f"Shape: {df.shape}")
    print("\nSummary statistics for numeric columns:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    if stage == "Final Stage":
        target_col = f"Target {target_num}" if target_num else None
        if target_col:
            print(f"\nTarget {target_num} statistics:")
            print(df[target_col].describe())
            
            # Print correlation between features and target
            features = df.filter(regex='Feature\d+')
            print(f"\nTop 5 features correlated with Target {target_num}:")
            correlations = features.corrwith(df[target_col]).sort_values(ascending=False)
            print(correlations.head())

def main():
    # Load data
    df = load_data('FidelFolio_Dataset.csv')
    analyze_data(df, "Initial Stage")
    
    # Handle missing values in features
    print("\nHandling missing values in features...")
    df_clean = handle_missing_features(df)
    analyze_data(df_clean, "After Missing Values Treatment")
    
    # Create target-specific datasets
    print("\nCreating target-specific datasets...")
    dataset_1, dataset_2, dataset_3 = create_target_specific_datasets(df_clean)
    
    # Process each dataset separately
    datasets = [(dataset_1, 1), (dataset_2, 2), (dataset_3, 3)]
    
    for dataset, target_num in datasets:
        print(f"\nProcessing dataset for Target {target_num}...")
        
        # Handle outliers
        dataset = handle_outliers(dataset)
        analyze_data(dataset, "After Outlier Treatment", target_num)
        
        # Normalize features
        dataset = normalize_features(dataset)
        analyze_data(dataset, "Final Stage", target_num)
        
        # Save processed data
        output_file = f'processed_data_target_{target_num}.csv'
        dataset.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to '{output_file}'")

if __name__ == "__main__":
    main() 