import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

def clean_numeric_column(series):
    """Clean numeric columns that have periods instead of decimal points"""
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.replace(',', '')
        parts = series.str.split('.')
        series = parts.str[0] + '.' + parts.str[1].fillna('0')
        series = pd.to_numeric(series, errors='coerce')
    return series

def load_and_clean_data(filepath):
    """Load and clean the apartment dataset"""
    print("Loading and cleaning data...")
    df = pd.read_csv(filepath)
    
    numeric_cols = [
        'Tract Median Apartment Contract Rent per Square Foot',
        'Tract Median Apartment Contract Rent per Unit',
        'Year over Year Change in Rent per Square Foot',
        'Year over Year Change in Rent per Unit',
        'PROPERTIES',
        'Shape__Area',
        'Shape__Length',
        'Community Reporting Area ID'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    print(f"Original shape: {df.shape}")
    return df

def prepare_features(df, target_col='Tract Median Apartment Contract Rent per Unit'):
    """Prepare features for modeling"""
    print("\nPreparing features...")
    
    df_model = df.copy()
    
    df_model = df_model[df_model[target_col] > 0].copy()
    df_model = df_model.dropna(subset=[target_col])
    
    feature_cols = [
        'Year',
        'Community Reporting Area ID',
        'PROPERTIES',
        'Shape__Area',
        'Shape__Length',
        'Mixed-Rate or Mixed-Income Apartments in Tract'
    ]
    
    if 'Community Reporting Area Name' in df_model.columns:
        le_area = LabelEncoder()
        df_model['Area_Encoded'] = le_area.fit_transform(df_model['Community Reporting Area Name'].fillna('Unknown'))
        feature_cols.append('Area_Encoded')
    
    if 'Cost Category' in df_model.columns:
        le_cost = LabelEncoder()
        df_model['Cost_Category_Encoded'] = le_cost.fit_transform(df_model['Cost Category'].fillna('Unknown'))
        feature_cols.append('Cost_Category_Encoded')
    
    if 'Year over Year Change in Rent Category' in df_model.columns:
        le_change = LabelEncoder()
        df_model['Change_Category_Encoded'] = le_change.fit_transform(df_model['Year over Year Change in Rent Category'].fillna('Unknown'))
        feature_cols.append('Change_Category_Encoded')
    
    df_model['Properties_Per_Area'] = df_model['PROPERTIES'] / (df_model['Shape__Area'] + 1)
    feature_cols.append('Properties_Per_Area')
    
    df_model['Year_Since_2000'] = df_model['Year'] - 2000
    feature_cols.append('Year_Since_2000')
    
    df_model = df_model.dropna(subset=feature_cols + [target_col])
    
    X = df_model[feature_cols]
    y = df_model[target_col]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Features used: {feature_cols}")
    print(f"Target: {target_col}")
    
    return X, y, feature_cols, df_model

def train_xgboost_model(X, y, tune_hyperparameters=True):
    """Train XGBoost model with optional hyperparameter tuning"""
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    if tune_hyperparameters:
        print("\nTuning hyperparameters with GridSearchCV...")
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        print("\nTraining XGBoost model with default parameters...")
        model = xgb.XGBRegressor(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\nTraining Set:")
    print(f"  MAE:  ${train_mae:.2f}")
    print(f"  RMSE: ${train_rmse:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Set:")
    print(f"  MAE:  ${test_mae:.2f}")
    print(f"  RMSE: ${test_rmse:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    print("="*60)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f"\n5-Fold Cross-Validation MAE: ${-cv_scores.mean():.2f} (+/- ${cv_scores.std():.2f})")
    
    return model, X_train, X_test, y_train, y_test, test_pred

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Top Feature Importances in XGBoost Model', fontweight='bold', fontsize=14)
    plt.xlabel('Importance Score', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nSaved: visualizations/feature_importance.png")
    plt.show()

def plot_predictions(y_test, y_pred):
    """Plot actual vs predicted values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Rent per Unit ($)', fontweight='bold')
    axes[0].set_ylabel('Predicted Rent per Unit ($)', fontweight='bold')
    axes[0].set_title('Actual vs Predicted Rent', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Rent per Unit ($)', fontweight='bold')
    axes[1].set_ylabel('Residuals ($)', fontweight='bold')
    axes[1].set_title('Residual Plot', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/prediction_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/prediction_analysis.png")
    plt.show()

def save_model(model, feature_names, filepath='aprice/xgboost_rent_model.pkl'):
    """Save the trained model and feature names"""
    print(f"\nSaving model to {filepath}...")
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved successfully!")

def predict_rent(model, feature_names, input_data):
    """Make predictions with the trained model"""
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(input_df)[0]
    return prediction

def main():
    input_file = 'Apartment_Market_Prices new.csv'
    
    df = load_and_clean_data(input_file)
    
    X, y, feature_names, df_model = prepare_features(
        df, 
        target_col='Tract Median Apartment Contract Rent per Unit'
    )
    
    model, X_train, X_test, y_train, y_test, y_pred = train_xgboost_model(
        X, y, 
        tune_hyperparameters=False
    )
    
    plot_feature_importance(model, feature_names)
    
    plot_predictions(y_test, y_pred)
    
    save_model(model, feature_names, 'aprice/xgboost_rent_model.pkl')
    
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    sample_idx = X_test.index[0]
    sample_features = X_test.iloc[0].values
    actual_rent = y_test.iloc[0]
    predicted_rent = model.predict([sample_features])[0]
    
    print(f"\nSample features:")
    for fname, fval in zip(feature_names, sample_features):
        print(f"  {fname}: {fval}")
    print(f"\nActual Rent: ${actual_rent:.2f}")
    print(f"Predicted Rent: ${predicted_rent:.2f}")
    print(f"Difference: ${abs(actual_rent - predicted_rent):.2f}")
    print("="*60)
    
    print("\n✓ Model training complete!")
    print("✓ Model saved to: aprice/xgboost_rent_model.pkl")
    print("✓ Visualizations saved to: visualizations/")

if __name__ == "__main__":
    main()
