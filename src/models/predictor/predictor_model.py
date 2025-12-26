
# """
# src/models/unified_model_training_and_selection.py

# UNIFIED TRAINING PIPELINE: Train ALL models, automatically select BEST for each target

# This script:
# 1. Trains 4 model types for each target (XGBoost, XGBoost-tuned, LightGBM, LightGBM-tuned)
# 2. Uses TIME SERIES CV during hyperparameter tuning
# 3. Evaluates all models on test set with comprehensive validation
# 4. Generates visualizations for model analysis
# 5. Tracks experiments with MLflow
# 6. Registers best models to MLflow Model Registry
# 7. Saves best model as {target}_best.pkl
# 8. Generates comprehensive comparison report

# Usage:
#     # Train all targets with all models
#     python src/models/unified_model_training_and_selection.py
    
#     # Train specific target
#     python src/models/unified_model_training_and_selection.py --target profit_margin
    
#     # Quick mode (skip tuning for faster results)
#     python src/models/unified_model_training_and_selection.py --quick
    
#     # Custom number of tuning trials
#     python src/models/unified_model_training_and_selection.py --trials 20
# """

# import sys
# import json
# import argparse
# from pathlib import Path
# from datetime import datetime
# import warnings
# import joblib
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.base import clone
# import optuna
# from typing import Dict, Tuple, List
# from scipy import stats

# # Visualization imports
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.gridspec as gridspec

# # MLflow imports (optional - gracefully handles if not installed)
# try:
#     import mlflow
#     import mlflow.sklearn
#     import mlflow.xgboost
#     import mlflow.lightgbm
#     MLFLOW_AVAILABLE = True
# except ImportError:
#     MLFLOW_AVAILABLE = False
#     print("WARNING: MLflow not available - experiment tracking disabled")

# warnings.filterwarnings("ignore")

# # Setup paths
# project_root = Path(__file__).resolve().parent.parent.parent.parent  
# sys.path.insert(0, str(project_root))

# from src.utils.split_utils import get_feature_target_split, drop_nan_targets

# print("Imports successful\n")

# # Set plotting style
# sns.set_style("whitegrid")
# plt.rcParams['figure.figsize'] = (12, 6)
# plt.rcParams['font.size'] = 10


# # ============================================
# # Visualization Functions
# # ============================================

# class ModelVisualizer:
#     """Handles all model visualizations"""
    
#     def __init__(self, output_dir: str):
#         self.output_dir = Path(output_dir)
#         self.viz_dir = self.output_dir / "visualizations"
#         self.viz_dir.mkdir(parents=True, exist_ok=True)
    
#     def plot_predictions_vs_actual(self, y_true, y_pred, title: str, filename: str):
#         """Plot predictions vs actual values"""
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
#         # Scatter plot
#         axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
#         min_val = min(y_true.min(), y_pred.min())
#         max_val = max(y_true.max(), y_pred.max())
#         axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
#         axes[0].set_xlabel('Actual Values')
#         axes[0].set_ylabel('Predicted Values')
#         axes[0].set_title(f'{title} - Predictions vs Actual')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
        
#         # Add R2 to plot
#         r2 = r2_score(y_true, y_pred)
#         axes[0].text(0.05, 0.95, f'R2 = {r2:.4f}', 
#                     transform=axes[0].transAxes, 
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
#         # Residual plot
#         residuals = y_true - y_pred
#         axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
#         axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
#         axes[1].set_xlabel('Predicted Values')
#         axes[1].set_ylabel('Residuals')
#         axes[1].set_title(f'{title} - Residual Plot')
#         axes[1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         return str(self.viz_dir / filename)
    
#     def plot_residual_analysis(self, y_true, y_pred, title: str, filename: str):
#         """Comprehensive residual analysis"""
#         residuals = y_true - y_pred
        
#         fig = plt.figure(figsize=(16, 10))
#         gs = gridspec.GridSpec(2, 3, figure=fig)
        
#         # 1. Residuals vs Predicted
#         ax1 = fig.add_subplot(gs[0, 0])
#         ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
#         ax1.axhline(y=0, color='r', linestyle='--', lw=2)
#         ax1.set_xlabel('Predicted Values')
#         ax1.set_ylabel('Residuals')
#         ax1.set_title('Residuals vs Predicted')
#         ax1.grid(True, alpha=0.3)
        
#         # 2. Histogram of residuals
#         ax2 = fig.add_subplot(gs[0, 1])
#         ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
#         ax2.axvline(x=0, color='r', linestyle='--', lw=2)
#         ax2.set_xlabel('Residuals')
#         ax2.set_ylabel('Frequency')
#         ax2.set_title('Distribution of Residuals')
#         ax2.grid(True, alpha=0.3)
        
#         # 3. Q-Q plot
#         ax3 = fig.add_subplot(gs[0, 2])
#         stats.probplot(residuals, dist="norm", plot=ax3)
#         ax3.set_title('Q-Q Plot')
#         ax3.grid(True, alpha=0.3)
        
#         # 4. Residuals vs Order (time series check)
#         ax4 = fig.add_subplot(gs[1, 0])
#         ax4.plot(residuals, marker='o', linestyle='', alpha=0.6)
#         ax4.axhline(y=0, color='r', linestyle='--', lw=2)
#         ax4.set_xlabel('Observation Order')
#         ax4.set_ylabel('Residuals')
#         ax4.set_title('Residuals vs Order')
#         ax4.grid(True, alpha=0.3)
        
#         # 5. Absolute residuals vs Predicted (heteroscedasticity)
#         ax5 = fig.add_subplot(gs[1, 1])
#         ax5.scatter(y_pred, np.abs(residuals), alpha=0.6, edgecolors='k', linewidth=0.5)
#         ax5.set_xlabel('Predicted Values')
#         ax5.set_ylabel('Absolute Residuals')
#         ax5.set_title('Scale-Location Plot')
#         ax5.grid(True, alpha=0.3)
        
#         # 6. Statistics text
#         ax6 = fig.add_subplot(gs[1, 2])
#         ax6.axis('off')
        
#         # Calculate statistics
#         mean_resid = np.mean(residuals)
#         std_resid = np.std(residuals)
#         _, p_value_shapiro = stats.shapiro(residuals[:min(5000, len(residuals))])
        
#         stats_text = f"""
#         Residual Statistics:
        
#         Mean: {mean_resid:.6f}
#         Std Dev: {std_resid:.6f}
#         Min: {np.min(residuals):.4f}
#         Max: {np.max(residuals):.4f}
        
#         Normality Test (Shapiro-Wilk):
#         p-value: {p_value_shapiro:.4f}
#         {'PASS' if p_value_shapiro > 0.05 else 'FAIL'} (alpha=0.05)
        
#         RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}
#         MAE: {mean_absolute_error(y_true, y_pred):.4f}
#         R2: {r2_score(y_true, y_pred):.4f}
#         """
        
#         ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
#                 fontfamily='monospace')
        
#         plt.suptitle(f'{title} - Comprehensive Residual Analysis', fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         return str(self.viz_dir / filename)
    
#     def plot_feature_importance(self, model, feature_names: List[str], title: str, 
#                                 filename: str, top_n: int = 20):
#         """Plot feature importance"""
#         # Get feature importance based on model type
#         if hasattr(model, 'feature_importances_'):
#             importance = model.feature_importances_
#         elif hasattr(model, 'coef_'):
#             importance = np.abs(model.coef_)
#         else:
#             print(f"Cannot extract feature importance from {type(model)}")
#             return None
        
#         # Create dataframe
#         importance_df = pd.DataFrame({
#             'feature': feature_names,
#             'importance': importance
#         }).sort_values('importance', ascending=False).head(top_n)
        
#         # Plot
#         fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
#         bars = ax.barh(range(len(importance_df)), importance_df['importance'])
#         ax.set_yticks(range(len(importance_df)))
#         ax.set_yticklabels(importance_df['feature'])
#         ax.set_xlabel('Importance')
#         ax.set_title(f'{title} - Top {top_n} Features')
#         ax.invert_yaxis()
        
#         # Color bars
#         colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
#         for bar, color in zip(bars, colors):
#             bar.set_color(color)
        
#         plt.tight_layout()
#         plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         return str(self.viz_dir / filename)
    
#     def plot_cv_scores(self, cv_scores: np.ndarray, title: str, filename: str):
#         """Plot cross-validation scores"""
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
#         # Box plot
#         axes[0].boxplot(cv_scores, vert=True)
#         axes[0].set_ylabel('R2 Score')
#         axes[0].set_title(f'{title} - CV Score Distribution')
#         axes[0].grid(True, alpha=0.3)
        
#         # Add statistics
#         mean_score = np.mean(cv_scores)
#         std_score = np.std(cv_scores)
#         axes[0].axhline(y=mean_score, color='r', linestyle='--', 
#                        label=f'Mean: {mean_score:.4f}')
#         axes[0].legend()
        
#         # Bar plot
#         axes[1].bar(range(len(cv_scores)), cv_scores, color='skyblue', edgecolor='black')
#         axes[1].axhline(y=mean_score, color='r', linestyle='--', lw=2)
#         axes[1].set_xlabel('Fold')
#         axes[1].set_ylabel('R2 Score')
#         axes[1].set_title(f'{title} - Scores by Fold')
#         axes[1].grid(True, alpha=0.3)
        
#         # Add text with stats
#         stats_text = f'Mean: {mean_score:.4f}\nStd: {std_score:.4f}'
#         axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
#                     verticalalignment='top', horizontalalignment='right',
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
#         plt.tight_layout()
#         plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         return str(self.viz_dir / filename)
    
#     def plot_model_comparison(self, comparison_data: List[Dict], target: str, filename: str):
#         """Plot comparison of all models"""
#         df = pd.DataFrame(comparison_data)
        
#         fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
#         # 1. Test R2 comparison
#         axes[0].barh(df['model'], df['test_r2'], color='skyblue', edgecolor='black')
#         axes[0].set_xlabel('Test R2')
#         axes[0].set_title('Model Performance (Test R2)')
#         axes[0].grid(True, alpha=0.3, axis='x')
        
#         # 2. Overfitting percentage
#         colors = ['green' if x < 30 else 'orange' for x in df['overfit_pct']]
#         axes[1].barh(df['model'], df['overfit_pct'], color=colors, edgecolor='black')
#         axes[1].axvline(x=30, color='r', linestyle='--', lw=2, label='30% threshold')
#         axes[1].set_xlabel('Overfitting %')
#         axes[1].set_title('Overfitting Analysis')
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3, axis='x')
        
#         # 3. Test RMSE comparison
#         axes[2].barh(df['model'], df['test_rmse'], color='coral', edgecolor='black')
#         axes[2].set_xlabel('Test RMSE')
#         axes[2].set_title('Prediction Error (Test RMSE)')
#         axes[2].grid(True, alpha=0.3, axis='x')
        
#         plt.suptitle(f'{target.upper()} - Model Comparison', fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         return str(self.viz_dir / filename)


# # ============================================
# # MLflow Integration
# # ============================================

# class MLflowTracker:
#     """Handles MLflow experiment tracking and model registry"""
    
#     def __init__(self, experiment_name: str = "financial-forecasting"):
#         self.experiment_name = experiment_name
#         self.enabled = MLFLOW_AVAILABLE
#         self.active_run = None
        
#         if self.enabled:
#             try:
#                 # Set tracking URI (optional - defaults to ./mlruns)
#                 # mlflow.set_tracking_uri("http://localhost:5000")
                
#                 # Create or get experiment
#                 mlflow.set_experiment(experiment_name)
#                 print(f"MLflow experiment set: {experiment_name}\n")
#             except Exception as e:
#                 print(f"MLflow initialization failed: {e}")
#                 self.enabled = False
#         else:
#             print("MLflow tracking disabled (library not available)\n")
    
#     def start_run(self, run_name: str):
#         """Start MLflow run"""
#         if not self.enabled:
#             return None
        
#         try:
#             self.active_run = mlflow.start_run(run_name=run_name)
#             return self.active_run
#         except Exception as e:
#             print(f"  [MLflow] Failed to start run: {e}")
#             return None
    
#     def end_run(self):
#         """End MLflow run"""
#         if not self.enabled or not self.active_run:
#             return
        
#         try:
#             mlflow.end_run()
#             self.active_run = None
#         except Exception as e:
#             print(f"  [MLflow] Failed to end run: {e}")
    
#     def log_experiment(self, target: str, model_name: str, model_obj, 
#                       params: Dict, metrics: Dict, artifacts: Dict = None):
#         """Log complete experiment to MLflow"""
#         if not self.enabled:
#             return None
        
#         run_id = None
        
#         try:
#             # Start run
#             run_name = f"{target}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
#             with mlflow.start_run(run_name=run_name) as run:
#                 run_id = run.info.run_id
                
#                 # Log parameters
#                 mlflow.log_params(params)
                
#                 # Log metrics
#                 mlflow.log_metrics(metrics)
                
#                 # Log model based on type
#                 model_artifact_path = f"{target}_{model_name}_model"
                
#                 if model_obj is not None:
#                     model_type_str = str(type(model_obj).__name__).lower()
                    
#                     if 'xgb' in model_type_str or isinstance(model_obj, xgb.XGBRegressor):
#                         mlflow.xgboost.log_model(model_obj, model_artifact_path)
#                     elif 'booster' in model_type_str and hasattr(model_obj, 'predict'):
#                         # LightGBM Booster object
#                         mlflow.lightgbm.log_model(model_obj, model_artifact_path)
#                     elif 'lgbm' in model_type_str or isinstance(model_obj, lgb.LGBMRegressor):
#                         mlflow.lightgbm.log_model(model_obj, model_artifact_path)
#                     else:
#                         # Fallback to sklearn
#                         mlflow.sklearn.log_model(model_obj, model_artifact_path)
                
#                 # Log artifacts (visualizations)
#                 if artifacts:
#                     for key, path in artifacts.items():
#                         if path and Path(path).exists():
#                             mlflow.log_artifact(path, artifact_path="visualizations")
                
#                 # Add tags for easier filtering
#                 mlflow.set_tags({
#                     "target": target,
#                     "model_type": model_name,
#                     "training_date": datetime.now().strftime('%Y-%m-%d'),
#                     "framework": "xgboost" if "XGBoost" in model_name else "lightgbm"
#                 })
                
#                 print(f"  [MLflow] Logged: {target}/{model_name} (run_id: {run_id[:8]}...)")
            
#             return run_id
            
#         except Exception as e:
#             print(f"  [MLflow] Logging failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
    
#     def register_best_model(self, target: str, model_name: str, run_id: str, 
#                            test_r2: float, deployment_rec: str):
#         """Register best model to MLflow Model Registry"""
#         if not self.enabled or not run_id:
#             return
        
#         try:
#             # Model URI from the run
#             model_artifact_path = f"{target}_{model_name}_model"
#             model_uri = f"runs:/{run_id}/{model_artifact_path}"
            
#             # Registered model name
#             registered_model_name = f"{target}_predictor"
            
#             # Register the model
#             model_version = mlflow.register_model(
#                 model_uri=model_uri,
#                 name=registered_model_name
#             )
            
#             # Add model version tags and description
#             client = mlflow.tracking.MlflowClient()
#             client.update_model_version(
#                 name=registered_model_name,
#                 version=model_version.version,
#                 description=f"Best model for {target} prediction. Model type: {model_name}. Test R2: {test_r2:.4f}"
#             )
            
#             # Set alias based on deployment recommendation
#             if deployment_rec == "production_ready":
#                 client.set_registered_model_alias(registered_model_name, "production", model_version.version)
#             elif deployment_rec == "use_with_caution":
#                 client.set_registered_model_alias(registered_model_name, "staging", model_version.version)
#             else:
#                 client.set_registered_model_alias(registered_model_name, "development", model_version.version)
            
#             print(f"  [MLflow] Registered model: {registered_model_name} (version {model_version.version})")
#             print(f"  [MLflow] Alias: {deployment_rec}")
            
#         except Exception as e:
#             print(f"  [MLflow] Model registration failed: {e}")
#             import traceback
#             traceback.print_exc()


# # ============================================
# # Model Trainers (Base Classes)
# # ============================================

# class BaseModelTrainer:
#     """Base class for all model trainers"""
    
#     def __init__(self, target_name: str):
#         self.target_name = target_name
#         self.target_col = f"target_{target_name}"
#         self.model = None
#         self.feature_names = None
#         self.train_metrics = None
#         self.val_metrics = None
#         self.test_metrics = None
#         self.cv_scores = None
#         self.cv_mean = None
#         self.cv_std = None
#         self.run_id = None
    
#     def load_and_prepare_data(self, splits_dir: str):
#         """Load and prepare data (common for all models)"""
#         splits_path = Path(splits_dir)
        
#         train_df = pd.read_csv(splits_path / "train_data.csv")
#         val_df = pd.read_csv(splits_path / "val_data.csv")
#         test_df = pd.read_csv(splits_path / "test_data.csv")
        
#         # Prepare features
#         X_train, y_train = get_feature_target_split(train_df, self.target_col, encode_categoricals=True)
#         X_val, y_val = get_feature_target_split(val_df, self.target_col, encode_categoricals=True)
#         X_test, y_test = get_feature_target_split(test_df, self.target_col, encode_categoricals=True)
        
#         # Align columns
#         train_cols = set(X_train.columns)
#         for col in train_cols:
#             if col not in X_val.columns:
#                 X_val[col] = 0
#             if col not in X_test.columns:
#                 X_test[col] = 0
        
#         X_val = X_val[X_train.columns]
#         X_test = X_test[X_train.columns]
        
#         # Impute missing values
#         if X_train.isna().sum().sum() > 0:
#             imputer = SimpleImputer(strategy="median")
#             X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
#             X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
#             X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
#         # Drop NaN targets
#         X_train, y_train = drop_nan_targets(X_train, y_train, "Train")
#         X_val, y_val = drop_nan_targets(X_val, y_val, "Val")
#         X_test, y_test = drop_nan_targets(X_test, y_test, "Test")
        
#         self.feature_names = X_train.columns.tolist()
        
#         return X_train, y_train, X_val, y_val, X_test, y_test
    
#     def evaluate_all_splits(self, X_train, y_train, X_val, y_val, X_test, y_test):
#         """Evaluate model on all splits"""
#         results = {}
        
#         for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
#             pred = self.predict(X)
            
#             results[name] = {
#                 "rmse": float(np.sqrt(mean_squared_error(y, pred))),
#                 "mae": float(mean_absolute_error(y, pred)),
#                 "r2": float(r2_score(y, pred))
#             }
        
#         self.train_metrics = results["train"]
#         self.val_metrics = results["val"]
#         self.test_metrics = results["test"]
        
#         return results
    
#     def predict(self, X):
#         """Prediction method (to be implemented by subclasses)"""
#         raise NotImplementedError
    
#     def get_model_info(self):
#         """Get model information for reporting"""
#         return {
#             "model_type": self.__class__.__name__,
#             "train": self.train_metrics,
#             "val": self.val_metrics,
#             "test": self.test_metrics,
#             "n_features": len(self.feature_names) if self.feature_names else 0,
#             "cv_mean": self.cv_mean,
#             "cv_std": self.cv_std
#         }


# class XGBoostTrainer(BaseModelTrainer):
#     """XGBoost baseline trainer"""
    
#     def __init__(self, target_name: str):
#         super().__init__(target_name)
#         self.model_type = "xgboost"
    
#     def train(self, X_train, y_train, X_val, y_val):
#         """Train XGBoost baseline"""
#         params = {
#             "n_estimators": 500,
#             "max_depth": 8,
#             "learning_rate": 0.05,
#             "subsample": 0.8,
#             "colsample_bytree": 0.8,
#             "min_child_weight": 3,
#             "gamma": 0.1,
#             "reg_alpha": 0.1,
#             "reg_lambda": 1.0,
#             "random_state": 42,
#             "n_jobs": -1,
#             "tree_method": "hist",
#             "verbosity": 0,
#         }
        
#         self.model = xgb.XGBRegressor(**params)
#         self.model.set_params(early_stopping_rounds=50)
        
#         self.model.fit(
#             X_train, y_train,
#             eval_set=[(X_val, y_val)],
#             verbose=False
#         )
        
#         # For baseline models, no CV
#         self.cv_mean = None
#         self.cv_std = None
#         self.cv_scores = None
        
#         return self
    
#     def predict(self, X):
#         return self.model.predict(X)


# class XGBoostTunedTrainer(BaseModelTrainer):
#     """XGBoost with Optuna tuning using TIME SERIES CV"""
    
#     def __init__(self, target_name: str):
#         super().__init__(target_name)
#         self.model_type = "xgboost_tuned"
#         self.best_params = None
    
#     def train(self, X_train, y_train, X_val, y_val, n_trials=30):
#         """Train XGBoost with CV-BASED hyperparameter tuning"""
        
#         # Combine train+val for CV-based hyperparameter search
#         X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
#         y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
#         print(f"   Using Time Series CV for hyperparameter tuning...")
        
#         def objective(trial):
#             """Optuna objective with TIME SERIES CV"""
            
#             # Suggest hyperparameters
#             params = {
#                 "max_depth": trial.suggest_int("max_depth", 3, 12),
#                 "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#                 "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
#                 "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
#                 "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#                 "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
#                 "gamma": trial.suggest_float("gamma", 0.0, 1.0),
#                 "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
#                 "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
#                 "random_state": 42,
#                 "n_jobs": -1,
#                 "tree_method": "hist",
#                 "verbosity": 0,
#             }
            
#             # TIME SERIES CROSS-VALIDATION
#             tscv = TimeSeriesSplit(n_splits=3)
#             cv_scores = []
            
#             for train_idx, val_idx in tscv.split(X_trainval):
#                 X_tr = X_trainval.iloc[train_idx]
#                 X_vl = X_trainval.iloc[val_idx]
#                 y_tr = y_trainval.iloc[train_idx]
#                 y_vl = y_trainval.iloc[val_idx]
                
#                 model = xgb.XGBRegressor(**params)
                
#                 try:
#                     model.fit(X_tr, y_tr)
#                     val_pred = model.predict(X_vl)
#                     fold_r2 = r2_score(y_vl, val_pred)
#                     cv_scores.append(fold_r2)
#                 except Exception:
#                     raise optuna.TrialPruned()
            
#             # Return AVERAGE CV score
#             avg_r2 = np.mean(cv_scores)
#             return -avg_r2
        
#         # Run Optuna optimization with CV
#         study = optuna.create_study(
#             direction="minimize",
#             sampler=optuna.samplers.TPESampler(seed=42),
#             pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
#         )
        
#         optuna.logging.set_verbosity(optuna.logging.WARNING)
#         study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
#         self.best_params = study.best_params
#         self.cv_mean = -study.best_value
        
#         # Get CV scores from best trial for reporting
#         tscv = TimeSeriesSplit(n_splits=3)
#         best_cv_scores = []
        
#         for train_idx, val_idx in tscv.split(X_trainval):
#             X_tr = X_trainval.iloc[train_idx]
#             X_vl = X_trainval.iloc[val_idx]
#             y_tr = y_trainval.iloc[train_idx]
#             y_vl = y_trainval.iloc[val_idx]
            
#             model = xgb.XGBRegressor(**self.best_params, random_state=42, 
#                                      n_jobs=-1, tree_method="hist", verbosity=0)
#             model.fit(X_tr, y_tr)
#             val_pred = model.predict(X_vl)
#             best_cv_scores.append(r2_score(y_vl, val_pred))
        
#         self.cv_scores = np.array(best_cv_scores)
#         self.cv_std = np.std(best_cv_scores)
        
#         print(f"   Best CV Score: {self.cv_mean:.4f} (+/- {self.cv_std:.4f}) from {n_trials} trials")
        
#         # Train FINAL model on ALL training data with best params
#         self.model = xgb.XGBRegressor(**self.best_params, random_state=42, 
#                                       n_jobs=-1, tree_method="hist", verbosity=0)
#         self.model.fit(X_trainval, y_trainval)
        
#         return self
    
#     def predict(self, X):
#         return self.model.predict(X)


# class LightGBMTrainer(BaseModelTrainer):
#     """LightGBM baseline trainer"""
    
#     def __init__(self, target_name: str):
#         super().__init__(target_name)
#         self.model_type = "lightgbm"
    
#     def train(self, X_train, y_train, X_val, y_val):
#         """Train LightGBM baseline"""
#         params = {
#             "objective": "regression",
#             "metric": "rmse",
#             "boosting_type": "gbdt",
#             "num_leaves": 31,
#             "max_depth": 8,
#             "learning_rate": 0.05,
#             "subsample": 0.8,
#             "colsample_bytree": 0.8,
#             "min_child_samples": 20,
#             "reg_alpha": 0.1,
#             "reg_lambda": 1.0,
#             "random_state": 42,
#             "verbose": -1,
#         }
        
#         train_data = lgb.Dataset(X_train, label=y_train)
#         val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
#         callbacks = [
#             lgb.early_stopping(stopping_rounds=50),
#             lgb.log_evaluation(period=0)
#         ]
        
#         self.model = lgb.train(
#             params,
#             train_data,
#             valid_sets=[train_data, val_data],
#             valid_names=['train', 'val'],
#             num_boost_round=500,
#             callbacks=callbacks
#         )
        
#         # For baseline models, no CV
#         self.cv_mean = None
#         self.cv_std = None
#         self.cv_scores = None
        
#         return self
    
#     def predict(self, X):
#         return self.model.predict(X, num_iteration=self.model.best_iteration)


# class LightGBMTunedTrainer(BaseModelTrainer):
#     """LightGBM with Optuna tuning using TIME SERIES CV"""
    
#     def __init__(self, target_name: str):
#         super().__init__(target_name)
#         self.model_type = "lightgbm_tuned"
#         self.best_params = None
    
#     def train(self, X_train, y_train, X_val, y_val, n_trials=30):
#         """Train LightGBM with CV-BASED hyperparameter tuning"""
        
#         # Combine train+val for CV-based hyperparameter search
#         X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
#         y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
#         print(f"   Using Time Series CV for hyperparameter tuning...")
        
#         def objective(trial):
#             """Optuna objective with TIME SERIES CV"""
            
#             # Suggest hyperparameters
#             params = {
#                 'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#                 'max_depth': trial.suggest_int('max_depth', 3, 12),
#                 'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
#                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#                 'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
#                 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#                 'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
#                 'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
#                 'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
#                 'random_state': 42,
#                 'n_jobs': -1,
#                 'verbose': -1,
#             }
            
#             # TIME SERIES CROSS-VALIDATION
#             tscv = TimeSeriesSplit(n_splits=3)
#             cv_scores = []
            
#             for train_idx, val_idx in tscv.split(X_trainval):
#                 X_tr = X_trainval.iloc[train_idx]
#                 X_vl = X_trainval.iloc[val_idx]
#                 y_tr = y_trainval.iloc[train_idx]
#                 y_vl = y_trainval.iloc[val_idx]
                
#                 model = lgb.LGBMRegressor(**params)
                
#                 try:
#                     model.fit(X_tr, y_tr)
#                     val_pred = model.predict(X_vl)
#                     fold_r2 = r2_score(y_vl, val_pred)
#                     cv_scores.append(fold_r2)
#                 except Exception:
#                     raise optuna.TrialPruned()
            
#             # Return AVERAGE CV score
#             avg_r2 = np.mean(cv_scores)
#             return -avg_r2
        
#         # Run Optuna optimization with CV
#         study = optuna.create_study(
#             direction="minimize",
#             sampler=optuna.samplers.TPESampler(seed=42),
#             pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
#         )
        
#         optuna.logging.set_verbosity(optuna.logging.WARNING)
#         study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
#         self.best_params = study.best_params
#         self.cv_mean = -study.best_value
        
#         # Get CV scores from best trial for reporting
#         tscv = TimeSeriesSplit(n_splits=3)
#         best_cv_scores = []
        
#         for train_idx, val_idx in tscv.split(X_trainval):
#             X_tr = X_trainval.iloc[train_idx]
#             X_vl = X_trainval.iloc[val_idx]
#             y_tr = y_trainval.iloc[train_idx]
#             y_vl = y_trainval.iloc[val_idx]
            
#             model = lgb.LGBMRegressor(**self.best_params, random_state=42, 
#                                      n_jobs=-1, verbose=-1)
#             model.fit(X_tr, y_tr)
#             val_pred = model.predict(X_vl)
#             best_cv_scores.append(r2_score(y_vl, val_pred))
        
#         self.cv_scores = np.array(best_cv_scores)
#         self.cv_std = np.std(best_cv_scores)
        
#         print(f"   Best CV Score: {self.cv_mean:.4f} (+/- {self.cv_std:.4f}) from {n_trials} trials")
        
#         # Train FINAL model on ALL training data with best params
#         self.model = lgb.LGBMRegressor(**self.best_params, random_state=42, 
#                                       n_jobs=-1, verbose=-1)
#         self.model.fit(X_trainval, y_trainval)
        
#         return self
    
#     def predict(self, X):
#         return self.model.predict(X)


# # ============================================
# # Unified Training Pipeline
# # ============================================

# class UnifiedModelPipeline:
#     """
#     Unified pipeline that trains all models and selects the best
#     WITH: Visualizations, Validation, MLflow tracking, PROPER CV integration
#     """
    
#     def __init__(self, splits_dir: str = "data/splits", output_dir: str = "models/best_models"):
#         self.splits_dir = splits_dir
#         self.output_dir = output_dir
#         self.results = {}
#         self.best_models = {}
        
#         Path(output_dir).mkdir(parents=True, exist_ok=True)
        
#         # Initialize visualizer and MLflow tracker
#         self.visualizer = ModelVisualizer(output_dir)
#         self.mlflow_tracker = MLflowTracker()
    
#     def train_all_models_for_target(self, target: str, n_trials: int = 30, skip_tuning: bool = False):
#         """
#         Train all 4 model types for a single target
        
#         Args:
#             target: Target variable name
#             n_trials: Number of Optuna trials for tuned models
#             skip_tuning: If True, only train baseline models (faster)
        
#         Returns:
#             Dictionary with results for all models
#         """
#         print(f"\n{'='*80}")
#         print(f"TRAINING ALL MODELS: {target.upper()}")
#         print(f"{'='*80}\n")
        
#         target_results = {}
        
#         # Initialize all trainers
#         trainers = [
#             ("XGBoost", XGBoostTrainer(target)),
#             ("LightGBM", LightGBMTrainer(target)),
#         ]
        
#         if not skip_tuning:
#             trainers.extend([
#                 ("XGBoost-Tuned", XGBoostTunedTrainer(target)),
#                 ("LightGBM-Tuned", LightGBMTunedTrainer(target)),
#             ])
        
#         # Load data once
#         print(f"Loading data...")
#         trainer = trainers[0][1]
#         X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_prepare_data(self.splits_dir)
#         print(f"   Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
#         print(f"   Features: {len(X_train.columns)}")
        
#         # Train each model
#         for model_name, trainer in trainers:
#             print(f"\n{'─'*80}")
#             print(f"Training: {model_name}")
#             print(f"{'─'*80}")
            
#             try:
#                 # Load data for this trainer
#                 X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_prepare_data(self.splits_dir)
                
#                 # TRAIN (CV happens INSIDE for tuned models)
#                 start_time = pd.Timestamp.now()
#                 if "Tuned" in model_name:
#                     trainer.train(X_train, y_train, X_val, y_val, n_trials=n_trials)
#                 else:
#                     trainer.train(X_train, y_train, X_val, y_val)
                
#                 training_time = (pd.Timestamp.now() - start_time).total_seconds()
                
#                 # EVALUATE on all splits
#                 results = trainer.evaluate_all_splits(X_train, y_train, X_val, y_val, X_test, y_test)
                
#                 # GENERATE VISUALIZATIONS
#                 print("   Generating visualizations...")
#                 viz_files = {}
                
#                 # 1. Predictions vs Actual
#                 y_pred_test = trainer.predict(X_test)
#                 viz_files['pred_vs_actual'] = self.visualizer.plot_predictions_vs_actual(
#                     y_test, y_pred_test, 
#                     f"{target} - {model_name}",
#                     f"{target}_{model_name.replace('-', '_')}_pred_vs_actual.png"
#                 )
                
#                 # 2. Residual Analysis
#                 viz_files['residuals'] = self.visualizer.plot_residual_analysis(
#                     y_test, y_pred_test,
#                     f"{target} - {model_name}",
#                     f"{target}_{model_name.replace('-', '_')}_residuals.png"
#                 )
                
#                 # 3. Feature Importance
#                 viz_files['feature_importance'] = self.visualizer.plot_feature_importance(
#                     trainer.model, trainer.feature_names,
#                     f"{target} - {model_name}",
#                     f"{target}_{model_name.replace('-', '_')}_features.png"
#                 )
                
#                 # 4. CV Scores (if available)
#                 if trainer.cv_scores is not None:
#                     viz_files['cv_scores'] = self.visualizer.plot_cv_scores(
#                         trainer.cv_scores,
#                         f"{target} - {model_name}",
#                         f"{target}_{model_name.replace('-', '_')}_cv.png"
#                     )
                
#                 # STORE RESULTS
#                 target_results[model_name] = {
#                     "trainer": trainer,
#                     "results": results,
#                     "training_time": training_time,
#                     "model_type": trainer.model_type,
#                     "cv_mean": trainer.cv_mean,
#                     "cv_std": trainer.cv_std,
#                     "cv_scores": trainer.cv_scores,
#                     "visualizations": viz_files
#                 }
                
#                 # LOG TO MLFLOW
#                 params = {
#                     "model_type": model_name, 
#                     "target": target,
#                     "n_features": len(trainer.feature_names)
#                 }
#                 if hasattr(trainer, 'best_params') and trainer.best_params:
#                     params.update(trainer.best_params)
                
#                 metrics = {
#                     "train_r2": results['train']['r2'],
#                     "val_r2": results['val']['r2'],
#                     "test_r2": results['test']['r2'],
#                     "test_rmse": results['test']['rmse'],
#                     "test_mae": results['test']['mae'],
#                     "training_time_sec": training_time
#                 }
                
#                 # Add CV metrics if available
#                 if trainer.cv_mean is not None:
#                     metrics["cv_mean_r2"] = trainer.cv_mean
#                     metrics["cv_std_r2"] = trainer.cv_std if trainer.cv_std else 0.0
                
#                 # Log to MLflow
#                 run_id = self.mlflow_tracker.log_experiment(
#                     target=target,
#                     model_name=model_name,
#                     model_obj=trainer.model,
#                     params=params,
#                     metrics=metrics,
#                     artifacts=viz_files
#                 )
                
#                 trainer.run_id = run_id
                
#                 # PRINT SUMMARY
#                 print(f"   Train R2: {results['train']['r2']:.4f}")
#                 print(f"   Val R2:   {results['val']['r2']:.4f}")
#                 print(f"   Test R2:  {results['test']['r2']:.4f}")
#                 if trainer.cv_mean is not None:
#                     print(f"   CV Mean:  {trainer.cv_mean:.4f} (+/- {trainer.cv_std:.4f})")
#                 print(f"   Time:     {training_time:.1f}s")
                
#             except Exception as e:
#                 print(f"   ERROR: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         return target_results
    
#     def select_best_model(self, target: str, target_results: Dict):
#         """
#         Select best model based on multiple criteria
        
#         Selection Criteria (in order):
#         1. Test R2 (primary)
#         2. Overfitting gap < 30% (quality check)
#         3. RMSE (tiebreaker)
        
#         Args:
#             target: Target variable name
#             target_results: Results from all models
        
#         Returns:
#             Best model name and selection reasoning
#         """
#         print(f"\n{'='*80}")
#         print(f"MODEL SELECTION: {target.upper()}")
#         print(f"{'='*80}\n")
        
#         # Create comparison table
#         comparison = []
#         for model_name, data in target_results.items():
#             results = data["results"]
            
#             train_r2 = results["train"]["r2"]
#             test_r2 = results["test"]["r2"]
#             overfit_gap = train_r2 - test_r2
#             overfit_pct = (overfit_gap / train_r2 * 100) if train_r2 > 0 else 0
            
#             comparison.append({
#                 "model": model_name,
#                 "test_r2": test_r2,
#                 "train_r2": train_r2,
#                 "overfit_gap": overfit_gap,
#                 "overfit_pct": overfit_pct,
#                 "test_rmse": results["test"]["rmse"],
#                 "trainer": data["trainer"],
#                 "cv_mean": data.get("cv_mean"),  # Can be None for baseline models
#                 "cv_std": data.get("cv_std", 0.0)
#             })
        
#         # Sort by test R2 (descending)
#         comparison_sorted = sorted(comparison, key=lambda x: x["test_r2"], reverse=True)
        
#         # Print comparison table
#         print(f"{'Model':<20} {'Test R2':>10} {'Train R2':>10} {'CV Mean':>10} {'Overfit %':>12} {'Test RMSE':>12}")
#         print(f"{'─'*80}")
        
#         for item in comparison_sorted:
#             overfit_symbol = "[OK]" if item["overfit_pct"] < 30 else "[WARN]"
#             # FIXED: Check if cv_mean is not None
#             cv_str = f"{item['cv_mean']:.4f}" if item['cv_mean'] is not None else "N/A"
#             print(f"{item['model']:<20} {item['test_r2']:>10.4f} {item['train_r2']:>10.4f} "
#                   f"{cv_str:>10} {overfit_symbol} {item['overfit_pct']:>9.1f}% "
#                   f"{item['test_rmse']:>12,.2f}")
        
#         # Selection logic
#         best = comparison_sorted[0]
        
#         # Generate model comparison visualization
#         print("\n   Generating model comparison chart...")
#         comparison_viz = self.visualizer.plot_model_comparison(
#             comparison_sorted, target,
#             f"{target}_model_comparison.png"
#         )
        
#         # Check for excessive overfitting
#         if best["overfit_pct"] > 30:
#             print(f"\nWARNING: Best model has {best['overfit_pct']:.1f}% overfitting!")
            
#             # Look for alternative with less overfitting
#             alternatives = [c for c in comparison_sorted[1:] if c["overfit_pct"] < 30]
#             if alternatives:
#                 alternative = alternatives[0]
#                 r2_sacrifice = best["test_r2"] - alternative["test_r2"]
#                 r2_sacrifice_pct = (r2_sacrifice / best["test_r2"]) * 100
                
#                 if r2_sacrifice_pct < 3.0:
#                     print(f"   Switching to {alternative['model']} (better generalization)")
#                     print(f"   R2 sacrifice: {r2_sacrifice:.4f} ({r2_sacrifice_pct:.1f}%)")
#                     best = alternative
        
#         print(f"\n{'─'*80}")
#         print(f"SELECTED: {best['model']}")
#         print(f"{'─'*80}")
#         print(f"   Test R2: {best['test_r2']:.4f}")
#         # FIXED: Check if cv_mean exists before printing
#         if best.get('cv_mean') is not None:
#             print(f"   CV Mean: {best['cv_mean']:.4f} (+/- {best['cv_std']:.4f})")
#         print(f"   Overfit: {best['overfit_pct']:.1f}%")
        
#         # Selection reasoning
#         reasoning = f"Selected {best['model']} with Test R2={best['test_r2']:.4f}. "
#         if best["overfit_pct"] < 20:
#             reasoning += "Excellent generalization."
#         elif best["overfit_pct"] < 30:
#             reasoning += "Good generalization."
#         else:
#             reasoning += f"Warning: {best['overfit_pct']:.1f}% overfitting detected."
        
#         return best["model"], best["trainer"], reasoning, comparison_sorted
    
#     def save_best_model(self, target: str, model_name: str, trainer: BaseModelTrainer, reasoning: str):
#         """
#         Save best model as {target}_best.pkl and register to MLflow
        
#         Args:
#             target: Target variable name
#             model_name: Name of selected model
#             trainer: Trained model object
#             reasoning: Selection reasoning
#         """
#         output_file = Path(self.output_dir) / f"{target}_best.pkl"
        
#         # Determine deployment recommendation
#         test_r2 = trainer.test_metrics["r2"]
#         if test_r2 >= 0.7:
#             deploy_rec = "production_ready"
#         elif test_r2 >= 0.5:
#             deploy_rec = "use_with_caution"
#         elif test_r2 >= 0.25:
#             deploy_rec = "research_only"
#         else:
#             deploy_rec = "low_confidence"
        
#         model_data = {
#             "target": target,
#             "model_type": model_name,
#             "model": trainer.model,
#             "feature_names": trainer.feature_names,
#             "train_metrics": trainer.train_metrics,
#             "val_metrics": trainer.val_metrics,
#             "test_metrics": trainer.test_metrics,
#             "cv_mean": trainer.cv_mean,
#             "cv_std": trainer.cv_std,
#             "cv_scores": trainer.cv_scores.tolist() if trainer.cv_scores is not None else None,
#             "selection_reasoning": reasoning,
#             "deployment_recommendation": deploy_rec,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         # Add model-specific parameters
#         if hasattr(trainer, 'best_params'):
#             model_data["hyperparameters"] = trainer.best_params
        
#         # Save to pickle
#         joblib.dump(model_data, output_file)
        
#         print(f"\nModel saved: {output_file}")
#         print(f"   Deployment: {deploy_rec}")
        
#         # Register to MLflow Model Registry
#         if trainer.run_id:
#             self.mlflow_tracker.register_best_model(
#                 target=target,
#                 model_name=model_name,
#                 run_id=trainer.run_id,
#                 test_r2=test_r2,
#                 deployment_rec=deploy_rec
#             )
        
#         return output_file
    
#     def generate_comparison_report(self, all_results: Dict, output_dir: str = None):
#         """
#         Generate comprehensive comparison report for all targets
        
#         Args:
#             all_results: Results from all targets
#             output_dir: Output directory (default: self.output_dir)
#         """
#         if output_dir is None:
#             output_dir = self.output_dir
        
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         print(f"\n{'='*80}")
#         print(f"GENERATING COMPARISON REPORT")
#         print(f"{'='*80}\n")
        
#         # Prepare report data
#         report = {
#             "timestamp": datetime.now().isoformat(),
#             "total_targets": len(all_results),
#             "methodology": "Time Series CV during tuning + Temporal train/val/test split",
#             "targets": {}
#         }
        
#         summary_table = []
#         deployment_ready = []
        
#         for target, data in all_results.items():
#             best_model_name = data["best_model"]
#             best_trainer = data["best_trainer"]
#             comparison = data["comparison"]
            
#             test_r2 = best_trainer.test_metrics["r2"]
#             cv_mean = best_trainer.cv_mean
#             cv_std = best_trainer.cv_std
            
#             # Determine deployment readiness
#             if test_r2 >= 0.7:
#                 deploy_status = "[PROD] Production Ready"
#                 deployment_ready.append(target)
#             elif test_r2 >= 0.5:
#                 deploy_status = "[WARN] Use with Caution"
#             elif test_r2 >= 0.25:
#                 deploy_status = "[RESEARCH] Research Only"
#             else:
#                 deploy_status = "[LOW] Low Confidence"
            
#             # Target-specific report
#             report["targets"][target] = {
#                 "selected_model": best_model_name,
#                 "selection_reasoning": data["reasoning"],
#                 "test_r2": test_r2,
#                 "test_rmse": best_trainer.test_metrics["rmse"],
#                 "test_mae": best_trainer.test_metrics["mae"],
#                 "cv_mean": float(cv_mean) if cv_mean is not None else None,
#                 "cv_std": float(cv_std) if cv_std is not None else None,
#                 "deployment_status": deploy_status,
#                 "mlflow_run_id": best_trainer.run_id,
#                 "all_models": [
#                     {
#                         "model": c["model"],
#                         "test_r2": c["test_r2"],
#                         "overfit_pct": c["overfit_pct"],
#                         "cv_mean": c.get("cv_mean"),
#                         "cv_std": c.get("cv_std")
#                     }
#                     for c in comparison
#                 ]
#             }
            
#             summary_table.append({
#                 "target": target,
#                 "model": best_model_name,
#                 "test_r2": test_r2,
#                 "test_rmse": best_trainer.test_metrics["rmse"],
#                 "cv_mean": cv_mean,
#                 "deploy_status": deploy_status
#             })
        
#         # Save JSON report
#         report_file = output_path / "model_comparison_report.json"
#         with open(report_file, "w") as f:
#             json.dump(report, f, indent=2)
        
#         print(f"   JSON report: {report_file}")
        
#         # Create summary table
#         print(f"\n{'='*80}")
#         print(f"FINAL MODEL SUMMARY")
#         print(f"{'='*80}\n")
        
#         print(f"{'Target':<20} {'Selected Model':<20} {'Test R2':>10} {'CV Mean':>10} {'Status':<30}")
#         print(f"{'─'*95}")
        
#         for row in summary_table:
#             # FIXED: Check if cv_mean is not None
#             cv_str = f"{row['cv_mean']:.4f}" if row['cv_mean'] is not None else "N/A"
#             print(f"{row['target']:<20} {row['model']:<20} {row['test_r2']:>10.4f} "
#                   f"{cv_str:>10} {row['deploy_status']:<30}")
        
#         # Average performance
#         avg_r2 = np.mean([r["test_r2"] for r in summary_table])
#         print(f"{'─'*95}")
#         print(f"{'AVERAGE':<20} {'':20} {avg_r2:>10.4f}")
        
#         print(f"\n{'='*80}")
#         print(f"DEPLOYMENT RECOMMENDATION")
#         print(f"{'='*80}")
#         print(f"Production Ready: {', '.join(deployment_ready) if deployment_ready else 'None'}")
#         print(f"Total Models: {len(summary_table)}")
        
#         # Visualizations summary
#         print(f"\n{'='*80}")
#         print(f"VISUALIZATIONS GENERATED")
#         print(f"{'='*80}")
#         print(f"Location: {self.visualizer.viz_dir}")
#         print(f"Files per model:")
#         print(f"  - predictions_vs_actual.png")
#         print(f"  - residual_analysis.png")
#         print(f"  - feature_importance.png")
#         print(f"  - cv_scores.png (for tuned models)")
#         print(f"  - model_comparison.png (per target)")
        
#         return report_file
    
#     def run_pipeline(self, targets: List[str], n_trials: int = 30, skip_tuning: bool = False):
#         """
#         Run complete pipeline for all targets
        
#         Args:
#             targets: List of target variable names
#             n_trials: Number of Optuna trials for tuned models
#             skip_tuning: If True, only train baseline models
        
#         Returns:
#             Dictionary with all results
#         """
#         print(f"\n{'='*80}")
#         print(f"UNIFIED MODEL TRAINING PIPELINE")
#         print(f"{'='*80}")
#         print(f"Targets: {', '.join(targets)}")
#         print(f"Models: XGBoost, LightGBM" + ("" if skip_tuning else ", XGBoost-Tuned, LightGBM-Tuned"))
#         print(f"Tuning trials: {n_trials if not skip_tuning else 'N/A (skipped)'}")
#         print(f"Methodology: Time Series CV (during tuning) + Temporal Test Split")
#         print(f"Features: Visualizations + MLflow Tracking")
#         print(f"{'='*80}\n")
        
#         all_results = {}
        
#         for i, target in enumerate(targets, 1):
#             print(f"\n{'#'*80}")
#             print(f"TARGET {i}/{len(targets)}: {target.upper()}")
#             print(f"{'#'*80}")
            
#             try:
#                 # Train all models for this target
#                 target_results = self.train_all_models_for_target(target, n_trials, skip_tuning)
                
#                 # Select best model
#                 best_model_name, best_trainer, reasoning, comparison = self.select_best_model(target, target_results)
                
#                 # Save best model
#                 model_file = self.save_best_model(target, best_model_name, best_trainer, reasoning)
                
#                 # Store results
#                 all_results[target] = {
#                     "best_model": best_model_name,
#                     "best_trainer": best_trainer,
#                     "reasoning": reasoning,
#                     "comparison": comparison,
#                     "model_file": str(model_file),
#                     "all_models": target_results
#                 }
                
#                 print(f"\n{target.upper()} COMPLETE")
                
#             except Exception as e:
#                 print(f"\nERROR processing {target}:")
#                 print(f"   {str(e)}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         # Generate comparison report
#         if all_results:
#             self.generate_comparison_report(all_results)
        
#         return all_results


# # ============================================
# # CLI Interface
# # ============================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Unified model training with proper CV integration, visualizations, and MLflow tracking",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Train all targets with all models (default)
#   python unified_model_training_and_selection.py
  
#   # Train specific target
#   python unified_model_training_and_selection.py --target profit_margin
  
#   # Quick mode (skip tuned models for faster results)
#   python unified_model_training_and_selection.py --quick
  
#   # Custom number of tuning trials
#   python unified_model_training_and_selection.py --trials 50
  
#   # Multiple specific targets
#   python unified_model_training_and_selection.py --target revenue eps
#         """
#     )
    
#     parser.add_argument(
#         "--target",
#         type=str,
#         nargs="+",
#         default=["all"],
#         help="Target(s) to train (default: all). Options: revenue, eps, debt_equity, profit_margin, stock_return, all"
#     )
#     parser.add_argument(
#         "--splits-dir",
#         type=str,
#         default="data/splits",
#         help="Directory containing train/val/test splits"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="models/best_models",
#         help="Directory to save best models"
#     )
#     parser.add_argument(
#         "--trials",
#         type=int,
#         default=30,
#         help="Number of Optuna trials for tuned models (default: 30)"
#     )
#     parser.add_argument(
#         "--quick",
#         action="store_true",
#         help="Quick mode: skip tuned models (faster, only baseline XGBoost and LightGBM)"
#     )
    
#     args = parser.parse_args()
    
#     # Determine targets
#     all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]
    
#     if "all" in args.target:
#         targets = all_targets
#     else:
#         targets = [t for t in args.target if t in all_targets]
#         if not targets:
#             print(f"Invalid target(s): {args.target}")
#             print(f"   Valid options: {', '.join(all_targets)}, all")
#             return
    
#     # Initialize pipeline
#     pipeline = UnifiedModelPipeline(
#         splits_dir=args.splits_dir,
#         output_dir=args.output_dir
#     )
    
#     # Run pipeline
#     results = pipeline.run_pipeline(
#         targets=targets,
#         n_trials=args.trials,
#         skip_tuning=args.quick
#     )
    
#     # Final summary
#     if results:
#         print(f"\n{'='*80}")
#         print(f"PIPELINE COMPLETE")
#         print(f"{'='*80}")
#         print(f"\nBest models saved to: {args.output_dir}/")
#         print(f"\nFiles created:")
#         for target in results.keys():
#             print(f"   - {target}_best.pkl")
#         print(f"   - model_comparison_report.json")
        
#         print(f"\nVisualizations:")
#         print(f"   - Location: {args.output_dir}/visualizations/")
#         print(f"   - Per model: predictions, residuals, features, CV scores")
#         print(f"   - Per target: model comparison charts")
        
#         if MLFLOW_AVAILABLE:
#             print(f"\nMLflow:")
#             print(f"   - Experiments logged to MLflow")
#             print(f"   - Models registered in Model Registry")
#             print(f"   - View at: http://localhost:5000")
#             print(f"   - Check 'Models' tab in MLflow UI")
        
#         print(f"\nNext steps:")
#         print(f"   1. Open MLflow UI: mlflow ui")
#         print(f"   2. Navigate to 'Models' tab to see registered models")
#         print(f"   3. Review visualizations in {args.output_dir}/visualizations/")
#         print(f"   4. Check model_comparison_report.json")
        
#         print(f"\nTo use a best model:")
#         print(f"   # From pickle file:")
#         print(f"   import joblib")
#         print(f"   model_data = joblib.load('{args.output_dir}/revenue_best.pkl')")
#         print(f"   model = model_data['model']")
#         print(f"   predictions = model.predict(X_new)")
#         print(f"")
#         print(f"   # From MLflow:")
#         print(f"   import mlflow")
#         print(f"   model = mlflow.pyfunc.load_model('models:/revenue_predictor/production')")
#         print(f"   predictions = model.predict(X_new)")
#     else:
#         print(f"\nNo models trained successfully")


# if __name__ == "__main__":
#     main()





"""
src/models/unified_model_training_and_selection.py

UNIFIED TRAINING PIPELINE with BIAS DETECTION

Added Features:
- Sector bias detection
- Company size bias detection  
- Temporal bias detection
- Prediction direction bias
- Comprehensive bias visualization
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import optuna
from typing import Dict, Tuple, List
from scipy import stats

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLflow not available - experiment tracking disabled")

warnings.filterwarnings("ignore")

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent.parent  
sys.path.insert(0, str(project_root))

from src.utils.split_utils import get_feature_target_split, drop_nan_targets

print("Imports successful\n")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BiasDetector:
    """Detects and analyzes bias in model predictions"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.bias_results = {}
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        if mask.sum() == 0:
            return None
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    def detect_sector_bias(self, model, X_test, y_test, test_df, target_name):
        """Detect if model performs differently across sectors - FIXED"""
        print(f"   Analyzing sector bias...")
        
        if 'Sector' not in test_df.columns:
            print(f"   WARNING: Sector column not found, skipping sector bias detection")
            return None
        
        # CRITICAL FIX: Align indices
        # Reset all indices to ensure alignment
        test_df = test_df.reset_index(drop=True)
        X_test_reset = X_test.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(y_test).reset_index(drop=True)
        
        # If lengths don't match, align using X_test's original index
        if len(test_df) != len(y_test_reset):
            print(f"   Aligning dataframes: test_df({len(test_df)}) vs y_test({len(y_test_reset)})")
            # Use iloc to match lengths
            min_len = min(len(test_df), len(y_test_reset))
            test_df = test_df.iloc[:min_len]
            X_test_reset = X_test_reset.iloc[:min_len]
            y_test_reset = y_test_reset.iloc[:min_len]
        
        sectors = test_df['Sector'].values
        predictions = model.predict(X_test_reset)
        
        sector_performance = {}
        
        for sector in sorted(test_df['Sector'].unique()):
            mask = sectors == sector
            
            if mask.sum() < 10:
                continue
            
            y_sector = y_test_reset.values[mask] if hasattr(y_test_reset, 'values') else y_test_reset[mask]
            pred_sector = predictions[mask]
            
            r2_sector = r2_score(y_sector, pred_sector)
            rmse_sector = np.sqrt(mean_squared_error(y_sector, pred_sector))
            mae_sector = mean_absolute_error(y_sector, pred_sector)
            mape_sector = self.calculate_mape(y_sector, pred_sector)
            
            residuals_sector = y_sector - pred_sector
            residual_mean = np.mean(residuals_sector)
            residual_std = np.std(residuals_sector)
            
            sector_performance[sector] = {
                'r2': r2_sector,
                'rmse': rmse_sector,
                'mae': mae_sector,
                'mape': mape_sector,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'n_samples': int(mask.sum())
            }
        
        r2_values = [v['r2'] for v in sector_performance.values()]
        r2_mean = np.mean(r2_values)
        r2_std = np.std(r2_values)
        r2_min = min(r2_values)
        r2_max = max(r2_values)
        
        bias_summary = {
            'r2_mean': r2_mean,
            'r2_std': r2_std,
            'r2_range': r2_max - r2_min,
            'bias_detected': r2_std > 0.10,
            'sectors': sector_performance
        }
        
        print(f"\n   Sector Bias Analysis:")
        print(f"   {'Sector':<20} {'R2':>8} {'RMSE':>12} {'Samples':>10} {'Bias':>10}")
        print(f"   {'─'*65}")
        
        for sector, perf in sorted(sector_performance.items(), key=lambda x: x[1]['r2'], reverse=True):
            bias_flag = "OVER" if perf['residual_mean'] > 0 else "UNDER" if perf['residual_mean'] < 0 else "OK"
            print(f"   {sector:<20} {perf['r2']:>8.4f} {perf['rmse']:>12,.2f} {perf['n_samples']:>10} {bias_flag:>10}")
        
        print(f"   {'─'*65}")
        print(f"   {'STD DEVIATION':<20} {r2_std:>8.4f}")
        
        if bias_summary['bias_detected']:
            print(f"   WARNING: Sector bias detected! (R² std = {r2_std:.4f} > 0.10)")
        else:
            print(f"   OK: No significant sector bias (R² std = {r2_std:.4f} < 0.10)")
        
        return bias_summary
    
    def detect_size_bias(self, model, X_test, y_test, test_df, target_name):
        """Detect company size bias - FIXED"""
        print(f"   Analyzing company size bias...")
        
        if 'Total_Revenue' not in test_df.columns:
            print(f"   WARNING: Total_Revenue column not found, skipping size bias detection")
            return None
        
        # CRITICAL FIX: Align indices
        test_df = test_df.reset_index(drop=True)
        X_test_reset = X_test.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(y_test).reset_index(drop=True)
        
        if len(test_df) != len(y_test_reset):
            min_len = min(len(test_df), len(y_test_reset))
            test_df = test_df.iloc[:min_len]
            X_test_reset = X_test_reset.iloc[:min_len]
            y_test_reset = y_test_reset.iloc[:min_len]
        
        revenue_values = test_df['Total_Revenue'].values
        predictions = model.predict(X_test_reset)
        
        revenue_percentiles = [0, 33, 67, 100]
        labels = ['Small', 'Medium', 'Large']
        
        size_buckets = pd.cut(revenue_values, 
                             bins=np.percentile(revenue_values, revenue_percentiles),
                             labels=labels,
                             include_lowest=True)
        
        size_performance = {}
        
        for size in labels:
            mask = size_buckets == size
            
            if mask.sum() < 10:
                continue
            
            y_size = y_test_reset.values[mask] if hasattr(y_test_reset, 'values') else y_test_reset[mask]
            pred_size = predictions[mask]
            
            r2_size = r2_score(y_size, pred_size)
            rmse_size = np.sqrt(mean_squared_error(y_size, pred_size))
            residual_mean = np.mean(y_size - pred_size)
            
            size_performance[size] = {
                'r2': r2_size,
                'rmse': rmse_size,
                'residual_mean': residual_mean,
                'n_samples': int(mask.sum())
            }
        
        r2_values = [v['r2'] for v in size_performance.values()]
        r2_std = np.std(r2_values)
        
        bias_summary = {
            'r2_std': r2_std,
            'bias_detected': r2_std > 0.10,
            'sizes': size_performance
        }
        
        print(f"\n   Company Size Bias Analysis:")
        print(f"   {'Size':<15} {'R2':>8} {'RMSE':>12} {'Samples':>10}")
        print(f"   {'─'*50}")
        
        for size, perf in size_performance.items():
            print(f"   {size:<15} {perf['r2']:>8.4f} {perf['rmse']:>12,.2f} {perf['n_samples']:>10}")
        
        print(f"   {'─'*50}")
        
        if bias_summary['bias_detected']:
            print(f"   WARNING: Size bias detected! (R² std = {r2_std:.4f} > 0.10)")
        else:
            print(f"   OK: No significant size bias (R² std = {r2_std:.4f} < 0.10)")
        
        return bias_summary
    
    def detect_temporal_bias(self, model, X_test, y_test, test_df):
        """Detect temporal bias - FIXED"""
        print(f"   Analyzing temporal bias...")
        
        if 'Year' not in test_df.columns:
            print(f"   WARNING: Year column not found, skipping temporal bias detection")
            return None
        
        # CRITICAL FIX: Align indices
        test_df = test_df.reset_index(drop=True)
        X_test_reset = X_test.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(y_test).reset_index(drop=True)
        
        if len(test_df) != len(y_test_reset):
            min_len = min(len(test_df), len(y_test_reset))
            test_df = test_df.iloc[:min_len]
            X_test_reset = X_test_reset.iloc[:min_len]
            y_test_reset = y_test_reset.iloc[:min_len]
        
        years = test_df['Year'].values
        predictions = model.predict(X_test_reset)
        
        pre_covid_mask = years < 2020
        post_covid_mask = years >= 2020
        
        temporal_performance = {}
        
        for period, mask in [('Pre-2020', pre_covid_mask), ('2020+', post_covid_mask)]:
            if mask.sum() < 10:
                continue
            
            y_period = y_test_reset.values[mask] if hasattr(y_test_reset, 'values') else y_test_reset[mask]
            pred_period = predictions[mask]
            
            r2_period = r2_score(y_period, pred_period)
            rmse_period = np.sqrt(mean_squared_error(y_period, pred_period))
            
            temporal_performance[period] = {
                'r2': r2_period,
                'rmse': rmse_period,
                'n_samples': int(mask.sum())
            }
        
        if len(temporal_performance) == 2:
            r2_diff = abs(temporal_performance['Pre-2020']['r2'] - temporal_performance['2020+']['r2'])
            bias_detected = r2_diff > 0.15
        else:
            r2_diff = 0
            bias_detected = False
        
        bias_summary = {
            'r2_difference': r2_diff,
            'bias_detected': bias_detected,
            'periods': temporal_performance
        }
        
        print(f"\n   Temporal Bias Analysis:")
        print(f"   {'Period':<15} {'R2':>8} {'RMSE':>12} {'Samples':>10}")
        print(f"   {'─'*50}")
        
        for period, perf in temporal_performance.items():
            print(f"   {period:<15} {perf['r2']:>8.4f} {perf['rmse']:>12,.2f} {perf['n_samples']:>10}")
        
        print(f"   {'─'*50}")
        
        if bias_detected:
            print(f"   WARNING: Temporal bias detected! (R² diff = {r2_diff:.4f} > 0.15)")
        else:
            print(f"   OK: No significant temporal bias (R² diff = {r2_diff:.4f} < 0.15)")
        
        return bias_summary
    
    def plot_sector_bias(self, sector_performance, target: str, filename: str):
        """Visualize sector bias"""
        if not sector_performance or 'sectors' not in sector_performance:
            return None
        
        sectors_data = sector_performance['sectors']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sectors = list(sectors_data.keys())
        r2_scores = [sectors_data[s]['r2'] for s in sectors]
        rmse_scores = [sectors_data[s]['rmse'] for s in sectors]
        residual_means = [sectors_data[s]['residual_mean'] for s in sectors]
        sample_counts = [sectors_data[s]['n_samples'] for s in sectors]
        
        # 1. R² by Sector
        ax1 = axes[0, 0]
        bars = ax1.barh(sectors, r2_scores, color='skyblue', edgecolor='black')
        avg_r2 = np.mean(r2_scores)
        ax1.axvline(x=avg_r2, color='r', linestyle='--', lw=2, label=f'Average: {avg_r2:.4f}')
        ax1.set_xlabel('R² Score')
        ax1.set_title('R² Performance by Sector')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
            if r2 < avg_r2 - 0.10:
                bar.set_color('coral')
            elif r2 > avg_r2 + 0.10:
                bar.set_color('lightgreen')
        
        # 2. RMSE by Sector
        ax2 = axes[0, 1]
        ax2.barh(sectors, rmse_scores, color='coral', edgecolor='black')
        avg_rmse = np.mean(rmse_scores)
        ax2.axvline(x=avg_rmse, color='r', linestyle='--', lw=2, label=f'Average: {avg_rmse:,.0f}')
        ax2.set_xlabel('RMSE')
        ax2.set_title('Error Magnitude by Sector')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Residual Mean by Sector
        ax3 = axes[1, 0]
        colors = ['green' if abs(rm) < np.std(residual_means) else 'orange' for rm in residual_means]
        ax3.barh(sectors, residual_means, color=colors, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='No Bias')
        ax3.set_xlabel('Mean Residual (Actual - Predicted)')
        ax3.set_title('Prediction Direction Bias by Sector')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Sample Count by Sector
        ax4 = axes[1, 1]
        ax4.barh(sectors, sample_counts, color='lightblue', edgecolor='black')
        ax4.set_xlabel('Number of Samples')
        ax4.set_title('Sample Distribution by Sector')
        ax4.grid(True, alpha=0.3, axis='x')
        
        r2_std = sector_performance['r2_std']
        bias_status = "BIAS DETECTED" if sector_performance['bias_detected'] else "NO BIAS"
        fig.suptitle(f'{target.upper()} - Sector Bias Analysis | Status: {bias_status} (R² std: {r2_std:.4f})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.visualizer.viz_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(self.visualizer.viz_dir / filename)
    
    def comprehensive_bias_report(self, model, X_test, y_test, test_df, target: str, model_name: str):
        """Run all bias detection analyses"""
        print(f"\n{'─'*80}")
        print(f"BIAS DETECTION: {target.upper()} - {model_name}")
        print(f"{'─'*80}")
        
        bias_report = {
            'target': target,
            'model': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Sector bias
            sector_bias = self.detect_sector_bias(model, X_test, y_test, test_df, target)
            if sector_bias:
                bias_report['sector_bias'] = sector_bias
                
                viz_path = self.plot_sector_bias(
                    sector_bias, 
                    f"{target} - {model_name}",
                    f"{target}_{model_name.replace('-', '_')}_sector_bias.png"
                )
                bias_report['sector_bias_viz'] = viz_path
        except Exception as e:
            print(f"   ERROR in sector bias detection: {e}")
            bias_report['sector_bias'] = None
        
        try:
            # Company size bias
            size_bias = self.detect_size_bias(model, X_test, y_test, test_df, target)
            if size_bias:
                bias_report['size_bias'] = size_bias
        except Exception as e:
            print(f"   ERROR in size bias detection: {e}")
            bias_report['size_bias'] = None
        
        try:
            # Temporal bias
            temporal_bias = self.detect_temporal_bias(model, X_test, y_test, test_df)
            if temporal_bias:
                bias_report['temporal_bias'] = temporal_bias
        except Exception as e:
            print(f"   ERROR in temporal bias detection: {e}")
            bias_report['temporal_bias'] = None
        
        # Overall bias assessment
        biases_detected = []
        if sector_bias and sector_bias.get('bias_detected'):
            biases_detected.append('sector')
        if size_bias and size_bias.get('bias_detected'):
            biases_detected.append('company_size')
        if temporal_bias and temporal_bias.get('bias_detected'):
            biases_detected.append('temporal')
        
        bias_report['biases_detected'] = biases_detected
        bias_report['overall_status'] = 'BIASED' if biases_detected else 'FAIR'
        
        print(f"\n   Overall Bias Status: {bias_report['overall_status']}")
        if biases_detected:
            print(f"   Biases found in: {', '.join(biases_detected)}")
        
        return bias_report

# ============================================
# Visualization Functions (ENHANCED)
# ============================================

class ModelVisualizer:
    """Handles all model visualizations"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_predictions_vs_actual(self, y_true, y_pred, title: str, filename: str):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{title} - Predictions vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add R2 to plot
        r2 = r2_score(y_true, y_pred)
        axes[0].text(0.05, 0.95, f'R2 = {r2:.4f}', 
                    transform=axes[0].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{title} - Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(self.viz_dir / filename)
    
    def plot_residual_analysis(self, y_true, y_pred, title: str, filename: str):
        """Comprehensive residual analysis"""
        residuals = y_true - y_pred
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Residuals vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        ax3 = fig.add_subplot(gs[0, 2])
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals vs Order (time series check)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(residuals, marker='o', linestyle='', alpha=0.6)
        ax4.axhline(y=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Observation Order')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Order')
        ax4.grid(True, alpha=0.3)
        
        # 5. Absolute residuals vs Predicted (heteroscedasticity)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(y_pred, np.abs(residuals), alpha=0.6, edgecolors='k', linewidth=0.5)
        ax5.set_xlabel('Predicted Values')
        ax5.set_ylabel('Absolute Residuals')
        ax5.set_title('Scale-Location Plot')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Calculate statistics
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)
        _, p_value_shapiro = stats.shapiro(residuals[:min(5000, len(residuals))])
        
        stats_text = f"""
        Residual Statistics:
        
        Mean: {mean_resid:.6f}
        Std Dev: {std_resid:.6f}
        Min: {np.min(residuals):.4f}
        Max: {np.max(residuals):.4f}
        
        Normality Test (Shapiro-Wilk):
        p-value: {p_value_shapiro:.4f}
        {'PASS' if p_value_shapiro > 0.05 else 'FAIL'} (alpha=0.05)
        
        RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}
        MAE: {mean_absolute_error(y_true, y_pred):.4f}
        R2: {r2_score(y_true, y_pred):.4f}
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace')
        
        plt.suptitle(f'{title} - Comprehensive Residual Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(self.viz_dir / filename)
    
    def plot_feature_importance(self, model, feature_names: List[str], title: str, 
                                filename: str, top_n: int = 20):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            print(f"Cannot extract feature importance from {type(model)}")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        bars = ax.barh(range(len(importance_df)), importance_df['importance'])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{title} - Top {top_n} Features')
        ax.invert_yaxis()
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(self.viz_dir / filename)
    
    def plot_cv_scores(self, cv_scores: np.ndarray, title: str, filename: str):
        """Plot cross-validation scores"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].boxplot(cv_scores, vert=True)
        axes[0].set_ylabel('R2 Score')
        axes[0].set_title(f'{title} - CV Score Distribution')
        axes[0].grid(True, alpha=0.3)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        axes[0].axhline(y=mean_score, color='r', linestyle='--', 
                       label=f'Mean: {mean_score:.4f}')
        axes[0].legend()
        
        axes[1].bar(range(len(cv_scores)), cv_scores, color='skyblue', edgecolor='black')
        axes[1].axhline(y=mean_score, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('R2 Score')
        axes[1].set_title(f'{title} - Scores by Fold')
        axes[1].grid(True, alpha=0.3)
        
        stats_text = f'Mean: {mean_score:.4f}\nStd: {std_score:.4f}'
        axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(self.viz_dir / filename)
    
    def plot_model_comparison(self, comparison_data: List[Dict], target: str, filename: str):
        """Plot comparison of all models"""
        df = pd.DataFrame(comparison_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].barh(df['model'], df['test_r2'], color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Test R2')
        axes[0].set_title('Model Performance (Test R2)')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        colors = ['green' if x < 30 else 'orange' for x in df['overfit_pct']]
        axes[1].barh(df['model'], df['overfit_pct'], color=colors, edgecolor='black')
        axes[1].axvline(x=30, color='r', linestyle='--', lw=2, label='30% threshold')
        axes[1].set_xlabel('Overfitting %')
        axes[1].set_title('Overfitting Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        axes[2].barh(df['model'], df['test_rmse'], color='coral', edgecolor='black')
        axes[2].set_xlabel('Test RMSE')
        axes[2].set_title('Prediction Error (Test RMSE)')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'{target.upper()} - Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(self.viz_dir / filename)


# ============================================
# MLflow Integration
# ============================================

class MLflowTracker:
    """Handles MLflow experiment tracking and model registry"""
    
    def __init__(self, experiment_name: str = "financial-forecasting"):
        self.experiment_name = experiment_name
        self.enabled = MLFLOW_AVAILABLE
        self.active_run = None
        
        if self.enabled:
            try:
                mlflow.set_experiment(experiment_name)
                print(f"MLflow experiment set: {experiment_name}\n")
            except Exception as e:
                print(f"MLflow initialization failed: {e}")
                self.enabled = False
        else:
            print("MLflow tracking disabled (library not available)\n")
    
    def log_experiment(self, target: str, model_name: str, model_obj, 
                      params: Dict, metrics: Dict, artifacts: Dict = None, bias_report: Dict = None):
        """Log complete experiment to MLflow"""
        if not self.enabled:
            return None
        
        run_id = None
        
        try:
            run_name = f"{target}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log bias metrics if available
                if bias_report:
                    if 'sector_bias' in bias_report and bias_report['sector_bias']:
                        mlflow.log_metric("sector_r2_std", bias_report['sector_bias']['r2_std'])
                        mlflow.log_metric("sector_bias_detected", 1 if bias_report['sector_bias']['bias_detected'] else 0)
                    
                    if 'size_bias' in bias_report and bias_report['size_bias']:
                        mlflow.log_metric("size_r2_std", bias_report['size_bias']['r2_std'])
                        mlflow.log_metric("size_bias_detected", 1 if bias_report['size_bias']['bias_detected'] else 0)
                    
                    if 'temporal_bias' in bias_report and bias_report['temporal_bias']:
                        mlflow.log_metric("temporal_r2_diff", bias_report['temporal_bias']['r2_difference'])
                        mlflow.log_metric("temporal_bias_detected", 1 if bias_report['temporal_bias']['bias_detected'] else 0)
                
                # Log model
                artifact_name = f"{target}_{model_name}_model"
                
                if model_obj is not None:
                    model_type_str = str(type(model_obj).__name__).lower()
                    
                    if 'xgb' in model_type_str or isinstance(model_obj, xgb.XGBRegressor):
                        mlflow.xgboost.log_model(model_obj, artifact_name)
                    elif 'booster' in model_type_str and hasattr(model_obj, 'predict'):
                        mlflow.lightgbm.log_model(model_obj, artifact_name)
                    elif 'lgbm' in model_type_str or isinstance(model_obj, lgb.LGBMRegressor):
                        mlflow.lightgbm.log_model(model_obj, artifact_name)
                    else:
                        mlflow.sklearn.log_model(model_obj, artifact_name)
                
                # Log artifacts
                if artifacts:
                    for key, path in artifacts.items():
                        if path and Path(path).exists():
                            mlflow.log_artifact(path, artifact_path="visualizations")
                
                # Log bias report as JSON artifact
                if bias_report:
                    bias_file = Path(self.experiment_name) / f"bias_report_{target}_{model_name}.json"
                    bias_file.parent.mkdir(exist_ok=True)
                    with open(bias_file, 'w') as f:
                        # Make numpy types JSON serializable
                        serializable_report = json.loads(json.dumps(bias_report, default=str))
                        json.dump(serializable_report, f, indent=2)
                    mlflow.log_artifact(str(bias_file), artifact_path="bias_reports")
                
                mlflow.set_tags({
                    "target": target,
                    "model_type": model_name,
                    "training_date": datetime.now().strftime('%Y-%m-%d'),
                    "framework": "xgboost" if "XGBoost" in model_name else "lightgbm",
                    "bias_status": bias_report.get('overall_status', 'unknown') if bias_report else 'not_checked'
                })
                
                print(f"  [MLflow] Logged: {target}/{model_name} (run_id: {run_id[:8]}...)")
            
            return run_id
            
        except Exception as e:
            print(f"  [MLflow] Logging failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def register_best_model(self, target: str, model_name: str, run_id: str, 
                           test_r2: float, deployment_rec: str):
        """Register best model to MLflow Model Registry"""
        if not self.enabled or not run_id:
            return
        
        try:
            model_artifact_path = f"{target}_{model_name}_model"
            model_uri = f"runs:/{run_id}/{model_artifact_path}"
            registered_model_name = f"{target}_predictor"
            
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )
            
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=model_version.version,
                description=f"Best model for {target} prediction. Model type: {model_name}. Test R2: {test_r2:.4f}"
            )
            
            if deployment_rec == "production_ready":
                client.set_registered_model_alias(registered_model_name, "production", model_version.version)
            elif deployment_rec == "use_with_caution":
                client.set_registered_model_alias(registered_model_name, "staging", model_version.version)
            else:
                client.set_registered_model_alias(registered_model_name, "development", model_version.version)
            
            print(f"  [MLflow] Registered model: {registered_model_name} (version {model_version.version})")
            print(f"  [MLflow] Alias: {deployment_rec}")
            
        except Exception as e:
            print(f"  [MLflow] Model registration failed: {e}")


# ============================================
# Model Trainers (keeping same as before)
# ============================================

class BaseModelTrainer:
    """Base class for all model trainers"""
    
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.target_col = f"target_{target_name}"
        self.model = None
        self.feature_names = None
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.cv_scores = None
        self.cv_mean = None
        self.cv_std = None
        self.run_id = None
    
    def load_and_prepare_data(self, splits_dir: str):
        """Load and prepare data (common for all models)"""
        splits_path = Path(splits_dir)
        
        train_df = pd.read_csv(splits_path / "train_data.csv")
        val_df = pd.read_csv(splits_path / "val_data.csv")
        test_df = pd.read_csv(splits_path / "test_data.csv")
        
        # Prepare features
        X_train, y_train = get_feature_target_split(train_df, self.target_col, encode_categoricals=True)
        X_val, y_val = get_feature_target_split(val_df, self.target_col, encode_categoricals=True)
        X_test, y_test = get_feature_target_split(test_df, self.target_col, encode_categoricals=True)
        
        # Align columns
        train_cols = set(X_train.columns)
        for col in train_cols:
            if col not in X_val.columns:
                X_val[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0
        
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        
        # Impute missing values
        if X_train.isna().sum().sum() > 0:
            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Drop NaN targets
        X_train, y_train = drop_nan_targets(X_train, y_train, "Train")
        X_val, y_val = drop_nan_targets(X_val, y_val, "Val")
        X_test, y_test = drop_nan_targets(X_test, y_test, "Test")
        
        self.feature_names = X_train.columns.tolist()
        
        # Store original dataframes for bias detection
        return X_train, y_train, X_val, y_val, X_test, y_test, test_df
    
    def evaluate_all_splits(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate model on all splits"""
        results = {}
        
        for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = self.predict(X)
            
            results[name] = {
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "mae": float(mean_absolute_error(y, pred)),
                "r2": float(r2_score(y, pred))
            }
        
        self.train_metrics = results["train"]
        self.val_metrics = results["val"]
        self.test_metrics = results["test"]
        
        return results
    
    def predict(self, X):
        """Prediction method (to be implemented by subclasses)"""
        raise NotImplementedError


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost baseline trainer"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "xgboost"
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost baseline"""
        params = {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        
        self.model = xgb.XGBRegressor(**params)
        self.model.set_params(early_stopping_rounds=50)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.cv_mean = None
        self.cv_std = None
        self.cv_scores = None
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class XGBoostTunedTrainer(BaseModelTrainer):
    """XGBoost with Optuna tuning using TIME SERIES CV"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "xgboost_tuned"
        self.best_params = None
    
    def train(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Train XGBoost with CV-BASED hyperparameter tuning"""
        
        X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
        print(f"   Using Time Series CV for hyperparameter tuning...")
        
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "verbosity": 0,
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_trainval):
                X_tr = X_trainval.iloc[train_idx]
                X_vl = X_trainval.iloc[val_idx]
                y_tr = y_trainval.iloc[train_idx]
                y_vl = y_trainval.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params)
                
                try:
                    model.fit(X_tr, y_tr)
                    val_pred = model.predict(X_vl)
                    fold_r2 = r2_score(y_vl, val_pred)
                    cv_scores.append(fold_r2)
                except Exception:
                    raise optuna.TrialPruned()
            
            avg_r2 = np.mean(cv_scores)
            return -avg_r2
        
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.cv_mean = -study.best_value
        
        tscv = TimeSeriesSplit(n_splits=3)
        best_cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_trainval):
            X_tr = X_trainval.iloc[train_idx]
            X_vl = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx]
            y_vl = y_trainval.iloc[val_idx]
            
            model = xgb.XGBRegressor(**self.best_params, random_state=42, 
                                     n_jobs=-1, tree_method="hist", verbosity=0)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_vl)
            best_cv_scores.append(r2_score(y_vl, val_pred))
        
        self.cv_scores = np.array(best_cv_scores)
        self.cv_std = np.std(best_cv_scores)
        
        print(f"   Best CV Score: {self.cv_mean:.4f} (+/- {self.cv_std:.4f}) from {n_trials} trials")
        
        self.model = xgb.XGBRegressor(**self.best_params, random_state=42, 
                                      n_jobs=-1, tree_method="hist", verbosity=0)
        self.model.fit(X_trainval, y_trainval)
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class LightGBMTrainer(BaseModelTrainer):
    """LightGBM baseline trainer"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "lightgbm"
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM baseline"""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=500,
            callbacks=callbacks
        )
        
        self.cv_mean = None
        self.cv_std = None
        self.cv_scores = None
        
        return self
    
    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class LightGBMTunedTrainer(BaseModelTrainer):
    """LightGBM with Optuna tuning using TIME SERIES CV"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "lightgbm_tuned"
        self.best_params = None
    
    def train(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Train LightGBM with CV-BASED hyperparameter tuning"""
        
        X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
        print(f"   Using Time Series CV for hyperparameter tuning...")
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_trainval):
                X_tr = X_trainval.iloc[train_idx]
                X_vl = X_trainval.iloc[val_idx]
                y_tr = y_trainval.iloc[train_idx]
                y_vl = y_trainval.iloc[val_idx]
                
                model = lgb.LGBMRegressor(**params)
                
                try:
                    model.fit(X_tr, y_tr)
                    val_pred = model.predict(X_vl)
                    fold_r2 = r2_score(y_vl, val_pred)
                    cv_scores.append(fold_r2)
                except Exception:
                    raise optuna.TrialPruned()
            
            avg_r2 = np.mean(cv_scores)
            return -avg_r2
        
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.cv_mean = -study.best_value
        
        tscv = TimeSeriesSplit(n_splits=3)
        best_cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_trainval):
            X_tr = X_trainval.iloc[train_idx]
            X_vl = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx]
            y_vl = y_trainval.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**self.best_params, random_state=42, 
                                     n_jobs=-1, verbose=-1)
            model.fit(X_tr, y_tr)
            val_pred = model.predict(X_vl)
            best_cv_scores.append(r2_score(y_vl, val_pred))
        
        self.cv_scores = np.array(best_cv_scores)
        self.cv_std = np.std(best_cv_scores)
        
        print(f"   Best CV Score: {self.cv_mean:.4f} (+/- {self.cv_std:.4f}) from {n_trials} trials")
        
        self.model = lgb.LGBMRegressor(**self.best_params, random_state=42, 
                                      n_jobs=-1, verbose=-1)
        self.model.fit(X_trainval, y_trainval)
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)


# ============================================
# Unified Training Pipeline (ENHANCED with Bias Detection)
# ============================================

class UnifiedModelPipeline:
    """
    Unified pipeline with BIAS DETECTION
    """
    
    def __init__(self, splits_dir: str = "data/splits", output_dir: str = "models/best_models"):
        self.splits_dir = splits_dir
        self.output_dir = output_dir
        self.results = {}
        self.best_models = {}
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.visualizer = ModelVisualizer(output_dir)
        self.mlflow_tracker = MLflowTracker()
        self.bias_detector = BiasDetector(self.visualizer)  # NEW!
    
    def train_all_models_for_target(self, target: str, n_trials: int = 30, skip_tuning: bool = False):
        """Train all 4 model types for a single target"""
        print(f"\n{'='*80}")
        print(f"TRAINING ALL MODELS: {target.upper()}")
        print(f"{'='*80}\n")
        
        target_results = {}
        
        trainers = [
            ("XGBoost", XGBoostTrainer(target)),
            ("LightGBM", LightGBMTrainer(target)),
        ]
        
        if not skip_tuning:
            trainers.extend([
                ("XGBoost-Tuned", XGBoostTunedTrainer(target)),
                ("LightGBM-Tuned", LightGBMTunedTrainer(target)),
            ])
        
        print(f"Loading data...")
        trainer = trainers[0][1]
        X_train, y_train, X_val, y_val, X_test, y_test, test_df = trainer.load_and_prepare_data(self.splits_dir)
        print(f"   Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        print(f"   Features: {len(X_train.columns)}")
        
        for model_name, trainer in trainers:
            print(f"\n{'─'*80}")
            print(f"Training: {model_name}")
            print(f"{'─'*80}")
            
            try:
                # Load data
                X_train, y_train, X_val, y_val, X_test, y_test, test_df = trainer.load_and_prepare_data(self.splits_dir)
                
                # TRAIN
                start_time = pd.Timestamp.now()
                if "Tuned" in model_name:
                    trainer.train(X_train, y_train, X_val, y_val, n_trials=n_trials)
                else:
                    trainer.train(X_train, y_train, X_val, y_val)
                
                training_time = (pd.Timestamp.now() - start_time).total_seconds()
                
                # EVALUATE
                results = trainer.evaluate_all_splits(X_train, y_train, X_val, y_val, X_test, y_test)
                
                # ════════════════════════════════════════════════════
                # BIAS DETECTION (NEW!)
                # ════════════════════════════════════════════════════
                bias_report = self.bias_detector.comprehensive_bias_report(
                    model=trainer.model,
                    X_test=X_test,
                    y_test=y_test,
                    test_df=test_df,
                    target=target,
                    model_name=model_name
                )
                
                # GENERATE VISUALIZATIONS
                print("   Generating visualizations...")
                viz_files = {}
                
                y_pred_test = trainer.predict(X_test)
                viz_files['pred_vs_actual'] = self.visualizer.plot_predictions_vs_actual(
                    y_test, y_pred_test, 
                    f"{target} - {model_name}",
                    f"{target}_{model_name.replace('-', '_')}_pred_vs_actual.png"
                )
                
                viz_files['residuals'] = self.visualizer.plot_residual_analysis(
                    y_test, y_pred_test,
                    f"{target} - {model_name}",
                    f"{target}_{model_name.replace('-', '_')}_residuals.png"
                )
                
                viz_files['feature_importance'] = self.visualizer.plot_feature_importance(
                    trainer.model, trainer.feature_names,
                    f"{target} - {model_name}",
                    f"{target}_{model_name.replace('-', '_')}_features.png"
                )
                
                if trainer.cv_scores is not None:
                    viz_files['cv_scores'] = self.visualizer.plot_cv_scores(
                        trainer.cv_scores,
                        f"{target} - {model_name}",
                        f"{target}_{model_name.replace('-', '_')}_cv.png"
                    )
                
                # Add bias visualization if available
                if bias_report and 'sector_bias_viz' in bias_report:
                    viz_files['sector_bias'] = bias_report['sector_bias_viz']
                
                # STORE RESULTS
                target_results[model_name] = {
                    "trainer": trainer,
                    "results": results,
                    "training_time": training_time,
                    "model_type": trainer.model_type,
                    "cv_mean": trainer.cv_mean,
                    "cv_std": trainer.cv_std,
                    "cv_scores": trainer.cv_scores,
                    "bias_report": bias_report,  # NEW!
                    "visualizations": viz_files
                }
                
                # LOG TO MLFLOW
                params = {
                    "model_type": model_name, 
                    "target": target,
                    "n_features": len(trainer.feature_names)
                }
                if hasattr(trainer, 'best_params') and trainer.best_params:
                    params.update(trainer.best_params)
                
                metrics = {
                    "train_r2": results['train']['r2'],
                    "val_r2": results['val']['r2'],
                    "test_r2": results['test']['r2'],
                    "test_rmse": results['test']['rmse'],
                    "test_mae": results['test']['mae'],
                    "training_time_sec": training_time
                }
                
                if trainer.cv_mean is not None:
                    metrics["cv_mean_r2"] = trainer.cv_mean
                    metrics["cv_std_r2"] = trainer.cv_std if trainer.cv_std else 0.0
                
                run_id = self.mlflow_tracker.log_experiment(
                    target=target,
                    model_name=model_name,
                    model_obj=trainer.model,
                    params=params,
                    metrics=metrics,
                    artifacts=viz_files,
                    bias_report=bias_report  # NEW!
                )
                
                trainer.run_id = run_id
                
                # PRINT SUMMARY
                print(f"   Train R2: {results['train']['r2']:.4f}")
                print(f"   Val R2:   {results['val']['r2']:.4f}")
                print(f"   Test R2:  {results['test']['r2']:.4f}")
                if trainer.cv_mean is not None:
                    print(f"   CV Mean:  {trainer.cv_mean:.4f} (+/- {trainer.cv_std:.4f})")
                print(f"   Time:     {training_time:.1f}s")
                
            except Exception as e:
                print(f"   ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return target_results
    
    def select_best_model(self, target: str, target_results: Dict):
        """Select best model"""
        print(f"\n{'='*80}")
        print(f"MODEL SELECTION: {target.upper()}")
        print(f"{'='*80}\n")
        
        comparison = []
        for model_name, data in target_results.items():
            results = data["results"]
            
            train_r2 = results["train"]["r2"]
            test_r2 = results["test"]["r2"]
            overfit_gap = train_r2 - test_r2
            overfit_pct = (overfit_gap / train_r2 * 100) if train_r2 > 0 else 0
            
            # Add bias information
            bias_report = data.get("bias_report", {})
            sector_bias = bias_report.get('sector_bias', {}).get('bias_detected', False) if bias_report else False
            
            comparison.append({
                "model": model_name,
                "test_r2": test_r2,
                "train_r2": train_r2,
                "overfit_gap": overfit_gap,
                "overfit_pct": overfit_pct,
                "test_rmse": results["test"]["rmse"],
                "trainer": data["trainer"],
                "cv_mean": data.get("cv_mean"),
                "cv_std": data.get("cv_std", 0.0),
                "sector_bias": sector_bias
            })
        
        comparison_sorted = sorted(comparison, key=lambda x: x["test_r2"], reverse=True)
        
        # Print comparison table with bias column
        print(f"{'Model':<20} {'Test R2':>10} {'CV Mean':>10} {'Overfit %':>12} {'Bias':>8}")
        print(f"{'─'*70}")
        
        for item in comparison_sorted:
            overfit_symbol = "[OK]" if item["overfit_pct"] < 30 else "[WARN]"
            cv_str = f"{item['cv_mean']:.4f}" if item['cv_mean'] is not None else "N/A"
            bias_str = "[YES]" if item.get('sector_bias') else "[NO]"
            print(f"{item['model']:<20} {item['test_r2']:>10.4f} {cv_str:>10} "
                  f"{overfit_symbol} {item['overfit_pct']:>9.1f}% {bias_str:>8}")
        
        best = comparison_sorted[0]
        
        # Generate model comparison visualization
        print("\n   Generating model comparison chart...")
        comparison_viz = self.visualizer.plot_model_comparison(
            comparison_sorted, target,
            f"{target}_model_comparison.png"
        )
        
        if best["overfit_pct"] > 30:
            print(f"\nWARNING: Best model has {best['overfit_pct']:.1f}% overfitting!")
            
            alternatives = [c for c in comparison_sorted[1:] if c["overfit_pct"] < 30]
            if alternatives:
                alternative = alternatives[0]
                r2_sacrifice = best["test_r2"] - alternative["test_r2"]
                r2_sacrifice_pct = (r2_sacrifice / best["test_r2"]) * 100
                
                if r2_sacrifice_pct < 3.0:
                    print(f"   Switching to {alternative['model']} (better generalization)")
                    print(f"   R2 sacrifice: {r2_sacrifice:.4f} ({r2_sacrifice_pct:.1f}%)")
                    best = alternative
        
        print(f"\n{'─'*80}")
        print(f"SELECTED: {best['model']}")
        print(f"{'─'*80}")
        print(f"   Test R2: {best['test_r2']:.4f}")
        if best.get('cv_mean') is not None:
            print(f"   CV Mean: {best['cv_mean']:.4f} (+/- {best['cv_std']:.4f})")
        print(f"   Overfit: {best['overfit_pct']:.1f}%")
        if best.get('sector_bias'):
            print(f"   WARNING: Sector bias detected!")
        
        reasoning = f"Selected {best['model']} with Test R2={best['test_r2']:.4f}. "
        if best["overfit_pct"] < 20:
            reasoning += "Excellent generalization."
        elif best["overfit_pct"] < 30:
            reasoning += "Good generalization."
        else:
            reasoning += f"Warning: {best['overfit_pct']:.1f}% overfitting detected."
        
        if best.get('sector_bias'):
            reasoning += " Sector bias detected - monitor sector-specific performance."
        
        return best["model"], best["trainer"], reasoning, comparison_sorted
    
    def save_best_model(self, target: str, model_name: str, trainer: BaseModelTrainer, reasoning: str):
        """Save best model"""
        output_file = Path(self.output_dir) / f"{target}_best.pkl"
        
        test_r2 = trainer.test_metrics["r2"]
        if test_r2 >= 0.7:
            deploy_rec = "production_ready"
        elif test_r2 >= 0.5:
            deploy_rec = "use_with_caution"
        elif test_r2 >= 0.25:
            deploy_rec = "research_only"
        else:
            deploy_rec = "low_confidence"
        
        model_data = {
            "target": target,
            "model_type": model_name,
            "model": trainer.model,
            "feature_names": trainer.feature_names,
            "train_metrics": trainer.train_metrics,
            "val_metrics": trainer.val_metrics,
            "test_metrics": trainer.test_metrics,
            "cv_mean": trainer.cv_mean,
            "cv_std": trainer.cv_std,
            "cv_scores": trainer.cv_scores.tolist() if trainer.cv_scores is not None else None,
            "selection_reasoning": reasoning,
            "deployment_recommendation": deploy_rec,
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(trainer, 'best_params'):
            model_data["hyperparameters"] = trainer.best_params
        
        joblib.dump(model_data, output_file)
        
        print(f"\nModel saved: {output_file}")
        print(f"   Deployment: {deploy_rec}")
        
        if trainer.run_id:
            self.mlflow_tracker.register_best_model(
                target=target,
                model_name=model_name,
                run_id=trainer.run_id,
                test_r2=test_r2,
                deployment_rec=deploy_rec
            )
        
        return output_file
    
    def generate_comparison_report(self, all_results: Dict, output_dir: str = None):
        """Generate comprehensive comparison report with bias analysis"""
        if output_dir is None:
            output_dir = self.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"GENERATING COMPARISON REPORT")
        print(f"{'='*80}\n")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_targets": len(all_results),
            "methodology": "Time Series CV + Temporal Split + Bias Detection",
            "targets": {}
        }
        
        summary_table = []
        deployment_ready = []
        
        for target, data in all_results.items():
            best_model_name = data["best_model"]
            best_trainer = data["best_trainer"]
            comparison = data["comparison"]
            
            test_r2 = best_trainer.test_metrics["r2"]
            cv_mean = best_trainer.cv_mean
            cv_std = best_trainer.cv_std
            
            if test_r2 >= 0.7:
                deploy_status = "[PROD] Production Ready"
                deployment_ready.append(target)
            elif test_r2 >= 0.5:
                deploy_status = "[WARN] Use with Caution"
            elif test_r2 >= 0.25:
                deploy_status = "[RESEARCH] Research Only"
            else:
                deploy_status = "[LOW] Low Confidence"
            
            report["targets"][target] = {
                "selected_model": best_model_name,
                "selection_reasoning": data["reasoning"],
                "test_r2": test_r2,
                "test_rmse": best_trainer.test_metrics["rmse"],
                "test_mae": best_trainer.test_metrics["mae"],
                "cv_mean": float(cv_mean) if cv_mean is not None else None,
                "cv_std": float(cv_std) if cv_std is not None else None,
                "deployment_status": deploy_status,
                "mlflow_run_id": best_trainer.run_id,
                "all_models": [
                    {
                        "model": c["model"],
                        "test_r2": c["test_r2"],
                        "overfit_pct": c["overfit_pct"],
                        "cv_mean": c.get("cv_mean"),
                        "cv_std": c.get("cv_std"),
                        "sector_bias": c.get("sector_bias", False)
                    }
                    for c in comparison
                ]
            }
            
            summary_table.append({
                "target": target,
                "model": best_model_name,
                "test_r2": test_r2,
                "test_rmse": best_trainer.test_metrics["rmse"],
                "cv_mean": cv_mean,
                "deploy_status": deploy_status
            })
        
        report_file = output_path / "model_comparison_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   JSON report: {report_file}")
        
        print(f"\n{'='*80}")
        print(f"FINAL MODEL SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Target':<20} {'Selected Model':<20} {'Test R2':>10} {'CV Mean':>10} {'Status':<30}")
        print(f"{'─'*95}")
        
        for row in summary_table:
            cv_str = f"{row['cv_mean']:.4f}" if row['cv_mean'] is not None else "N/A"
            print(f"{row['target']:<20} {row['model']:<20} {row['test_r2']:>10.4f} "
                  f"{cv_str:>10} {row['deploy_status']:<30}")
        
        avg_r2 = np.mean([r["test_r2"] for r in summary_table])
        print(f"{'─'*95}")
        print(f"{'AVERAGE':<20} {'':20} {avg_r2:>10.4f}")
        
        print(f"\n{'='*80}")
        print(f"DEPLOYMENT RECOMMENDATION")
        print(f"{'='*80}")
        print(f"Production Ready: {', '.join(deployment_ready) if deployment_ready else 'None'}")
        print(f"Total Models: {len(summary_table)}")
        
        print(f"\n{'='*80}")
        print(f"VISUALIZATIONS GENERATED")
        print(f"{'='*80}")
        print(f"Location: {self.visualizer.viz_dir}")
        print(f"Files per model:")
        print(f"  - predictions_vs_actual.png")
        print(f"  - residual_analysis.png")
        print(f"  - feature_importance.png")
        print(f"  - cv_scores.png (for tuned models)")
        print(f"  - sector_bias.png (bias analysis)")
        print(f"  - model_comparison.png (per target)")
        
        return report_file
    
    def run_pipeline(self, targets: List[str], n_trials: int = 30, skip_tuning: bool = False):
        """Run complete pipeline for all targets"""
        print(f"\n{'='*80}")
        print(f"UNIFIED MODEL TRAINING PIPELINE WITH BIAS DETECTION")
        print(f"{'='*80}")
        print(f"Targets: {', '.join(targets)}")
        print(f"Models: XGBoost, LightGBM" + ("" if skip_tuning else ", XGBoost-Tuned, LightGBM-Tuned"))
        print(f"Tuning trials: {n_trials if not skip_tuning else 'N/A (skipped)'}")
        print(f"Features: Time Series CV + Visualizations + MLflow + Bias Detection")
        print(f"{'='*80}\n")
        
        all_results = {}
        
        for i, target in enumerate(targets, 1):
            print(f"\n{'#'*80}")
            print(f"TARGET {i}/{len(targets)}: {target.upper()}")
            print(f"{'#'*80}")
            
            try:
                target_results = self.train_all_models_for_target(target, n_trials, skip_tuning)
                best_model_name, best_trainer, reasoning, comparison = self.select_best_model(target, target_results)
                model_file = self.save_best_model(target, best_model_name, best_trainer, reasoning)
                
                all_results[target] = {
                    "best_model": best_model_name,
                    "best_trainer": best_trainer,
                    "reasoning": reasoning,
                    "comparison": comparison,
                    "model_file": str(model_file),
                    "all_models": target_results
                }
                
                print(f"\n{target.upper()} COMPLETE")
                
            except Exception as e:
                print(f"\nERROR processing {target}:")
                print(f"   {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_results:
            self.generate_comparison_report(all_results)
        
        return all_results


# ============================================
# CLI Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified model training with CV, visualizations, MLflow, and bias detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--target", type=str, nargs="+", default=["all"])
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="models/best_models")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    
    all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]
    
    if "all" in args.target:
        targets = all_targets
    else:
        targets = [t for t in args.target if t in all_targets]
        if not targets:
            print(f"Invalid target(s): {args.target}")
            return
    
    pipeline = UnifiedModelPipeline(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir
    )
    
    results = pipeline.run_pipeline(
        targets=targets,
        n_trials=args.trials,
        skip_tuning=args.quick
    )
    
    if results:
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"\nBest models saved to: {args.output_dir}/")
        
        if MLFLOW_AVAILABLE:
            print(f"\nMLflow:")
            print(f"   - Experiments logged with bias detection")
            print(f"   - View at: http://localhost:5000")
            print(f"   - Bias reports in Artifacts tab")
        
        print(f"\nVisualizations include bias analysis:")
        print(f"   - Sector performance comparison")
        print(f"   - Company size bias analysis")  
        print(f"   - Temporal bias detection")
    else:
        print(f"\nNo models trained successfully")


if __name__ == "__main__":
    main()