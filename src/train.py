# src/train.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow # Cho MLflow Tracking
import mlflow.lightgbm # Cho logging model LightGBM tối ưu

from sklearn.metrics import (
    roc_auc_score, log_loss, confusion_matrix,
    classification_report, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

from . import config # Import config từ cùng package

# KHÔNG import azure.ai.ml hay azure.identity ở đây

def train_lgbm_model(df_processed, selected_v_cols_for_filter=None):
    """
    Huấn luyện model LightGBM, đánh giá, log vào MLflow CỤC BỘ (mlruns), 
    lưu model/artifacts cục bộ.
    Trả về: Đường dẫn model cục bộ, đường dẫn info cục bộ, và MLflow run ID cục bộ.
    """
    print("\n--- Model Training Phase (Logging locally to ./mlruns) ---")
    mlflow_run_id = None # Khởi tạo run_id

    # Đảm bảo MLflow log vào thư mục mlruns cục bộ trong thư mục gốc của project
    # config.MLRUNS_DIR nên được định nghĩa trong config.py là os.path.join(config.BASE_DIR, "mlruns")
    os.makedirs(config.MLRUNS_DIR, exist_ok=True) # Tạo thư mục nếu chưa có
    mlflow.set_tracking_uri(f"file:{os.path.abspath(config.MLRUNS_DIR)}") # Đường dẫn tuyệt đối đến mlruns
    
    # Đặt tên experiment cục bộ
    mlflow.set_experiment(config.LGBM_PARAMS.get('mlflow_experiment_name', "Fraud_Detection_Local_Runs"))

    with mlflow.start_run() as run: # Bắt đầu một MLflow run CỤC BỘ
        mlflow_run_id = run.info.run_id # Lấy run_id của run cục bộ này
        print(f"  MLflow Run ID (local): {mlflow_run_id}")
        mlflow.log_param("tracking_location", "local_mlruns")
        mlflow.log_param("mlflow_run_id", mlflow_run_id)


        # --- 2. Xử lý cột target và tách features/target ---
        valid_target_values = {0.0, 1.0, 0, 1}
        initial_rows = len(df_processed)
        df_train_candidate = df_processed[df_processed[config.TARGET_COL].isin(valid_target_values)].copy()
        rows_dropped = initial_rows - len(df_train_candidate)
        if rows_dropped > 0:
            print(f"  Dropped {rows_dropped} rows with invalid target values. Remaining: {len(df_train_candidate)}")

        if df_train_candidate[config.TARGET_COL].isnull().any():
            df_train_candidate.dropna(subset=[config.TARGET_COL], inplace=True)
        df_train_candidate[config.TARGET_COL] = df_train_candidate[config.TARGET_COL].astype(int)

        y = df_train_candidate[config.TARGET_COL]
        
        cols_to_exclude_from_features = [
            config.TARGET_COL, 'TransactionID', 'TransactionDT', 'TransactionDT_Orig', 
            'TransactionDT_numeric', 'uid', 'uid2', 'uid3', 
        ]
        
        X = df_train_candidate.drop(columns=[col for col in cols_to_exclude_from_features if col in df_train_candidate.columns], errors='ignore')
        
        # Đảm bảo tất cả các cột feature còn lại là số
        X = X.select_dtypes(include=np.number)
        final_features_list = X.columns.tolist() # Lấy danh sách feature ban đầu

        # Lọc các cột V nếu selected_v_cols_for_filter được cung cấp
        if selected_v_cols_for_filter and len(selected_v_cols_for_filter) > 0:
            print(f"  Initial feature count: {len(final_features_list)}. Applying V_cols selection.")
            # Giữ lại các feature không phải V gốc, V_PCA, V_mean, V_std
            non_v_original_features = [
                col for col in final_features_list 
                if not (col.startswith(config.V_COLS_PREFIX) and 
                        col not in ['V_mean', 'V_std'] and 
                        'V_PCA' not in col)
            ]
            # Chỉ lấy các cột V gốc đã chọn mà thực sự tồn tại trong X
            valid_selected_v_original_cols = [col for col in selected_v_cols_for_filter if col in X.columns]
            
            final_features_list = list(set(non_v_original_features + valid_selected_v_original_cols)) # Kết hợp và loại bỏ trùng lặp
            X = X[final_features_list] # Áp dụng bộ feature mới
            print(f"  Using {len(final_features_list)} features after V_cols selection.")
        else:
            print(f"  Using all {len(final_features_list)} available numeric features (no V_cols filter or filter empty).")
        
        mlflow.log_param("num_features_final", X.shape[1])

        # Lưu và log danh sách features cuối cùng
        features_path = os.path.join(config.ARTIFACTS_DIR, config.FINAL_TRAINING_FEATURES_FILENAME)
        os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
        with open(features_path, 'w') as f:
            json.dump(final_features_list, f, indent=4)
        print(f"  Final training features list ({len(final_features_list)}) saved locally to {features_path}")
        mlflow.log_artifact(features_path, "run_artifacts") # Log vào thư mục con của run

        # --- 3. Chia dữ liệu ---
        total_len = len(X)
        if total_len < 100: # Ngưỡng nhỏ tùy chỉnh, có thể đặt vào config
            print("  Warning: Dataset too small for robust 60/20/20 split. Using 70/15/15 train/valid/test split via train_test_split.")
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=config.LGBM_PARAMS.get('seed', 42), stratify=y if y.nunique() > 1 else None)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=config.LGBM_PARAMS.get('seed', 42), stratify=y_train_val if y_train_val.nunique() > 1 else None) # 0.15 / (1-0.15) ~ 0.1765 -> 15% valid
        else:
            part = total_len // 10
            train_idx_end = 6 * part
            valid_idx_start = 6 * part
            valid_idx_end = 8 * part
            test_idx_start = 8 * part 

            X_train, y_train = X.iloc[:train_idx_end], y.iloc[:train_idx_end]
            X_valid, y_valid = X.iloc[valid_idx_start:valid_idx_end], y.iloc[valid_idx_start:valid_idx_end]
            X_test, y_test = X.iloc[test_idx_start:], y.iloc[test_idx_start:]
                
        print(f"  Dataset shapes -> Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
        mlflow.log_param("train_rows", X_train.shape[0]); mlflow.log_param("valid_rows", X_valid.shape[0]); mlflow.log_param("test_rows", X_test.shape[0])

        # --- 4. Tạo LightGBM Datasets ---
        lgb_train_data = lgb.Dataset(X_train, label=y_train, feature_name=final_features_list)
        lgb_valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=final_features_list, reference=lgb_train_data)

        # --- 5. Huấn luyện model ---
        print("  Training LightGBM model...")
        mlflow.log_params(config.LGBM_PARAMS) # Log tham số model
        
        current_lgbm_params = config.LGBM_PARAMS.copy() # Tạo bản sao để không thay đổi dict gốc
        if 'mlflow_experiment_name' in current_lgbm_params: # Loại bỏ key không phải của LGBM
            del current_lgbm_params['mlflow_experiment_name']

        model = lgb.train(
            current_lgbm_params, # Sử dụng params đã lọc
            lgb_train_data,
            num_boost_round=current_lgbm_params.get('n_estimators', 500), # Lấy n_estimators từ params đã lọc
            valid_sets=[lgb_train_data, lgb_valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.LGBM_EARLY_STOPPING_ROUNDS, verbose=1),
                lgb.log_evaluation(period=config.LGBM_LOG_EVALUATION_PERIOD)
            ]
        )
        mlflow.log_param("best_iteration", model.best_iteration)
        
        # --- 6. Đánh giá trên tập Validation và log MLflow ---
        print("\n  --- Evaluating on Validation Set ---")
        valid_preds_proba = model.predict(X_valid, num_iteration=model.best_iteration)
        auc_valid = roc_auc_score(y_valid, valid_preds_proba); mlflow.log_metric("auc_valid", auc_valid)
        lloss_valid = log_loss(y_valid, valid_preds_proba); mlflow.log_metric("logloss_valid", lloss_valid)
        valid_preds_binary = (valid_preds_proba >= config.PREDICTION_THRESHOLD).astype(int)
        precision_valid = precision_score(y_valid, valid_preds_binary, zero_division=0); mlflow.log_metric("precision_valid", precision_valid)
        recall_valid = recall_score(y_valid, valid_preds_binary, zero_division=0); mlflow.log_metric("recall_valid", recall_valid)
        f1_valid = f1_score(y_valid, valid_preds_binary, zero_division=0); mlflow.log_metric("f1_valid", f1_valid)
        print(f"  Validation AUC: {auc_valid:.5f} | LogLoss: {lloss_valid:.5f} | Precision: {precision_valid:.4f} | Recall: {recall_valid:.4f} | F1: {f1_valid:.4f}")
        
        cm_valid = confusion_matrix(y_valid, valid_preds_binary)
        cm_valid_path = os.path.join(config.ARTIFACTS_DIR, "confusion_matrix_valid.png")
        try:
            plt.figure(figsize=(6, 4)); sns.heatmap(cm_valid, annot=True, fmt='d', cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Validation Confusion Matrix')
            plt.savefig(cm_valid_path); plt.close(); mlflow.log_artifact(cm_valid_path, "evaluation_plots")
            print(f"  Validation confusion matrix saved locally and logged to MLflow.")
        except Exception as plot_err:
            print(f"  Warning: Could not generate/save validation confusion matrix plot: {plot_err}")
        
        # --- 7. (Tùy chọn) Đánh giá trên tập Test và log MLflow ---
        auc_test, lloss_test, precision_test, recall_test, f1_test = None, None, None, None, None
        if not X_test.empty and not y_test.empty and y_test.nunique() > 1:
            print("\n  --- Evaluating on Test Set ---")
            test_preds_proba = model.predict(X_test, num_iteration=model.best_iteration)
            auc_test = roc_auc_score(y_test, test_preds_proba); mlflow.log_metric("auc_test", auc_test)
            lloss_test = log_loss(y_test, test_preds_proba); mlflow.log_metric("logloss_test", lloss_test)
            test_preds_binary = (test_preds_proba >= config.PREDICTION_THRESHOLD).astype(int)
            precision_test = precision_score(y_test, test_preds_binary, zero_division=0); mlflow.log_metric("precision_test", precision_test)
            recall_test = recall_score(y_test, test_preds_binary, zero_division=0); mlflow.log_metric("recall_test", recall_test)
            f1_test = f1_score(y_test, test_preds_binary, zero_division=0); mlflow.log_metric("f1_test", f1_test)
            print(f"  Test AUC: {auc_test:.5f} | LogLoss: {lloss_test:.5f} | Precision: {precision_test:.4f} | Recall: {recall_test:.4f} | F1: {f1_test:.4f}")
        else:
            print("\n  Test set is empty or has only one class; skipping test set evaluation.")

        # --- 8. Lưu model cục bộ ---
        os.makedirs(config.MODELS_DIR, exist_ok=True) 
        model_path_local = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)
        joblib.dump(model, model_path_local)
        print(f"\n  Model saved locally to: {model_path_local}")

        # --- 9. Log Model và Preprocessors vào MLflow Run CỤC BỘ ---
        print("  Logging model and preprocessors artifacts to LOCAL MLflow run...")
        try:
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="lgbm_model_mlflow" # Log vào thư mục con trong artifacts của run cục bộ
                # KHÔNG CÓ registered_model_name ở đây nữa
            )
            print("    Model artifact logged locally to MLflow run.")
        except Exception as log_model_err:
             print(f"    ERROR logging model artifact locally to MLflow: {log_model_err}")
        
        preprocessor_artifact_subfolder = "preprocessors"
        for preprocessor_filename in [config.LABEL_ENCODERS_FILENAME, config.SCALER_V_FILENAME, config.PCA_V_FILENAME]:
            local_file_path = os.path.join(config.MODELS_DIR, preprocessor_filename)
            if os.path.exists(local_file_path):
                mlflow.log_artifact(local_file_path, preprocessor_artifact_subfolder)
                print(f"    Logged {preprocessor_filename} to local MLflow run artifacts.")
            else:
                print(f"    Warning: Preprocessor file {local_file_path} not found for local MLflow logging.")
        
        # --- 10. Lưu và log model_info.json CỤC BỘ ---
        model_info = {
            "mlflow_run_id": mlflow_run_id, # Lưu lại run ID cục bộ
            "model_path_local": model_path_local,
            "training_features_path_local": features_path, # Đã lưu ở trên
            "lgbm_params": config.LGBM_PARAMS,
            "best_iteration": model.best_iteration,
            "num_features_trained_on": len(final_features_list),
            "auc_valid": round(auc_valid, 5), "log_loss_valid": round(lloss_valid, 5),
            "precision_valid": round(precision_valid, 5), "recall_valid": round(recall_valid, 5), "f1_score_valid": round(f1_valid, 5),
            "auc_test": round(auc_test, 5) if auc_test is not None else None,
            "log_loss_test": round(lloss_test, 5) if lloss_test is not None else None,
            "precision_test": round(precision_test, 5) if precision_test is not None else None,
            "recall_test": round(recall_test, 5) if recall_test is not None else None,
            "f1_score_test": round(f1_test, 5) if f1_test is not None else None,
            "training_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "training_timestamp_local": pd.Timestamp.now(tz='Asia/Ho_Chi_Minh').isoformat() # Ví dụ timezone VN
        }
        info_path_local = os.path.join(config.ARTIFACTS_DIR, config.MODEL_INFO_FILENAME)
        with open(info_path_local, "w") as f: json.dump(model_info, f, indent=4)
        mlflow.log_artifact(info_path_local, "run_info") # Log file này vào artifacts CỤC BỘ
        print(f"  Model info saved locally to: {info_path_local} and logged to MLflow.")

        # --- 11. Log Feature Importance Plot CỤC BỘ ---
        if hasattr(model, 'feature_importance'):
            try:
                importance_df = pd.DataFrame({
                    'feature': final_features_list,
                    'importance_gain': model.feature_importance(importance_type='gain'),
                }).sort_values(by='importance_gain', ascending=False)
                
                print("\n  Top 20 Feature Importances (by gain):"); print(importance_df.head(20))
                
                importance_plot_path = os.path.join(config.ARTIFACTS_DIR, "feature_importance.png")
                plt.figure(figsize=(12, max(10, len(importance_df.head(30)) * 0.35))) # Điều chỉnh chiều cao plot
                sns.barplot(x="importance_gain", y="feature", data=importance_df.head(30)) # Vẽ top 30
                plt.title("LightGBM Feature Importance (Top 30 by Gain)"); plt.tight_layout(); plt.savefig(importance_plot_path); plt.close()
                mlflow.log_artifact(importance_plot_path, "evaluation_plots") # Log vào artifacts CỤC BỘ
                print(f"  Feature importance plot saved locally and logged to MLflow.")
            except Exception as fi_err:
                 print(f"  Warning: Could not generate/save feature importance plot: {fi_err}")

    # Trả về các đường dẫn cục bộ và run_id CỤC BỘ
    return model_path_local, info_path_local, mlflow_run_id