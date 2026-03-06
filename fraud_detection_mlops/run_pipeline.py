# run_pipeline.py (Phiên bản điều chỉnh cho Azure App Service)
import pandas as pd
import os
import json
from datetime import datetime
import mlflow

try:
    from src import config
    from src import utils
    from src import data_preprocessing
    from src import feature_engineering
    from src import train
except ImportError as e:
    print(f"Lỗi import module từ src/: {e}")
    print("Đảm bảo bạn đang chạy script này từ thư mục gốc của project (fraud_ml_ops_project/)")
    print("Và thư mục src/ có file __init__.py")
    exit()

# BỎ QUA HOẶC XÓA BỎ HOÀN TOÀN PHẦN IMPORT AZURE ML SDK NẾU KHÔNG DÙNG
# try:
#     from azure.ai.ml import MLClient
#     from azure.ai.ml.entities import Model
#     from azure.ai.ml.constants import AssetTypes
#     from azure.identity import DefaultAzureCredential
#     from azure.core.exceptions import AzureError
#     AZURE_SDK_AVAILABLE = True
# except ImportError:
#     print("WARNING: Thư viện azure-ai-ml hoặc azure-identity chưa được cài đặt.")
#     print("Các bước liên quan đến đăng ký model lên Azure ML sẽ bị bỏ qua.")
#     AZURE_SDK_AVAILABLE = False
#     MLClient = None


def main_training_pipeline():
    print(f"--- Starting Training Artifacts Generation Pipeline at {datetime.now()} ---")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    try:
        # === PHASE 1: Load and Merge Raw Data ===
        print("\n[PHASE 1/5] Loading and Merging Raw Data...") # Đánh số lại phase
        df_train_raw = data_preprocessing.load_and_merge_data(
            config.TRAIN_TRANSACTION_PATH,
            config.TRAIN_IDENTITY_PATH
        )
        print(f"  Raw training data shape: {df_train_raw.shape}")

        # === PHASE 2: Base Preprocessing (fit and save LabelEncoders) ===
        print("\n[PHASE 2/5] Running Base Preprocessing...")
        df_processed = data_preprocessing.run_base_preprocessing(
            df_train_raw,
            fit_encoders_flag=True
        )
        print(f"  Data shape after base preprocessing: {df_processed.shape}")

        # === PHASE 3: Feature Engineering (fit and save Scaler & PCA, select V_cols) ===
        print("\n[PHASE 3/5] Running Feature Engineering...")
        df_featured, selected_v_cols_for_training = feature_engineering.run_all_feature_engineering(
            df_processed.copy(),
            fit_transformers_flag=True,
            df_for_corr_fitting=df_processed.copy()
        )
        print(f"  Data shape after feature engineering: {df_featured.shape}")
        if selected_v_cols_for_training:
            print(f"  Selected {len(selected_v_cols_for_training)} original V-columns based on correlation for potential filtering.")

        # === PHASE 4: Final NA Value Imputation & Memory Reduction & Save ===
        print("\n[PHASE 4/5] Final NA Value Imputation, Memory Reduction & Save...")
        df_final_for_training = df_featured.fillna(-999)
        df_final_for_training = utils.reduce_memory_usage(df_final_for_training)
        print(f"  Final DataFrame shape for training: {df_final_for_training.shape}")

        processed_file_path = os.path.join(config.PROCESSED_DATA_DIR, "final_processed_training_data.parquet")
        try:
            df_final_for_training.to_parquet(processed_file_path, index=False)
            print(f"  Final processed data for training saved to {processed_file_path}")
        except Exception as e_save:
            print(f"  Could not save final processed data to parquet: {e_save}. Attempting CSV.")
            csv_path = os.path.splitext(processed_file_path)[0] + ".csv"
            try:
                df_final_for_training.to_csv(csv_path, index=False)
                print(f"  Final processed data for training saved to {csv_path}")
            except Exception as e_csv_save:
                 print(f"  Could not save final processed data to CSV either: {e_csv_save}")

        # === PHASE 5: Model Training (Logs to LOCAL ./mlruns & Saves Artifacts Locally) ===
        print("\n[PHASE 5/5] Training Model, Logging Locally to MLflow, and Saving Artifacts...")
        model_path_local, info_path_local, local_mlflow_run_id = train.train_lgbm_model(
            df_final_for_training,
            selected_v_cols_for_filter=selected_v_cols_for_training
        )
        
        if local_mlflow_run_id: # Vẫn hữu ích để in ra run_id cục bộ
            print(f"  Local MLflow Run ID: {local_mlflow_run_id}")
        else:
            print("WARNING: Local MLflow run ID was not returned from training.")

        print(f"  Model saved locally: {model_path_local}") # Quan trọng cho Docker build
        print(f"  Info saved locally: {info_path_local}") # Quan trọng cho Docker build
        print(f"  LabelEncoders, Scaler, PCA, Final Features List also saved locally by train/preprocessing/feature_engineering steps.")

        # === PHASE 6 ĐÃ BỊ LOẠI BỎ HOẶC COMMENT OUT ===
        # print("\n[PHASE 6/6] Azure ML Model Registration steps are now skipped for App Service deployment.")

        print(f"\n--- Training Artifacts Generation Pipeline Finished Successfully at {datetime.now()} ---")
        print("--- Next steps: Build Docker image, push to ACR, and deploy to Azure App Service. ---")

    except FileNotFoundError as e_fnf:
         print(f"\nPIPELINE ERROR: File not found. Please check paths in config.py and ensure data files exist in data/raw/.")
         print(f"Details: {e_fnf}")
    except ImportError as e_imp:
         print(f"\nPIPELINE ERROR: Missing library. Please check requirements.txt and your virtual environment.")
         print(f"Details: {e_imp}")
    except Exception as e_main:
         print(f"\nAn unexpected error occurred in the main pipeline:")
         import traceback
         traceback.print_exc()

if __name__ == "__main__":
    start_time = datetime.now()
    main_training_pipeline()
    end_time = datetime.now()
    print(f"\nTotal pipeline execution time: {end_time - start_time}")