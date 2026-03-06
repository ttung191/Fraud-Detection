# register_aml_deployment_package.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import os
# Import config từ src để lấy tên model đã định nghĩa
try:
    from src import config 
except ImportError:
    print("ERROR: Cannot import config from src. Make sure you are running from project root and src has __init__.py")
    exit()

# --- Cấu hình ---
model_package_local_path = "./aml_deploy_package/"
# Sử dụng một tên riêng cho deployment package, định nghĩa trong config.py hoặc ở đây
model_registry_name = getattr(config, 'AZUREML_DEPLOYMENT_PACKAGE_NAME', 'fraud-detection-deploy-pkg') # Dùng getattr để có giá trị mặc định
model_version = "1" # << TĂNG SỐ NÀY MỖI KHI CẬP NHẬT PACKAGE

try:
    print("Connecting to Azure ML Workspace...")
    # Sử dụng DefaultAzureCredential kết hợp với config.json hoặc biến môi trường
    ml_client = MLClient.from_config(credential=DefaultAzureCredential(), logging_enable=False)
    print(f"Connected to workspace: {ml_client.workspace_name}")

    print(f"\nRegistering model '{model_registry_name}' version '{model_version}'")
    print(f"Using local package path: '{os.path.abspath(model_package_local_path)}'")

    if not os.path.isdir(model_package_local_path):
        raise FileNotFoundError(f"Model package directory not found at expected path: {model_package_local_path}")
    if not os.path.exists(os.path.join(model_package_local_path, "score.py")):
         print(f"WARNING: score.py not found in {model_package_local_path}. Deployment might fail.")
    if not os.path.exists(os.path.join(model_package_local_path, config.MODEL_FILENAME)):
         print(f"WARNING: Model file {config.MODEL_FILENAME} not found in {model_package_local_path}. Deployment might fail.")


    # Đăng ký thư mục package làm Custom Model
    # Azure ML sẽ tự động tải nội dung thư mục này lên khi tạo Model Asset
    model_asset = Model(
        path=model_package_local_path, # Đường dẫn tới thư mục cục bộ
        name=model_registry_name,
        version=model_version,
        description=f"Deployable fraud detection package (v{model_version}) containing model, score.py, and dependencies.",
        type=AssetTypes.CUSTOM_MODEL # Sử dụng CUSTOM_MODEL vì chúng ta có score.py tùy chỉnh
        # Nếu bạn chỉ đăng ký thư mục model MLflow (không có score.py), bạn có thể dùng MLFLOW_MODEL
    )

    registered_model = ml_client.models.create_or_update(model_asset)
    print(f"\nSUCCESS: Model registered in Azure ML:")
    print(f"  Name:    '{registered_model.name}'")
    print(f"  Version: '{registered_model.version}'")
    print(f"  ID:      '{registered_model.id}'")

except FileNotFoundError as fnf_err:
    print(f"\nERROR: {fnf_err}")
    print("Please ensure you have created the 'aml_deploy_package/' directory and copied all necessary files (score.py, model, preprocessors, src_code/, features list) into it BEFORE running this script.")
except Exception as e:
    print(f"\nAn error occurred during model registration: {e}")
    import traceback
    traceback.print_exc()