# create_aml_scoring_env.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment as AzureMLEnvironment
from azure.identity import DefaultAzureCredential
import os

# --- Cấu hình ---
environment_name = "fraud-scoring-environment" # Tên môi trường trên Azure ML
environment_version = "1" # Bắt đầu với version 1, tăng lên nếu environment.yml thay đổi
conda_file_path = "./env/environment_scoring.yml" # Đường dẫn tới file YAML

try:
    print("Connecting to Azure ML Workspace...")
    ml_client = MLClient.from_config(credential=DefaultAzureCredential(), logging_enable=False)
    print(f"Connected to workspace: {ml_client.workspace_name}")

    print(f"\nCreating/Updating environment '{environment_name}' version '{environment_version}' from {conda_file_path}...")

    if not os.path.exists(conda_file_path):
         raise FileNotFoundError(f"Conda environment file not found at: {conda_file_path}")

    scoring_env = AzureMLEnvironment(
        name=environment_name,
        version=environment_version,
        description="Environment with necessary packages for fraud detection model scoring.",
        conda_file=conda_file_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest" # Base image chuẩn của Azure ML
        # Bạn cũng có thể dùng image từ ACR nếu muốn: image = "<yourACRName>.azurecr.io/..."
    )

    # Tạo hoặc cập nhật environment
    ml_client.environments.create_or_update(scoring_env)

    print(f"\nSUCCESS: Environment '{environment_name}' version '{environment_version}' created/updated in Azure ML.")
    print("You can now see this environment in Azure ML Studio under 'Environments'.")

except FileNotFoundError as fnf_err:
    print(f"ERROR: {fnf_err}")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()