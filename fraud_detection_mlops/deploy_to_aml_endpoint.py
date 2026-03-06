# deploy_to_aml_endpoint.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Model as AzureMLModel,
    Environment as AzureMLEnvironment
)
# QUAN TRỌNG: Thử dùng AzureCliCredential nếu DefaultAzureCredential không ổn định
# from azure.identity import DefaultAzureCredential
from azure.identity import AzureCliCredential # << THỬ DÙNG CÁI NÀY
from datetime import datetime
import os
import time

# --- Cấu hình ---
try:
    from src import config # Giả sử src/config.py đã được tạo và có biến cần thiết
    # Sử dụng tên của DEPLOYMENT PACKAGE MODEL từ src/config.py
    MODEL_NAME_FOR_DEPLOYMENT = config.AZUREML_DEPLOYMENT_PACKAGE_NAME # << PHẢI KHỚP VỚI TÊN PACKAGE
    print(f"INFO: Using deployment package model name from src.config: {MODEL_NAME_FOR_DEPLOYMENT}")
except ImportError:
    print("WARNING: Cannot import src.config. Using default name for deployment package model.")
    # Đặt tên mặc định cho deployment package nếu không import được src.config
    # Đảm bảo tên này khớp với tên bạn đã sử dụng trong register_aml_deployment_package.py
    MODEL_NAME_FOR_DEPLOYMENT = "fraud-detection-deploy-pkg" # << TÊN PACKAGE CỦA BẠN

ENDPOINT_NAME = os.getenv("AML_ENDPOINT_NAME", "fraud-endpoint-" + datetime.now().strftime("%m%d%H%M%S"))
# QUAN TRỌNG: Phiên bản này phải là phiên bản MỚI NHẤT của DEPLOYMENT PACKAGE MODEL
# mà bạn đã đăng ký bằng register_aml_deployment_package.py
MODEL_VERSION_FOR_DEPLOYMENT = os.getenv("AML_DEPLOYMENT_MODEL_VERSION", "1") # << CẬP NHẬT PHIÊN BẢN CHO ĐÚNG!

ENVIRONMENT_NAME_FOR_DEPLOYMENT = os.getenv("AML_ENVIRONMENT_NAME", "fraud-scoring-environment")
# QUAN TRỌNG: Cập nhật phiên bản environment nếu có bản mới hơn
ENVIRONMENT_VERSION_FOR_DEPLOYMENT = os.getenv("AML_ENVIRONMENT_VERSION", "1") # << CẬP NHẬT PHIÊN BẢN CHO ĐÚNG!

DEPLOYMENT_NAME = "blue"
INSTANCE_TYPE = "Standard_F4s_v2"
INSTANCE_COUNT = 1

try:
    print("Connecting to Azure ML Workspace...")
    # Sử dụng AzureCliCredential để đảm bảo lấy đúng context từ `az login`
    ml_client = MLClient.from_config(credential=AzureCliCredential(), logging_enable=False)
    print(f"Connected to workspace: {ml_client.workspace_name}")

    # Lấy location của workspace để sử dụng cho endpoint
    workspace_details = ml_client.workspaces.get(name=ml_client.workspace_name)
    workspace_location = workspace_details.location
    print(f"INFO: Workspace location: {workspace_location}")

    # --- 1. Tạo hoặc lấy Endpoint ---
    print(f"\nChecking/Creating Endpoint '{ENDPOINT_NAME}'...")
    try:
        endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
        print(f"Endpoint '{ENDPOINT_NAME}' already exists.")
    except Exception as e_get_endpoint:
        print(f"Endpoint '{ENDPOINT_NAME}' not found. Creating new endpoint...")
        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            description="Online endpoint for fraud detection model.",
            auth_mode="key",
            location=workspace_location  # << SỬ DỤNG LOCATION CỦA WORKSPACE
        )
        poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
        print("Waiting for endpoint creation...")
        poller.result() # Đây là dòng code gây ra lỗi trong traceback của bạn khi endpoint không tạo được
        print(f"Endpoint '{ENDPOINT_NAME}' created successfully.")
        endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)

    # --- 2. Lấy model và environment đã đăng ký ---
    print(f"\nFetching registered model for deployment: {MODEL_NAME_FOR_DEPLOYMENT}, version: {MODEL_VERSION_FOR_DEPLOYMENT}")
    model_asset = ml_client.models.get(name=MODEL_NAME_FOR_DEPLOYMENT, version=MODEL_VERSION_FOR_DEPLOYMENT)
    print(f"  Found Model ID for deployment: {model_asset.id}")

    print(f"Fetching registered environment: {ENVIRONMENT_NAME_FOR_DEPLOYMENT}, version: {ENVIRONMENT_VERSION_FOR_DEPLOYMENT}")
    inference_env = ml_client.environments.get(name=ENVIRONMENT_NAME_FOR_DEPLOYMENT, version=ENVIRONMENT_VERSION_FOR_DEPLOYMENT)
    print(f"  Found Environment ID: {inference_env.id}")

    # --- 3. Tạo hoặc cập nhật Deployment ---
    print(f"\nCreating/Updating Deployment '{DEPLOYMENT_NAME}' for endpoint '{ENDPOINT_NAME}'...")
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model_asset.id,
        environment=inference_env.id,
        code_configuration=CodeConfiguration(
             code=".",
             scoring_script="score.py"
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT
    )

    poller = ml_client.online_deployments.begin_create_or_update(deployment)
    print("Waiting for deployment creation/update...")
    status = poller.status()
    while status not in ["Succeeded", "Failed", "Canceled"]:
        print(f"  Deployment status: {status}")
        time.sleep(30)
        status = poller.status()

    if status == "Succeeded":
        print(f"Deployment '{DEPLOYMENT_NAME}' created/updated successfully.")
    else:
        print(f"Deployment '{DEPLOYMENT_NAME}' failed with status: {status}")
        print("Please check deployment logs in Azure ML Studio for more details.")
        raise Exception(f"Deployment failed with status {status}")

    # --- 4. Phân bổ traffic ---
    print(f"\nAllocating 100% traffic to deployment '{DEPLOYMENT_NAME}'...")
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    print("Waiting for traffic allocation update...")
    poller.result()
    print("Traffic allocation complete.")

    # --- 5. Hiển thị thông tin Endpoint ---
    final_endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    print(f"\nEndpoint '{final_endpoint.name}' provisioning state: {final_endpoint.provisioning_state}")
    print(f"Scoring URI: {final_endpoint.scoring_uri}")
    print("\nDeployment successful!")
    print("Use the 'Consume' tab in Azure ML Studio for this endpoint to get keys and test.")

except FileNotFoundError as fnf_err_deploy:
    print(f"\nERROR: {fnf_err_deploy}")
    print("Please ensure 'config.json' is correctly set up and you are logged into Azure ('az login').")
except Exception as e_deploy:
    print(f"\nAn error occurred during deployment: {e_deploy}")
    import traceback
    traceback.print_exc()