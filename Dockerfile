# Sử dụng một base image Python nhẹ và ổn định.
FROM python:3.11-slim

# Thiết lập các biến môi trường
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PORT=8080
# Dòng trên: Port mà ứng dụng Flask sẽ lắng nghe bên trong container

# Tạo thư mục làm việc
WORKDIR ${APP_HOME}

# === CÀI ĐẶT CÁC GÓI HỆ THỐNG CẦN THIẾT ===
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*
# === KẾT THÚC CÀI ĐẶT GÓI HỆ THỐNG ===

# Sao chép file requirements.txt vào trước để tận dụng Docker layer caching.
COPY requirements.txt .

# Cài đặt các thư viện Python.
# Thêm --default-timeout=300 nếu mạng chậm để tránh lỗi timeout khi pip install
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt --default-timeout=300

# Sao chép toàn bộ code của ứng dụng vào thư mục làm việc.
# Bao gồm src/, models/, artifacts/, và các file khác không bị ignore bởi .dockerignore
COPY . .

# Expose port mà ứng dụng Flask sẽ chạy (App Service sẽ sử dụng thông tin này).
EXPOSE ${PORT}

# Lệnh để chạy ứng dụng Flask khi container khởi động.
# Đảm bảo src/predict.py có thể chạy như một module và khởi động Flask app.
CMD ["python", "-m", "src.predict"]