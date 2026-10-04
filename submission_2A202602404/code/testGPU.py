"""testGPU.py — Script kiểm tra môi trường và khả năng chạy GPU (CUDA) của PyTorch."""

import sys
import time

# Đảm bảo in tiếng Việt có dấu không bị lỗi font trên Windows cmd / powershell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

def check_gpu():
    print("=" * 60)
    print("KIỂM TRA TRẠNG THÁI GPU / CUDA PYTORCH")
    print("=" * 60)
    
    print(f"1. Phiên bản Python  : {sys.version.split()[0]}")
    print(f"2. Phiên bản PyTorch : {torch.__version__}")
    print(f"3. Phiên bản CUDA    : {torch.version.cuda or 'Không khả dụng (Bản CPU)'}")
    
    cuda_available = torch.cuda.is_available()
    print(f"4. CUDA khả dụng     : {'[V] CÓ' if cuda_available else '[X] KHÔNG'}")
    
    if not cuda_available:
        print("\n [!] CẢNH BÁO: PyTorch hiện CHƯA nhận diện được GPU!")
        print("  - Nguyên nhân có thể do:")
        print("    + Máy tính không có card rời NVIDIA.")
        print("    + Chưa cài Driver NVIDIA tương thích.")
        print("    + Đang dùng phiên bản PyTorch CPU-only.")
        print("  - Để cài PyTorch hỗ trợ GPU CUDA (ví dụ CUDA 12.1):")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 60)
        return

    # Thông tin chi tiết GPU
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    total_mem = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
    
    print(f"\n THÔNG TIN PHẦN CỨNG GPU:")
    print(f"  - Số lượng GPU        : {gpu_count}")
    print(f"  - GPU đang kích hoạt  : GPU {current_device} - {device_name}")
    print(f"  - Tổng dung lượng VRAM : {total_mem:.2f} GB")
    
    # Chạy thử phép toán trên GPU
    print("\n THỬ NGHIỆM TÍNH TOÁN TRÊN GPU (Matrix Multiplication 4096 x 4096):")
    try:
        t0 = time.time()
        # Tạo 2 ma trận lớn 4096 x 4096 trên GPU
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        
        # Nhân ma trận
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Đợi GPU hoàn tất tính toán
        elapsed = (time.time() - t0) * 1000
        
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mem = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print(f"  - Kết quả phép nhân   : Thành công (Shape {tuple(c.shape)})")
        print(f"  - Thời gian xử lý     : {elapsed:.2f} ms")
        print(f"  - VRAM sử dụng        : {allocated_mem:.2f} MB (Reserved: {reserved_mem:.2f} MB)")
        print("\n [V] KẾT LUẬN: GPU hoạt động hoàn hảo và sẵn sàng cho huấn luyện mô hình!")
    except Exception as e:
        print(f"  [X] LỖI khi tính toán trên GPU: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    check_gpu()
