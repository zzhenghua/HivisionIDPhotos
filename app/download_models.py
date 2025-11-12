# download_models.py
import requests
import os
from pathlib import Path

def download_light_models():
    models = {
        'u2netp': 'https://github.com/NathanUA/U-2-Net/releases/download/1.0/u2netp.onnx'
    }
    
    Path("models").mkdir(exist_ok=True)
    
    for name, url in models.items():
        model_path = f"models/{name}.onnx"
        if not os.path.exists(model_path):
            print(f"下载 {name}...")
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"{name} 下载完成")
            except Exception as e:
                print(f"{name} 下载失败: {e}")
                # 创建一个空文件以避免程序崩溃
                with open(model_path, 'w') as f:
                    f.write("")
                print(f"已创建占位文件: {model_path}")

if __name__ == "__main__":
    download_light_models()