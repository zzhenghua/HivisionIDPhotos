# download_models.py
import requests
import os
from pathlib import Path

def download_light_models():
    models = {
        'u2netp': 'https://github.com/NathanUA/U-2-Net/releases/download/1.0/u2netp.onnx',
        'modnet': 'https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz&export=download'
    }
    
    Path("models").mkdir(exist_ok=True)
    
    for name, url in models.items():
        model_path = f"models/{name}.onnx"
        if not os.path.exists(model_path):
            print(f"下载 {name}...")
            
            if 'drive.google.com' in url:
                # 需要特殊处理Google Drive链接
                download_gdrive(url, model_path)
            else:
                response = requests.get(url, stream=True)
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print(f"{name} 下载完成")

def download_gdrive(gdrive_url, output_path):
    """下载Google Drive文件"""
    import gdown
    gdown.download(gdrive_url, output_path, fuzzy=True)

if __name__ == "__main__":
    download_light_models()