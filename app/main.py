from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import onnxruntime as ort
import os
import logging
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Light RMBG API",
    description="轻量级背景移除API - 部署在Render",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RenderRMBG:
    def __init__(self):
        self.model_path = self.get_model_path()
        self.session = self.load_model()
        
    def get_model_path(self) -> str:
        """获取模型路径，兼容Render环境"""
        possible_paths = [
            "app/models/u2netp.onnx",
            "./app/models/u2netp.onnx", 
            "/opt/render/project/src/app/models/u2netp.onnx",
            "models/u2netp.onnx"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"找到模型文件: {path}")
                return path
        
        raise FileNotFoundError("未找到模型文件")
    
    def load_model(self):
        """加载模型，优化Render环境配置"""
        # Render免费实例内存有限，使用最小配置
        providers = ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = False  # 节省内存
        
        try:
            session = ort.InferenceSession(
                self.model_path,
                providers=providers,
                sess_options=sess_options
            )
            logger.info("模型加载成功")
            return session
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def preprocess(self, image: np.ndarray, target_size: tuple = (320, 320)) -> np.ndarray:
        """轻量预处理"""
        # 调整尺寸减少内存占用
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        
        # 简化标准化
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = (image - mean) / std
        
        # CHW格式
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
    
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """移除背景主逻辑"""
        try:
            original_size = (image.shape[1], image.shape[0])
            
            # 预处理
            input_data = self.preprocess(image)
            
            # 推理
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})
            mask = outputs[0][0][0]
            
            # 后处理
            mask = cv2.resize(mask, original_size)
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 应用透明背景
            if len(image.shape) == 3 and image.shape[2] == 3:
                b, g, r = cv2.split(image)
                result = cv2.merge([b, g, r, mask])
            else:
                result = image
                
            return result
            
        except Exception as e:
            logger.error(f"背景移除失败: {e}")
            raise

# 全局模型实例
rmbg_model = RenderRMBG()

@app.get("/")
async def root():
    return {
        "message": "RMBG API运行中",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rmbg-api",
        "memory_usage": "low"
    }

@app.post("/api/remove-background")
async def remove_background(
    file: UploadFile = File(...),
    size: Optional[str] = "medium"
):
    """
    移除图片背景
    - file: 图片文件 (支持jpg, png, jpeg)
    - size: 输出尺寸 small(256x256) | medium(512x512) | large(原始尺寸)
    """
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "只支持图片文件")
    
    # 读取文件
    contents = await file.read()
    
    # 限制文件大小 (Render免费实例内存有限)
    if len(contents) > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(400, "图片大小不能超过5MB")
    
    try:
        # 解码图片
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(400, "无法解码图片")
        
        # 调整尺寸策略
        if size == "small" and max(image.shape) > 512:
            scale = 512 / max(image.shape)
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
        elif size == "medium" and max(image.shape) > 1024:
            scale = 1024 / max(image.shape)
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # 处理图片
        logger.info(f"开始处理图片: {image.shape}")
        result_image = rmbg_model.remove_background(image)
        
        # 编码结果
        success, encoded_image = cv2.imencode('.png', result_image)
        if not success:
            raise HTTPException(500, "图片编码失败")
        
        return Response(
            content=encoded_image.tobytes(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=removed_bg.png"
            }
        )
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise HTTPException(500, f"处理失败: {str(e)}")

@app.post("/api/health/deep")
async def deep_health_check():
    """深度健康检查，测试模型推理"""
    try:
        # 创建测试图片
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = rmbg_model.remove_background(test_image)
        
        return {
            "status": "healthy", 
            "model_working": True,
            "test_image_processed": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_working": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port, 
        workers=1,  # Render免费实例只运行1个worker
        log_level="info"
    )