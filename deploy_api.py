# 带缓存的轻量版本
import sqlite3
import hashlib
from pathlib import Path

class CachedRMBG:
    def __init__(self, model_path: str):
        self.model = LightRMBG(model_path)
        self.setup_cache()
    
    def setup_cache(self):
        """设置SQLite缓存"""
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.conn = sqlite3.connect('cache.db', check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS image_cache (
                hash TEXT PRIMARY KEY,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    def get_image_hash(self, image_data: bytes) -> str:
        return hashlib.md5(image_data).hexdigest()
    
    def process_cached(self, image_data: bytes) -> bytes:
        image_hash = self.get_image_hash(image_data)
        
        # 检查缓存
        cursor = self.conn.execute(
            'SELECT file_path FROM image_cache WHERE hash = ?', 
            (image_hash,)
        )
        result = cursor.fetchone()
        
        if result:
            # 返回缓存结果
            with open(result[0], 'rb') as f:
                return f.read()
        
        # 处理新图像
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result_image = self.model.remove_background(image)
        
        # 保存结果
        success, encoded = cv2.imencode('.png', result_image)
        result_data = encoded.tobytes()
        
        # 缓存到文件
        cache_file = self.cache_dir / f"{image_hash}.png"
        with open(cache_file, 'wb') as f:
            f.write(result_data)
        
        # 更新数据库
        self.conn.execute(
            'INSERT OR REPLACE INTO image_cache (hash, file_path) VALUES (?, ?)',
            (image_hash, str(cache_file))
        )
        self.conn.commit()
        
        return result_data