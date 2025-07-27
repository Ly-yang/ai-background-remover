# backend/app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import uvicorn
import os
import io
import uuid
from PIL import Image
import asyncio
from typing import Optional, List
import logging
from datetime import datetime

from app.core.config import settings
from app.services.ai_processor import AIProcessor
from app.services.image_service import ImageService
from app.services.queue_service import QueueService
from app.models.task import Task, TaskStatus
from app.utils.image_utils import validate_image, optimize_image
from app.core.security import get_current_user
from app.api.v1 import auth, images, models

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="生产级AI背景移除工具",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 初始化服务
ai_processor = AIProcessor()
image_service = ImageService()
queue_service = QueueService()

# 包含路由
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(images.router, prefix=f"{settings.API_V1_STR}/images", tags=["images"])
app.include_router(models.router, prefix=f"{settings.API_V1_STR}/models", tags=["models"])

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("启动AI背景移除服务...")
    
    # 创建必要的目录
    os.makedirs("uploads/original", exist_ok=True)
    os.makedirs("uploads/processed", exist_ok=True)
    os.makedirs("uploads/temp", exist_ok=True)
    
    # 初始化AI模型
    await ai_processor.load_models()
    
    # 初始化队列服务
    await queue_service.start()
    
    logger.info("服务启动完成!")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("正在关闭服务...")
    await queue_service.stop()
    logger.info("服务已关闭")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI Background Remover API",
        "version": settings.VERSION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查AI模型状态
        model_status = await ai_processor.health_check()
        
        # 检查队列服务状态
        queue_status = await queue_service.health_check()
        
        return {
            "status": "healthy",
            "models": model_status,
            "queue": queue_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/api/v1/process")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = "rmbg-2.0",
    mode: str = "auto",
    edge_optimization: str = "smooth",
    output_format: str = "png",
    current_user: dict = Depends(get_current_user)
):
    """处理单张图片"""
    try:
        # 验证文件
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="文件必须是图片格式")
        
        # 读取图片数据
        image_data = await file.read()
        
        # 验证图片
        if not validate_image(image_data):
            raise HTTPException(status_code=400, detail="无效的图片文件")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        task = Task(
            id=task_id,
            user_id=current_user.get("user_id"),
            status=TaskStatus.PENDING,
            model=model,
            mode=mode,
            edge_optimization=edge_optimization,
            output_format=output_format,
            created_at=datetime.utcnow()
        )
        
        # 保存原始图片
        original_path = f"uploads/original/{task_id}.{file.filename.split('.')[-1]}"
        with open(original_path, "wb") as f:
            f.write(image_data)
        
        task.original_path = original_path
        
        # 添加到处理队列
        background_tasks.add_task(process_image_task, task)
        
        return {
            "task_id": task_id,
            "status": "accepted",
            "message": "图片已加入处理队列"
        }
        
    except Exception as e:
        logger.error(f"处理图片时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_image_task(task: Task):
    """后台处理图片任务"""
    try:
        # 更新任务状态
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.utcnow()
        
        # 读取原始图片
        with open(task.original_path, "rb") as f:
            image_data = f.read()
        
        # 使用AI模型处理
        processed_data = await ai_processor.process_image(
            image_data=image_data,
            model=task.model,
            mode=task.mode,
            edge_optimization=task.edge_optimization
        )
        
        # 优化输出图片
        optimized_data = optimize_image(processed_data, task.output_format)
        
        # 保存处理结果
        processed_path = f"uploads/processed/{task.id}.{task.output_format}"
        with open(processed_path, "wb") as f:
            f.write(optimized_data)
        
        task.processed_path = processed_path
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        
        logger.info(f"任务 {task.id} 处理完成")
        
    except Exception as e:
        logger.error(f"处理任务 {task.id} 时出错: {e}")
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.completed_at = datetime.utcnow()

@app.get("/api/v1/task/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """获取任务状态"""
    try:
        # 这里应该从数据库查询任务状态
        # 为了简化，使用文件系统检查
        original_path = f"uploads/original/{task_id}.*"
        processed_path = f"uploads/processed/{task_id}.*"
        
        import glob
        
        original_files = glob.glob(original_path)
        processed_files = glob.glob(processed_path)
        
        if not original_files:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if processed_files:
            return {
                "task_id": task_id,
                "status": "completed",
                "result_url": f"/uploads/processed/{os.path.basename(processed_files[0])}"
            }
        else:
            return {
                "task_id": task_id,
                "status": "processing"
            }
            
    except Exception as e:
        logger.error(f"获取任务状态时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/result/{task_id}")
async def download_result(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    """下载处理结果"""
    try:
        import glob
        
        processed_path = f"uploads/processed/{task_id}.*"
        processed_files = glob.glob(processed_path)
        
        if not processed_files:
            raise HTTPException(status_code=404, detail="处理结果不存在")
        
        file_path = processed_files[0]
        
        def iterfile():
            with open(file_path, mode="rb") as file_like:
                yield from file_like
        
        return StreamingResponse(
            iterfile(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
        )
        
    except Exception as e:
        logger.error(f"下载结果时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch")
async def batch_process(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: str = "rmbg-2.0",
    mode: str = "auto",
    current_user: dict = Depends(get_current_user)
):
    """批量处理图片"""
    try:
        if len(files) > 10:  # 限制批量处理数量
            raise HTTPException(status_code=400, detail="批量处理最多支持10张图片")
        
        batch_id = str(uuid.uuid4())
        task_ids = []
        
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
                
            task_id = str(uuid.uuid4())
            task_ids.append(task_id)
            
            # 创建单个任务
            task = Task(
                id=task_id,
                batch_id=batch_id,
                user_id=current_user.get("user_id"),
                status=TaskStatus.PENDING,
                model=model,
                mode=mode,
                created_at=datetime.utcnow()
            )
            
            # 保存文件并添加到处理队列
            image_data = await file.read()
            original_path = f"uploads/original/{task_id}.{file.filename.split('.')[-1]}"
            with open(original_path, "wb") as f:
                f.write(image_data)
            
            task.original_path = original_path
            background_tasks.add_task(process_image_task, task)
        
        return {
            "batch_id": batch_id,
            "task_ids": task_ids,
            "status": "accepted",
            "message": f"已接收 {len(task_ids)} 张图片进行批量处理"
        }
        
    except Exception as e:
        logger.error(f"批量处理时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
