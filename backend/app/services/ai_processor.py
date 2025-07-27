# backend/app/services/ai_processor.py
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import io
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import os

from app.core.config import settings

logger = logging.getLogger(__name__)

class AIProcessor:
    """AI模型处理器，支持多种背景移除模型"""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_configs = {
            "rmbg-2.0": {
                "input_size": (1024, 1024),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://huggingface.co/briaai/RMBG-2.0/resolve/main/model.pth"
            },
            "u2net": {
                "input_size": (320, 320),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2net/u2net.pth"
            },
            "birefnet": {
                "input_size": (1024, 1024),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth"
            },
            "sam": {
                "input_size": (1024, 1024),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            },
            "isnet-anime": {
                "input_size": (1024, 1024),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://github.com/xuebinqin/DIS/releases/download/isnet/isnet-general-use.pth"
            }
        }
        
    async def load_models(self):
        """加载所有AI模型"""
        logger.info("开始加载AI模型...")
        
        # 并行加载模型
        tasks = []
        for model_name in self.model_configs.keys():
            task = asyncio.create_task(self._load_single_model(model_name))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        loaded_count = 0
        for i, result in enumerate(results):
            model_name = list(self.model_configs.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"加载模型 {model_name} 失败: {result}")
            else:
                loaded_count += 1
                logger.info(f"模型 {model_name} 加载成功")
        
        logger.info(f"共加载 {loaded_count}/{len(self.model_configs)} 个模型")
        
        if loaded_count == 0:
            raise RuntimeError("所有模型加载失败")
    
    async def _load_single_model(self, model_name: str):
        """加载单个模型"""
        try:
            model_path = f"./models/{model_name}/model.pth"
            
            # 如果模型文件不存在，尝试下载
            if not os.path.exists(model_path):
                await self._download_model(model_name)
            
            # 根据模型类型加载不同的网络结构
            if model_name == "rmbg-2.0":
                model = self._load_rmbg_model(model_path)
            elif model_name == "u2net":
                model = self._load_u2net_model(model_path)
            elif model_name == "birefnet":
                model = self._load_birefnet_model(model_path)
            elif model_name == "sam":
                model = self._load_sam_model(model_path)
            elif model_name == "isnet-anime":
                model = self._load_isnet_model(model_path)
            else:
                raise ValueError(f"不支持的模型: {model_name}")
            
            model = model.to(self.device)
            model.eval()
            
            # 模型预热
            dummy_input = torch.randn(1, 3, *self.model_configs[model_name]["input_size"]).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            self.models[model_name] = model
            return model
            
        except Exception as e:
            logger.error(f"加载模型 {model_name} 时出错: {e}")
            raise
    
    async def _download_model(self, model_name: str):
        """下载模型文件"""
        logger.info(f"下载模型 {model_name}...")
        
        model_dir = f"./models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        url = self.model_configs[model_name]["url"]
        model_path = f"{model_dir}/model.pth"
        
        def download():
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # 在线程池中执行下载
        await asyncio.get_event_loop().run_in_executor(self.executor, download)
        logger.info(f"模型 {model_name} 下载完成")
    
    def _load_rmbg_model(self, model_path: str):
        """加载RMBG模型"""
        from models.rmbg.model import RMBGModel
        model = RMBGModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def _load_u2net_model(self, model_path: str):
        """加载U²-Net模型"""
        from models.u2net.model import U2NET
        model = U2NET(3, 1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def _load_birefnet_model(self, model_path: str):
        """加载BiRefNet模型"""
        from models.birefnet.model import BiRefNet
        model = BiRefNet()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def _load_sam_model(self, model_path: str):
        """加载SAM模型"""
        from models.sam.build_sam import sam_model_registry
        model = sam_model_registry["vit_h"](checkpoint=model_path)
        return model.to(self.device)
    
    def _load_isnet_model(self, model_path: str):
        """加载ISNet模型"""
        from models.isnet.model import ISNetDIS
        model = ISNetDIS()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    async def process_image(
        self,
        image_data: bytes,
        model: str = "rmbg-2.0",
        mode: str = "auto",
        edge_optimization: str = "smooth",
        **kwargs
    ) -> bytes:
        """处理单张图片"""
        try:
            start_time = time.time()
            
            # 检查模型是否已加载
            if model not in self.models:
                raise ValueError(f"模型 {model} 未加载")
            
            # 预处理图片
            input_tensor, original_size = await self._preprocess_image(image_data, model)
            
            # AI推理
            with torch.no_grad():
                if model == "sam":
                    mask = await self._process_with_sam(input_tensor, **kwargs)
                else:
                    mask = await self._process_with_standard_model(input_tensor, model)
            
            # 后处理
            result = await self._postprocess_mask(
                mask, original_size, image_data, edge_optimization, mode
            )
            
            processing_time = time.time() - start_time
            logger.info(f"图片处理完成，耗时: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"处理图片时出错: {e}")
            raise
    
    async def _preprocess_image(self, image_data: bytes, model: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """预处理图片"""
        # 在线程池中执行图片预处理
        def preprocess():
            # 读取图片
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            original_size = image.size
            
            # 获取模型配置
            config = self.model_configs[model]
            input_size = config["input_size"]
            mean = config["mean"]
            std = config["std"]
            
            # 转换为tensor
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            tensor = transform(image).unsqueeze(0)
            return tensor, original_size
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, preprocess)
    
    async def _process_with_standard_model(self, input_tensor: torch.Tensor, model: str) -> torch.Tensor:
        """使用标准模型处理"""
        def inference():
            model_instance = self.models[model]
            input_tensor_gpu = input_tensor.to(self.device)
            
            if model == "u2net":
                # U²-Net返回多个输出，取第一个
                d1, d2, d3, d4, d5, d6, d7 = model_instance(input_tensor_gpu)
                return torch.sigmoid(d1)
            elif model == "rmbg-2.0":
                # RMBG直接返回mask
                return torch.sigmoid(model_instance(input_tensor_gpu))
            elif model == "birefnet":
                # BiRefNet返回精细化的mask
                return torch.sigmoid(model_instance(input_tensor_gpu))
            elif model == "isnet-anime":
                # ISNet for anime
                return torch.sigmoid(model_instance(input_tensor_gpu)[0])
            else:
                return torch.sigmoid(model_instance(input_tensor_gpu))
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, inference)
    
    async def _process_with_sam(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """使用SAM模型处理"""
        def sam_inference():
            sam_model = self.models["sam"]
            
            # SAM需要特殊的处理流程
            input_tensor_gpu = input_tensor.to(self.device)
            
            # 编码图片
            sam_model.set_image(input_tensor_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy())
            
            # 如果没有提供点击点，使用自动模式
            if "points" not in kwargs:
                # 生成网格点进行自动分割
                h, w = input_tensor.shape[2:]
                points = []
                for i in range(0, h, h//4):
                    for j in range(0, w, w//4):
                        points.append([j, i])
                points = np.array(points)
                labels = np.ones(len(points))
            else:
                points = np.array(kwargs["points"])
                labels = np.array(kwargs.get("labels", [1] * len(points)))
            
            # 预测mask
            masks, scores, logits = sam_model.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            # 选择最佳mask
            best_mask = masks[np.argmax(scores)]
            return torch.from_numpy(best_mask).float().unsqueeze(0).unsqueeze(0)
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, sam_inference)
    
    async def _postprocess_mask(
        self,
        mask: torch.Tensor,
        original_size: Tuple[int, int],
        original_image_data: bytes,
        edge_optimization: str,
        mode: str
    ) -> bytes:
        """后处理mask"""
        def postprocess():
            # 调整mask尺寸到原图大小
            mask_resized = F.interpolate(
                mask, size=original_size[::-1], mode='bilinear', align_corners=False
            )
            mask_np = mask_resized.squeeze().cpu().numpy()
            
            # 边缘优化
            if edge_optimization == "smooth":
                mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0)
            elif edge_optimization == "sharp":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                mask_np = cv2.filter2D(mask_np, -1, kernel)
                mask_np = np.clip(mask_np, 0, 1)
            elif edge_optimization == "feather":
                mask_np = cv2.GaussianBlur(mask_np, (5, 5), 2.0)
            
            # 阈值化
            mask_np = (mask_np > 0.5).astype(np.uint8) * 255
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
            
            # 读取原图
            original_image = Image.open(io.BytesIO(original_image_data)).convert('RGBA')
            original_np = np.array(original_image)
            
            # 应用mask
            if original_np.shape[2] == 3:
                # 如果原图是RGB，添加alpha通道
                alpha_channel = mask_np
                result_np = np.dstack([original_np, alpha_channel])
            else:
                # 如果原图已有alpha通道，替换它
                result_np = original_np.copy()
                result_np[:, :, 3] = mask_np
            
            # 转换回PIL图片
            result_image = Image.fromarray(result_np, 'RGBA')
            
            # 保存为bytes
            output_buffer = io.BytesIO()
            result_image.save(output_buffer, format='PNG', optimize=True)
            return output_buffer.getvalue()
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, postprocess)
    
    async def batch_process(
        self,
        image_data_list: list,
        model: str = "rmbg-2.0",
        **kwargs
    ) -> list:
        """批量处理图片"""
        try:
            logger.info(f"开始批量处理 {len(image_data_list)} 张图片")
            
            # 并行处理
            tasks = []
            for image_data in image_data_list:
                task = asyncio.create_task(
                    self.process_image(image_data, model, **kwargs)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"批量处理第 {i+1} 张图片失败: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            success_count = len([r for r in processed_results if r is not None])
            logger.info(f"批量处理完成: {success_count}/{len(image_data_list)} 张图片成功")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"批量处理时出错: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = {
                "device": str(self.device),
                "models_loaded": len(self.models),
                "available_models": list(self.models.keys()),
                "gpu_available": torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                status["gpu_memory"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_memory": torch.cuda.max_memory_allocated()
                }
            
            return status
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_name not in self.model_configs:
            return None
        
        config = self.model_configs[model_name].copy()
        config["loaded"] = model_name in self.models
        config["device"] = str(self.device)
        
        return config
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        return list(self.models.keys())
