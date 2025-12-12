"""
ComfyUI Qwen-Image-i2L 节点实现
"""

import torch
import numpy as np
from PIL import Image
import folder_paths
import os

try:
    from diffsynth.pipelines.qwen_image import (
        QwenImagePipeline, 
        ModelConfig,
        QwenImageUnit_Image2LoRAEncode, 
        QwenImageUnit_Image2LoRADecode
    )
    from diffsynth.utils.lora import merge_lora
    from diffsynth import load_state_dict
    from safetensors.torch import save_file
    DIFFSYNTH_AVAILABLE = True
except ImportError as e:
    DIFFSYNTH_AVAILABLE = False
    print(f"警告: DiffSynth-Studio 未安装或导入失败。错误: {e}")
    print("请运行: pip install diffsynth")
except Exception as e:
    DIFFSYNTH_AVAILABLE = False
    print(f"警告: DiffSynth-Studio 导入时发生错误: {e}")
    import traceback
    traceback.print_exc()


class ImageQwenI2L_Loader:
    """
    加载 Qwen-Image-i2L 模型管道
    支持 Style、Coarse+Fine+Bias 两种模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["Style", "Coarse+Fine+Bias"], {
                    "default": "Style"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16"
                }),
            }
        }
    
    RETURN_TYPES = ("I2L_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Qwen-Image-i2L"
    
    def load_pipeline(self, model_type, device, dtype):
        if not DIFFSYNTH_AVAILABLE:
            raise RuntimeError("DiffSynth-Studio 未安装，请先安装: pip install diffsynth")
        
        # 转换 dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        # VRAM 配置
        vram_config_disk_offload = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch_dtype,
            "preparing_device": device,
            "computation_dtype": torch_dtype,
            "computation_device": device,
        }
        
        # 根据模型类型配置不同的模型
        if model_type == "Style":
            model_configs = [
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern="SigLIP2-G384/model.safetensors", 
                    **vram_config_disk_offload
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern="DINOv3-7B/model.safetensors", 
                    **vram_config_disk_offload
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/Qwen-Image-i2L", 
                    origin_file_pattern="Qwen-Image-i2L-Style.safetensors", 
                    **vram_config_disk_offload
                ),
            ]
        else:  # Coarse+Fine+Bias
            model_configs = [
                ModelConfig(
                    model_id="Qwen/Qwen-Image", 
                    origin_file_pattern="text_encoder/model*.safetensors", 
                    **vram_config_disk_offload
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern="SigLIP2-G384/model.safetensors", 
                    **vram_config_disk_offload
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern="DINOv3-7B/model.safetensors", 
                    **vram_config_disk_offload
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/Qwen-Image-i2L", 
                    origin_file_pattern="Qwen-Image-i2L-Coarse.safetensors", 
                    **vram_config_disk_offload
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/Qwen-Image-i2L", 
                    origin_file_pattern="Qwen-Image-i2L-Fine.safetensors", 
                    **vram_config_disk_offload
                ),
            ]
        
        # 计算 VRAM 限制
        if device == "cuda" and torch.cuda.is_available():
            vram_limit = torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - 0.5
        else:
            vram_limit = 8.0  # CPU 模式默认限制
        
        # 加载管道
        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            processor_config=ModelConfig(
                model_id="Qwen/Qwen-Image-Edit", 
                origin_file_pattern="processor/"
            ),
            vram_limit=vram_limit,
        )
        
        return ({
            "pipe": pipe,
            "model_type": model_type,
            "device": device,
            "dtype": torch_dtype
        },)


class ImageQwenI2L_LoadGenerator:
    """
    从训练图像生成 LoRA 权重
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("I2L_PIPELINE",),
                "training_images": ("IMAGE",),  # ComfyUI 图像批次
            },
            "optional": {
                "output_name": ("STRING", {
                    "default": "generated_lora.safetensors"
                }),
                "save_to_loras": ("BOOLEAN", {
                    "default": True,
                    "label_on": "保存到 LoRAs 目录",
                    "label_off": "仅临时文件"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("lora_path", "lora_name",)
    FUNCTION = "generate_lora"
    CATEGORY = "Qwen-Image-i2L"
    
    def generate_lora(self, pipeline, training_images, output_name="generated_lora.safetensors", save_to_loras=True):
        if not DIFFSYNTH_AVAILABLE:
            raise RuntimeError("DiffSynth-Studio 未安装")
        
        pipe_data = pipeline
        pipe = pipe_data["pipe"]
        model_type = pipe_data["model_type"]
        device = pipe_data["device"]
        torch_dtype = pipe_data["dtype"]
        
        # 转换 ComfyUI 图像格式到 PIL
        # ComfyUI 图像格式: [B, H, W, C] 范围 [0, 1]
        pil_images = []
        for img_tensor in training_images:
            # 转换为 numpy 并调整范围到 [0, 255]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)
        
        print(f"处理 {len(pil_images)} 张训练图像...")
        
        # 生成 LoRA
        with torch.no_grad():
            embs = QwenImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=pil_images)
            lora = QwenImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]
            
            # 如果是 Coarse+Fine+Bias 模式，合并 Bias LoRA
            if model_type == "Coarse+Fine+Bias":
                print("加载并合并 Bias LoRA...")
                lora_bias_config = ModelConfig(
                    model_id="DiffSynth-Studio/Qwen-Image-i2L", 
                    origin_file_pattern="Qwen-Image-i2L-Bias.safetensors"
                )
                lora_bias_config.download_if_necessary()
                lora_bias = load_state_dict(
                    lora_bias_config.path, 
                    torch_dtype=torch_dtype, 
                    device=device
                )
                lora = merge_lora([lora, lora_bias])
        
        # 保存 LoRA
        if save_to_loras:
            # 保存到 ComfyUI 的 LoRAs 目录
            loras_dir = folder_paths.get_folder_paths("loras")[0]
            output_path = os.path.join(loras_dir, output_name)
        else:
            # 保存到临时目录
            output_path = os.path.join(folder_paths.get_temp_directory(), output_name)
        
        save_file(lora, output_path)
        print(f"LoRA 已保存到: {output_path}")
        
        return (output_path, output_name,)


class ImageQwenI2L_ApplyLora:
    """
    应用生成的 LoRA 到模型（仅模型）
    可以直接接收路径输入
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_path": ("STRING", {"forceInput": True}),
                "strength_model": ("FLOAT", {
                    "default": 1.0, 
                    "min": -100.0, 
                    "max": 100.0, 
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_lora"
    CATEGORY = "Qwen-Image-i2L"
    
    def apply_lora(self, model, lora_path, strength_model):
        import comfy.utils
        import comfy.sd
        
        if strength_model == 0:
            return (model,)
        
        # 检查文件是否存在
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA 文件不存在: {lora_path}")
        
        # 加载 LoRA
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        # 应用到模型（仅模型，不包含CLIP）
        model_lora = comfy.sd.load_lora_for_models(
            model, None, lora, strength_model, 0
        )[0]
        
        return (model_lora,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageQwenI2L_Loader": ImageQwenI2L_Loader,
    "ImageQwenI2L_LoadGenerator": ImageQwenI2L_LoadGenerator,
    "ImageQwenI2L_ApplyLora": ImageQwenI2L_ApplyLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageQwenI2L_Loader": "Qwen-Image-i2L Loader (Style)",
    "ImageQwenI2L_LoadGenerator": "Qwen-Image-i2L Load Generator",
    "ImageQwenI2L_ApplyLora": "Qwen-Image-i2L Apply LoRA",
}
