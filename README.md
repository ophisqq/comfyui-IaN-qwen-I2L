# ComfyUI-IaN-Qwen-I2L 插件

这是一个 ComfyUI 自定义节点插件，用于实现 Qwen-Image-i2L（图片转 LoRA）功能。

## 功能特性

- **ImageQwenI2L_Loader (Style)**: 加载 Qwen-Image-i2L 模型管道
  - 支持两种模式：
    - `Style`: 风格迁移模式（轻量级，2.4B 参数）
    - `Coarse+Fine+Bias`: 完整模式（保留内容和细节，7.9B + 7.6B + 30M 参数）
  
- **ImageQwenI2L_LoadGenerator**: 从训练图像生成 LoRA 权重
  - 输入多张训练图像
  - 自动生成 LoRA 权重文件
  - 可选保存到 ComfyUI 的 LoRAs 目录

## 安装步骤

### 1. 安装依赖

首先需要安装 DiffSynth-Studio：

```bash
# 克隆 DiffSynth-Studio 仓库
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

或者直接安装：

```bash
pip install diffsynth
```

### 2. 安装插件

将此插件目录复制到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone <此仓库地址> comfyui-IaN-qwen-I2L
```

或手动复制 `comfyui-IaN-qwen-I2L` 文件夹到 `ComfyUI/custom_nodes/` 目录。

### 3. 重启 ComfyUI

重启 ComfyUI 后，节点将出现在节点菜单的 `Qwen-Image-i2L` 分类下。

## 使用方法

### 基础工作流

1. **加载模型管道**
   - 添加 `ImageQwenI2L_Loader (Style)` 节点
   - 选择模型类型：
     - `Style`: 适合风格迁移，需要 4-6 张风格统一的图片
     - `Coarse+Fine+Bias`: 适合保留内容细节，需要更多训练图像
   - 选择设备（cuda/cpu）和精度（bfloat16/float16/float32）

2. **准备训练图像**
   - 使用 `Load Image` 节点加载多张图像
   - 使用 `Image Batch` 节点将多张图像合并为批次

3. **生成 LoRA**
   - 添加 `ImageQwenI2L_LoadGenerator` 节点
   - 连接 `pipeline` 输入到 Loader 节点的输出
   - 连接 `training_images` 到图像批次
   - 设置输出文件名（可选）
   - 选择是否保存到 LoRAs 目录

4. **使用生成的 LoRA**
   - 生成的 LoRA 路径会作为字符串输出
   - 可以使用 `Load LoRA` 节点加载生成的 LoRA
   - 配合 Qwen-Image 或其他扩散模型使用

### 示例：风格迁移

```
Load Image (x4-6) → Image Batch → ImageQwenI2L_LoadGenerator
                                          ↑
ImageQwenI2L_Loader (Style) ──────────────┘
```

### 示例：内容保留

```
Load Image (x5-10) → Image Batch → ImageQwenI2L_LoadGenerator
                                           ↑
ImageQwenI2L_Loader (Coarse+Fine+Bias) ────┘
```

## 节点参数说明

### ImageQwenI2L_Loader

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_type | 下拉选择 | Style | 模型类型：Style 或 Coarse+Fine+Bias |
| device | 下拉选择 | cuda | 运行设备：cuda 或 cpu |
| dtype | 下拉选择 | bfloat16 | 数据精度：bfloat16/float16/float32 |

**输出:**
- `pipeline`: I2L_PIPELINE 类型，传递给 LoadGenerator 节点

### ImageQwenI2L_LoadGenerator

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| pipeline | I2L_PIPELINE | - | 从 Loader 节点输入的管道 |
| training_images | IMAGE | - | ComfyUI 图像批次（多张图像） |
| output_name | STRING | generated_lora.safetensors | 输出文件名 |
| save_to_loras | BOOLEAN | True | 是否保存到 LoRAs 目录 |

**输出:**
- `lora_path`: STRING 类型，生成的 LoRA 文件路径

## 模型说明

### Style 模式
- **用途**: 风格迁移
- **特点**: 细节保持能力弱，但能有效提取风格信息
- **推荐图片数**: 4-6 张风格统一的图片
- **参数量**: 2.4B
- **显存需求**: 约 8-12GB

### Coarse+Fine+Bias 模式
- **用途**: 保留图像内容和细节
- **特点**: 可作为 LoRA 训练的初始化权重，加速收敛
- **推荐图片数**: 5-10 张
- **参数量**: 7.9B + 7.6B + 30M
- **显存需求**: 约 16-24GB

## 注意事项

1. **显存要求**: 
   - Style 模式至少需要 8GB 显存
   - Coarse+Fine+Bias 模式建议 16GB+ 显存
   - 可以使用 CPU 模式，但速度会很慢

2. **首次运行**: 
   - 首次运行会自动从 ModelScope 下载模型
   - 下载时间取决于网络速度
   - 模型会缓存到本地，后续运行无需重新下载

3. **图像要求**:
   - Style 模式：需要风格统一的图片
   - Coarse+Fine+Bias 模式：需要包含目标对象的多角度图片

4. **生成时间**:
   - 取决于图片数量、分辨率和硬件配置
   - 通常需要几分钟到十几分钟

## 故障排除

### 导入错误
```
ImportError: No module named 'diffsynth'
```
**解决方案**: 安装 DiffSynth-Studio
```bash
pip install diffsynth
```

### 显存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 
- 使用 float16 或 bfloat16 精度
- 减少训练图像数量
- 降低图像分辨率
- 使用 CPU 模式

### 模型下载失败
**解决方案**:
- 检查网络连接
- 手动从 ModelScope 下载模型到缓存目录
- 配置 ModelScope 镜像源

## 参考资料

- [Qwen-Image-i2L 模型页面](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L)
- [DiffSynth-Studio GitHub](https://github.com/modelscope/DiffSynth-Studio)
- [ComfyUI 官方文档](https://github.com/comfyanonymous/ComfyUI)

## 许可证

本插件遵循 Apache License 2.0 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！
