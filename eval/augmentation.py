import torch
import numpy as np
from lavis.common.gradcam import getAttMap
from torchvision import transforms
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from torchvision import transforms
import textwrap
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import time


def _compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    pass

def remove_cls_token(features):
    return features[:, 1:, :]


def adjust_weights_by_similarity(patch_features, patch_weights, temperature=0.1):

    if patch_features.dtype == torch.float16:
        patch_features = patch_features.float()
    if patch_weights.dtype == torch.float16:
        patch_weights = patch_weights.float()

    features_normalized = F.normalize(patch_features, p=2, dim=-1)
    similarity_matrix = torch.matmul(features_normalized, features_normalized.T)
    attention_weights = F.softmax(similarity_matrix / temperature, dim=-1)
    adjusted_weights = torch.matmul(attention_weights, patch_weights.unsqueeze(-1)).squeeze(-1)
    change_norm = torch.norm(adjusted_weights - patch_weights)

    return adjusted_weights

def augmentation(image, question, tensor_image, model, tokenized_text, raw_image, vision_encoder, vision_processor, blocks: list[int],
                 weights: list[float] | None=None, save_base_dir=None, sample_id=None, problem=None):
    # start_time = time.time()
    if weights is None:
        weights = [1.0 / len(blocks)] * len(blocks)
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    with torch.set_grad_enabled(True):
        gradcams, _ = compute_gradcam(model=model,
                                      visual_input=image,
                                      text_input=question,
                                      tokenized_text=tokenized_text,
                                      blocks=blocks,
                                      weights=weights)
    # start_time2 = time.time()

    with torch.no_grad():
        inputs = vision_processor(raw_image, return_tensors="pt").to(dtype=torch.float16)
        inputs = {key: value.to(vision_encoder.device) for key, value in inputs.items()}
        original_features = vision_encoder(**inputs, output_hidden_states=True).last_hidden_state
        original_features = remove_cls_token(original_features).squeeze()
    gradcams = [gradcam_[1] for gradcam_ in gradcams]
    gradcams1 = torch.stack(gradcams).reshape(image.size(0), -1)
    # end_time = time.time()

    itc_score = model({"image": image, "text_input": question}, match_head='itc')
    # third_time = time.time()
    # print(f"执行时间:| GradCAM: {start_time2 - start_time:.6f} 秒 | vision encoder: {(end_time - start_time2):.6f} 秒 | itc compare: {third_time - end_time:.6f} 秒")
    ratio = 1 - itc_score / 2
    ratio = min(ratio, 1 - 10 ** (-5))
    resized_img = raw_image.resize((384, 384))
    norm_img = np.float32(resized_img) / 255
    # gradcams1 = adjust_weights_by_similarity(original_features.to(device=gradcams1.device,dtype=torch.float32), gradcams1)
    gradcam = gradcams1.reshape(24, 24)
    gradcam = adjust_weights_by_similarity(original_features.to(device=gradcam.device,dtype=torch.float32), gradcam.flatten()).reshape(24,24)
    avg_gradcam = getAttMap(norm_img, gradcam.cpu().numpy(), blur=True, overlap=False)

    temp, _ = torch.sort(torch.tensor(avg_gradcam).reshape(-1), descending=True)
    cam1 = torch.tensor(avg_gradcam).unsqueeze(2)
    cam = torch.cat([cam1, cam1, cam1], dim=2)
    mask = torch.where(cam < temp[int(384 * 384 * ratio)], 0, 1)

    if save_base_dir and sample_id:
        print("你好呀")
        sample_dir = os.path.join(save_base_dir, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        save_cam_visualization(cam, sample_dir, int(384 * 384 * ratio))

        save_mask_visualization(mask, sample_dir, int(384 * 384 * ratio))
    new_image = tensor_image.permute(1, 2, 0) * mask
    unloader = transforms.ToPILImage()
    imag = new_image.clone().permute(2, 0, 1)
    augmented_image = unloader(imag)

    if save_base_dir and sample_id:

        sample_dir = os.path.join(save_base_dir, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        raw_image.save(os.path.join(sample_dir, "original.jpg"))

        augmented_image.save(os.path.join(sample_dir, "augmented.jpg"))

        with open(os.path.join(sample_dir, "problem.txt"), "w") as f:
            f.write(problem if problem else question)

        create_visualizations(raw_image, augmented_image, gradcam, sample_dir,
                              problem if problem else question, itc_score)

    return augmented_image


def save_cam_visualization(cam, sample_dir, threshold_value):
    """
    保存cam（注意力热力图）的可视化
    """
    # cam是3通道的，取第一个通道即可
    cam_single = cam[:, :, 0].cpu().numpy()

    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 原始cam热力图
    im1 = ax1.imshow(cam_single, cmap='jet')
    ax1.set_title("Attention Heatmap (CAM)")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 添加阈值线
    ax1.axhline(y=threshold_value * 384, color='white', linestyle='--', linewidth=1)
    ax1.text(10, threshold_value * 384 + 10, f'Threshold: {threshold_value:.4f}',
             color='white', fontsize=10, backgroundcolor='black')

    # 阈值化后的cam
    cam_thresholded = np.where(cam_single < threshold_value, 0, cam_single)
    im2 = ax2.imshow(cam_thresholded, cmap='jet')
    ax2.set_title("Thresholded CAM\n(Values < threshold set to 0)")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "cam_visualization.png"),
                bbox_inches='tight', dpi=150)
    plt.close()

    # 单独保存cam数据
    np.save(os.path.join(sample_dir, "cam_data.npy"), cam_single)

    # print(f"CAM可视化已保存: {os.path.join(sample_dir, 'cam_visualization.png')}")


def save_mask_visualization(mask, sample_dir, threshold_value):
    """
    保存mask（二值掩码）的可视化
    """
    # mask是3通道的，取第一个通道即可
    mask_single = mask[:, :, 0].cpu().numpy()

    # 创建可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 二值mask
    ax1.imshow(mask_single, cmap='gray')
    ax1.set_title("Binary Mask (0/1)")
    ax1.axis('off')

    # 统计信息
    total_pixels = mask_single.size
    preserved_pixels = np.sum(mask_single)
    preserved_ratio = preserved_pixels / total_pixels

    # mask统计
    ax2.axis('off')
    stats_text = f"Mask Statistics:\n\n" \
                 f"Total pixels: {total_pixels:,}\n" \
                 f"Preserved pixels: {preserved_pixels:,}\n" \
                 f"Preserved ratio: {preserved_ratio:.2%}\n" \
                 f"Threshold: {threshold_value:.4f}"
    ax2.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    # 像素值分布直方图
    ax3.hist(mask_single.flatten(), bins=[-0.5, 0.5, 1.5],
             alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title("Mask Value Distribution")
    ax3.set_xlabel("Mask Value")
    ax3.set_ylabel("Pixel Count")
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['0 (Masked)', '1 (Preserved)'])

    # 在直方图上添加数值标注
    for i, count in enumerate([total_pixels - preserved_pixels, preserved_pixels]):
        ax3.text(i, count + total_pixels * 0.01, f'{count:,}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "mask_visualization.png"),
                bbox_inches='tight', dpi=150)
    plt.close()

    # 保存mask为图像
    mask_image = (mask_single * 255).astype(np.uint8)
    Image.fromarray(mask_image, mode='L').save(os.path.join(sample_dir, "mask.png"))

    # 单独保存mask数据
    np.save(os.path.join(sample_dir, "mask_data.npy"), mask_single)

    # print(f"Mask可视化已保存: {os.path.join(sample_dir, 'mask_visualization.png')}")
def create_visualizations(raw_image, augmented_image, patch_heatmap, sample_dir, problem, itc_score):

    raw_img_resized = raw_image.resize((384, 384))
    augmented_img_resized = augmented_image.resize((384, 384))

    raw_img_np = np.array(raw_img_resized)
    augmented_img_np = np.array(augmented_img_resized)

    # 将24x24热力图上采样到384x384（使用最近邻插值保持网格边界）
    patch_heatmap_np = patch_heatmap.cpu().numpy()
    # print("MAX 2: ", np.max(patch_heatmap_np))
    heatmap_upscaled = cv2.resize(patch_heatmap_np, (384, 384), interpolation=cv2.INTER_NEAREST)

    # 保存上采样后的热力图（无模糊）
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap_upscaled, cmap='jet')
    plt.colorbar()
    plt.title("Upscaled Heatmap (384x384, No Blur)")
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, "upsampled_heatmap.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # 应用高斯模糊处理
    heatmap_blurred = cv2.GaussianBlur(heatmap_upscaled, (15, 15), 0)

    # 保存高斯模糊后的热力图
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap_blurred, cmap='jet')
    plt.colorbar()
    plt.title("Gaussian Blurred Heatmap (384x384)")
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, "blurred_heatmap.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # 归一化热力图到0-1范围（使用模糊后的版本）
    if heatmap_blurred.max() > heatmap_blurred.min():
        heatmap_normalized = (heatmap_blurred - heatmap_blurred.min()) / (
                heatmap_blurred.max() - heatmap_blurred.min())
    else:
        heatmap_normalized = heatmap_blurred

    # 创建热力图颜色映射（使用jet色彩）
    heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]  # 取RGB，忽略alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # 将原图转换为0-1浮点数
    raw_img_float = raw_img_np.astype(np.float32) / 255.0

    # 创建叠加图像（增大透明度到0.8）
    alpha = 0.2  # 增大热力图透明度
    overlay = raw_img_float * (1 - alpha) + heatmap_colored.astype(np.float32) / 255.0 * alpha
    overlay = (overlay * 255).astype(np.uint8)

    # 添加网格线以突出patch边界
    patch_size = 384 // 24  # 每个patch的像素大小
    for i in range(0, 384, patch_size):
        cv2.line(overlay, (i, 0), (i, 383), (255, 255, 255), 1)
        cv2.line(overlay, (0, i), (383, i), (255, 255, 255), 1)

    # 保存叠加图像
    overlay_pil = Image.fromarray(overlay)
    overlay_pil.save(os.path.join(sample_dir, "patch_overlay.png"))

    # 保存热力图
    plt.figure(figsize=(6, 6))
    plt.imshow(patch_heatmap_np, cmap='jet')
    plt.colorbar()
    plt.title("24x24 Patch-level Attention Weights")
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, "heatmap.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # 创建包含增强图像和问题文本的对比网格
    fig = plt.figure(figsize=(20, 16))  # 增大图形尺寸以容纳更多子图

    # 创建网格布局：顶部为问题文本，底部为6个图像
    gs = fig.add_gridspec(7, 2, height_ratios=[0.7, 1, 1, 1, 1, 1, 1])

    # 添加问题文本区域（顶部跨两列）
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')

    # 自动换行处理长问题
    wrapped_problem = textwrap.fill(problem, width=80)

    # 显示问题文本和ITC分数
    ax_text.text(0.5, 0.7, "Problem:", ha='center', va='center', fontsize=16, fontweight='bold',
                 transform=ax_text.transAxes)
    ax_text.text(0.5, 0.3, wrapped_problem, ha='center', va='center', fontsize=14,
                 wrap=True, transform=ax_text.transAxes)

    # 创建六个图像子图
    ax1 = fig.add_subplot(gs[1, 0])  # 原图
    ax2 = fig.add_subplot(gs[1, 1])  # 叠加图
    ax3 = fig.add_subplot(gs[2, 0])  # 增强图像
    ax4 = fig.add_subplot(gs[2, 1])  # 24x24热力图
    ax5 = fig.add_subplot(gs[3, 0])  # 上采样热力图（无模糊）
    ax6 = fig.add_subplot(gs[3, 1])  # 高斯模糊热力图
    ax7 = fig.add_subplot(gs[4, 0])  # 上采样热力图颜色条
    ax8 = fig.add_subplot(gs[4, 1])  # 高斯模糊热力图颜色条
    ax9 = fig.add_subplot(gs[5, :])  # ITC分数和说明（底部跨两列）

    # 原图
    ax1.imshow(raw_img_np)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 叠加图
    ax2.imshow(overlay)
    ax2.set_title("Patch Overlay (α=0.2)", fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 增强图像
    ax3.imshow(augmented_img_np)
    ax3.set_title("Augmented Image", fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 24x24热力图
    im4 = ax4.imshow(patch_heatmap_np, cmap='jet')
    ax4.set_title("24x24 Patch Heatmap", fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 上采样热力图（无模糊）
    im5 = ax5.imshow(heatmap_upscaled, cmap='jet')
    ax5.set_title("Upscaled Heatmap (No Blur)", fontsize=12, fontweight='bold')
    ax5.axis('off')

    # 高斯模糊热力图
    im6 = ax6.imshow(heatmap_blurred, cmap='jet')
    ax6.set_title("Gaussian Blurred Heatmap", fontsize=12, fontweight='bold')
    ax6.axis('off')

    # 上采样热力图颜色条
    plt.colorbar(im5, cax=ax7)
    ax7.set_title("Attention Intensity (No Blur)", fontsize=10)

    # 高斯模糊热力图颜色条
    plt.colorbar(im6, cax=ax8)
    ax8.set_title("Attention Intensity (Blurred)", fontsize=10)

    # 添加ITC分数和说明
    ax9.axis('off')

    # 修复：将张量转换为Python标量
    if hasattr(itc_score, 'item'):  # 如果是PyTorch张量
        itc_score_value = itc_score.item()
    elif hasattr(itc_score, 'numpy'):  # 如果是TensorFlow张量
        itc_score_value = itc_score.numpy()
    else:  # 已经是标量
        itc_score_value = itc_score

    info_text = f"ITC Score: {itc_score_value:.4f}\n\n" \
                "Augmentation removes regions with low attention weights,\n" \
                "keeping only the areas most relevant to the problem.\n\n" \
                "Processing pipeline: 24x24 heatmap → Upscale to 384x384 → Gaussian blur → Overlay"
    ax9.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
             transform=ax9.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "comparison_grid.png"), bbox_inches='tight', dpi=150)
    plt.close()

    # 保存热力图数据（供后续分析使用）
    np.save(os.path.join(sample_dir, "heatmap_data.npy"), patch_heatmap_np)
    np.save(os.path.join(sample_dir, "upsampled_heatmap_data.npy"), heatmap_upscaled)
    np.save(os.path.join(sample_dir, "blurred_heatmap_data.npy"), heatmap_blurred)

    # print(f"样本可视化已保存到: {sample_dir}")
    # print(f"包含文件: original.jpg, augmented.jpg, problem.txt, heatmap.png, " +
    #       f"patch_overlay.png, upsampled_heatmap.png, blurred_heatmap.png, " +
    #       f"comparison_grid.png, heatmap_data.npy, upsampled_heatmap_data.npy, blurred_heatmap_data.npy")
    #

def multi_layer_augmentation(image, question, tensor_image, model, tokenized_text, raw_image,
                             layers, save_base_dir=None, sample_id=None, problem=None):
    """
    处理多个层，生成对应的attention map和增强图像

    参数:
    - layers: 要处理的层数列表，如 [4, 6, 8, 10]
    - 其他参数同augmentation函数
    """
    # 确保保存目录存在
    if save_base_dir and sample_id:
        sample_dir = os.path.join(save_base_dir, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

    # 存储每个层的结果
    layer_results = {}

    # 处理每个层
    for layer_num in layers:
        # print(f"处理第 {layer_num} 层...")

        with torch.set_grad_enabled(True):
            gradcams, _ = compute_gradcam(model=model,
                                          visual_input=image,
                                          text_input=question,
                                          tokenized_text=tokenized_text,
                                          block_num=layer_num)

        gradcams = [gradcam_[1] for gradcam_ in gradcams]
        gradcams1 = torch.stack(gradcams).reshape(image.size(0), -1)
        itc_score_tensor = model({"image": image, "text_input": question}, match_head='itc')

        # 将ITC分数转换为标量
        if isinstance(itc_score_tensor, torch.Tensor):
            itc_score = itc_score_tensor.item()
        else:
            itc_score = itc_score_tensor

        ratio = 1 - itc_score / 2
        ratio = min(ratio, 1 - 10 ** (-5))
        resized_img = raw_image.resize((384, 384))
        norm_img = np.float32(resized_img) / 255
        gradcam = gradcams1.reshape(24, 24)

        # 生成增强图像
        avg_gradcam = getAttMap(norm_img, gradcam.cpu().numpy(), blur=True, overlap=False)
        temp, _ = torch.sort(torch.tensor(avg_gradcam).reshape(-1), descending=True)
        cam1 = torch.tensor(avg_gradcam).unsqueeze(2)
        cam = torch.cat([cam1, cam1, cam1], dim=2)

        mask = torch.where(cam < temp[int(384 * 384 * ratio)], 0, 1)
        new_image = tensor_image.permute(1, 2, 0) * mask
        unloader = transforms.ToPILImage()
        imag = new_image.clone().permute(2, 0, 1)
        augmented_image = unloader(imag)

        # 生成叠加图
        overlay_image = create_layer_overlay(raw_image, gradcam, alpha=0.2)

        # 存储结果
        layer_results[layer_num] = {
            'augmented_image': augmented_image,
            'overlay_image': overlay_image,
            'itc_score': itc_score,
            'gradcam': gradcam
        }

    # 创建多图层对比图
    if save_base_dir and sample_id:
        create_multi_layer_comparison(layer_results, raw_image, sample_dir, problem)

    return layer_results


def create_layer_overlay(raw_image, patch_heatmap, alpha=0.8):
    """
    为单个层创建叠加图
    """
    # 确保原图是384x384
    raw_img_resized = raw_image.resize((384, 384))
    raw_img_np = np.array(raw_img_resized)

    # 将24x24热力图上采样到384x384
    patch_heatmap_np = patch_heatmap.cpu().numpy()
    heatmap_upscaled = cv2.resize(patch_heatmap_np, (384, 384), interpolation=cv2.INTER_NEAREST)

    # 归一化热力图
    if heatmap_upscaled.max() > heatmap_upscaled.min():
        heatmap_normalized = (heatmap_upscaled - heatmap_upscaled.min()) / (
                    heatmap_upscaled.max() - heatmap_upscaled.min())
    else:
        heatmap_normalized = heatmap_upscaled

    # 创建热力图颜色映射
    heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # 将原图转换为浮点数
    raw_img_float = raw_img_np.astype(np.float32) / 255.0

    # 创建叠加图像
    overlay = raw_img_float * (1 - alpha) + heatmap_colored.astype(np.float32) / 255.0 * alpha
    overlay = (overlay * 255).astype(np.uint8)

    # 添加网格线
    patch_size = 384 // 24
    for i in range(0, 384, patch_size):
        cv2.line(overlay, (i, 0), (i, 383), (255, 255, 255), 1)
        cv2.line(overlay, (0, i), (383, i), (255, 255, 255), 1)

    return Image.fromarray(overlay)


def create_multi_layer_comparison(layer_results, raw_image, sample_dir, problem):
    """
    创建多图层对比图，第一行是叠加图，第二行是增强图像
    """
    layers = list(layer_results.keys())
    n_layers = len(layers)

    # 计算子图布局
    fig_width = min(5 * n_layers, 20)  # 限制最大宽度
    fig_height = 10

    # 创建图形
    fig, axes = plt.subplots(2, n_layers, figsize=(fig_width, fig_height))

    # 如果只有一层，确保axes是二维数组
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    # 添加问题文本
    fig.suptitle(f"Problem: {problem}", fontsize=16, y=0.95)

    # 第一行：叠加图
    for i, layer_num in enumerate(layers):
        ax = axes[0, i]
        overlay_image = layer_results[layer_num]['overlay_image']
        ax.imshow(np.array(overlay_image))
        ax.set_title(f"Layer {layer_num} Overlay\n(ITC: {layer_results[layer_num]['itc_score']:.4f})",
                     fontsize=12)
        ax.axis('off')

    # 第二行：增强图像
    for i, layer_num in enumerate(layers):
        ax = axes[1, i]
        augmented_image = layer_results[layer_num]['augmented_image']
        ax.imshow(np.array(augmented_image))
        ax.set_title(f"Layer {layer_num} Augmented", fontsize=12)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    plt.savefig(os.path.join(sample_dir, "multi_layer_comparison.png"),
                bbox_inches='tight', dpi=150)
    plt.close()

    # 单独保存每个层的结果
    for layer_num, results in layer_results.items():
        # 保存叠加图
        results['overlay_image'].save(os.path.join(sample_dir, f"layer_{layer_num}_overlay.png"))

        # 保存增强图像
        results['augmented_image'].save(os.path.join(sample_dir, f"layer_{layer_num}_augmented.png"))

        # 保存热力图数据
        np.save(os.path.join(sample_dir, f"layer_{layer_num}_heatmap.npy"),
                results['gradcam'].cpu().numpy())

    # print(f"多图层对比图已保存到: {os.path.join(sample_dir, 'multi_layer_comparison.png')}")