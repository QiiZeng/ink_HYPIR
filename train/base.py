import logging
import os
import shutil
from pathlib import Path
from typing import overload, List, Dict
import importlib
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.serialization import get_unsafe_globals_in_checkpoint, add_safe_globals
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import transformers
import lpips
import diffusers
from diffusers import AutoencoderKL
from PIL import Image

from HYPIR.model.D import ImageConvNextDiscriminator
from HYPIR.utils.common import instantiate_from_config, log_txt_as_img, print_vram_state, SuppressLogging
from HYPIR.utils.ema import EMAModel
from HYPIR.utils.tabulate import tabulate
import numpy as np

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class BatchInput:

    def __init__(self,** kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ValueError(f"Duplicated key in BatchInput: {name}")
        self.__dict__[name] = value

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value


class BaseTrainer:

    def __init__(self, config):
        self.config = config
        set_seed(config.seed)
        self.init_environment()
        self.init_models()
        self.summary_models()
        self.init_optimizers()
        self.init_dataset()
        self.prepare_all()
        
        # 添加简单监控
        self.last_real_logit = 0.5
        self.last_fake_logit = 0.5
        self.blur_history = []  # 记录模糊程度

        # ========== 2新增：笔画相关监控 ==========
        self.current_stroke_weight = 0.0  # 当前笔画损失权重
        self.current_stroke_improvement = 0.0  # 当前笔画改善度
        self.stroke_improvement_history = []  # 笔画改善度历史记录

    def init_environment(self):
        logging_dir = Path(self.config.output_dir, self.config.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.config.output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with=self.config.report_to,
            project_config=accelerator_project_config,
            mixed_precision=self.config.mixed_precision,
        )
        logger.info(accelerator.state, main_process_only=True)
        if accelerator.is_main_process:
            accelerator.init_trackers("train")
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        if accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.device = accelerator.device
        torch.backends.cuda.matmul.allow_tf32 = True  # 开启TF32（A100/3090/4090必开）
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # 卷积自动选最优算法

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        return model

    def init_models(self):
        self.init_scheduler()
        self.init_text_models()
        self.init_vae()
        self.init_generator()
        self.init_discriminator()
        self.init_lpips()

    @overload
    def init_scheduler(self):
        ...

    @overload
    def init_text_models(self):
        ...

    @overload
    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        ...

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model_path, 
            subfolder="vae", 
            torch_dtype=self.weight_dtype,
            local_files_only=True,
            cache_dir=None
        ).to(self.device)
        self.vae.eval().requires_grad_(False)

    def init_lpips(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        self.net_lpips.eval().requires_grad_(False)

    @overload
    def init_generator(self):
        ...

    def init_discriminator(self):
        ctx = (
            nullcontext()
            if self.accelerator.is_local_main_process
            else SuppressLogging(logging.WARNING)
        )
        with ctx:
            self.D = ImageConvNextDiscriminator(precision="fp32").to(device=self.device)
        self.D.train().requires_grad_(True)

    def summary_models(self):
        table_data = []
        for attr, value in self.__dict__.items():
            if not isinstance(value, torch.nn.Module):
                continue
            model = value
            model_type = type(model).__name__
            total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
            learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
            table_data.append([attr, model_type, f"{total_params:.2f}", f"{learnable_params:.2f}"])
        headers = ["Model Name", "Model Type", "Total Parameters (M)", "Learnable Parameters (M)"]
        table = tabulate(table_data, headers=headers, tablefmt="pretty")
        logger.info(f"Model Summary:\n{table}")

    def init_optimizers(self):
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        if self.config.optimizer_type == "AdamW":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = None

        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        self.G_opt = optimizer_cls(
            self.G_params,
            lr=self.config.lr_G,
            **self.config.opt_kwargs,
        )

        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        self.D_opt = optimizer_cls(
            self.D_params,
            lr=self.config.lr_D,
            **self.config.opt_kwargs,
        )

    def init_dataset(self):
        data_cfg = self.config.data_config
        dataset = instantiate_from_config(data_cfg.train.dataset)
        torch.multiprocessing.set_start_method('fork', force=True)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=data_cfg.train.batch_size,
            num_workers=min(8, os.cpu_count()), 
            pin_memory=True,
            prefetch_factor=4,
            multiprocessing_context="fork",
            persistent_workers=True  # 新增：保持workers进程，避免重复创建
        )
        self.batch_transform = instantiate_from_config(data_cfg.train.batch_transform)

    def prepare_all(self):
        logger.info("Wrapping models, optimizers and dataloaders")
        attrs = ["G", "D", "G_opt", "D_opt", "dataloader"]
        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)
        print_vram_state("After accelerator.prepare", logger=logger)

    def force_optimizer_ckpt_safe(self, checkpoint_dir):
        def get_symbol(s):
            module_name, symbol_name = s.rsplit('.', 1)
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name)
            return symbol

        for file_name in os.listdir(checkpoint_dir):
            if "optimizer" in file_name and not file_name.endswith("safetensors"):
                path = os.path.join(checkpoint_dir, file_name)
                unsafe_globals = get_unsafe_globals_in_checkpoint(path)
                logger.info(f"Unsafe globals in {path}: {unsafe_globals}")
                unsafe_globals = list(map(get_symbol, unsafe_globals))
                add_safe_globals(unsafe_globals)

    def attach_accelerator_hooks(self):
        ...

    def on_training_start(self):
        logger.info(f"Creating EMA handler, Use EMA = {self.config.use_ema}, EMA decay = {self.config.ema_decay}")
        if self.config.resume_from_checkpoint is not None and self.config.resume_ema:
            ema_resume_pth = os.path.join(self.config.resume_from_checkpoint, "ema_state_dict.pth")
        else:
            ema_resume_pth = None
        self.ema_handler = EMAModel(
            self.unwrap_model(self.G),
            decay=self.config.ema_decay,
            use_ema=self.config.use_ema,
            ema_resume_pth=ema_resume_pth,
            verbose=self.accelerator.is_local_main_process,
        )

        global_step = 0
        if self.config.resume_from_checkpoint:
            path = self.config.resume_from_checkpoint
            ckpt_name = os.path.basename(path)
            logger.info(f"Resuming from checkpoint {path}")
            self.force_optimizer_ckpt_safe(path)
            self.accelerator.load_state(path)
            global_step = int(ckpt_name.split("-")[1])
            init_global_step = global_step
        else:
            init_global_step = 0

        self.global_step = global_step
        self.pbar = tqdm(
            range(0, self.config.max_train_steps),
            initial=init_global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )
        self.d_train_counter = 0

    def prepare_batch_inputs(self, batch):
        batch = self.batch_transform(batch)
        gt = (batch["GT"] * 2 - 1).float()
        lq = (batch["LQ"] * 2 - 1).float()
        bs = gt.shape[0]
        text_embed = torch.zeros(bs, 77, 1024, device=self.device, dtype=self.weight_dtype)
        c_txt = {"text_embed": text_embed}
        
        z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
        timesteps = torch.full((bs,), self.config.model_t, dtype=torch.long, device=self.device)
        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_lq=z_lq,
            c_txt=c_txt,
            timesteps=timesteps,
            x_hq=gt,
        )

    @overload
    def forward_generator(self) -> torch.Tensor:
        ...

    def compute_blur_metric(self, image):
        """计算图像模糊程度"""
        with torch.no_grad():
            # 1. 转换为灰度图（彻底解决通道数问题，同时不影响模糊度计算）
            if image.shape[1] == 3:
                # RGB转灰度：符合人眼感知的权重
                gray_image = 0.2989 * image[:, 0:1, :, :] + 0.5870 * image[:, 1:2, :, :] + 0.1140 * image[:, 2:3, :, :]
            else:
                gray_image = image  # 已经是单通道
            
            # 2. 定义单通道卷积核（完全匹配灰度图输入）
            kernel_x = torch.tensor([[[[-1, 0, 1]]]], device=gray_image.device, dtype=gray_image.dtype)  # shape=(1,1,1,3)
            kernel_y = torch.tensor([[[[-1], [0], [1]]]], device=gray_image.device, dtype=gray_image.dtype)  # shape=(1,1,3,1)
            
            # 3. 计算梯度（无分组，通道数完全匹配）
            grad_x = F.conv2d(gray_image, kernel_x, padding=(0, 1))
            grad_y = F.conv2d(gray_image, kernel_y, padding=(1, 0))
        
            # 4. 计算模糊分数
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            blur_score = 1.0 / (gradient_magnitude.mean() + 1e-8)
            return blur_score.item()

    def compute_stroke_improvement(self, pred, gt):
        """计算笔画改善程度（简单版本）"""
        with torch.no_grad():
            laplacian_kernel = torch.tensor([
                [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
            ], device=pred.device, dtype=pred.dtype)
            laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1)
            
            pred_edges = F.conv2d(pred, laplacian_kernel, padding=1, groups=3).abs()
            gt_edges = F.conv2d(gt, laplacian_kernel, padding=1, groups=3).abs()
            
            pred_norm = pred_edges / (pred_edges.max() + 1e-8)
            gt_norm = gt_edges / (gt_edges.max() + 1e-8)
            
            similarity = 1.0 - F.l1_loss(pred_norm, gt_norm)
            return similarity.item()

    def get_dynamic_stroke_weight(self):
        current_step = self.global_step
        max_steps = self.config.max_train_steps
        
        if max_steps <= 5000:
            start_step = 1000
        elif max_steps <= 10000:
            start_step = 2000
        else:
            start_step = 3000
        
        if current_step < start_step:
            return 0.1
        
        relative_step = current_step - start_step
        
        if relative_step < 3000:
            weight = 0.2 + 0.4 * (relative_step / 3000)
        else:
            weight = 0.6
        
        if hasattr(self, 'blur_oscillation_count') and self.blur_oscillation_count > 0:
            weight = weight * 0.8
        
        return min(0.8, max(0.1, weight))

    def compute_stroke_loss(self, pred, gt):
        stroke_loss = 0
        scales = [1, 2]
    
        for scale in scales:
            if scale > 1:
                pred_scaled = F.avg_pool2d(pred, scale)
                gt_scaled = F.avg_pool2d(gt, scale)
            else:
                pred_scaled, gt_scaled = pred, gt
        
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                                  device=pred.device, dtype=pred.dtype).repeat(3,1,1,1)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                                  device=pred.device, dtype=pred.dtype).repeat(3,1,1,1)
            
            pred_grad_x = F.conv2d(pred_scaled, sobel_x, padding=1, groups=3)
            pred_grad_y = F.conv2d(pred_scaled, sobel_y, padding=1, groups=3)
            gt_grad_x = F.conv2d(gt_scaled, sobel_x, padding=1, groups=3)
            gt_grad_y = F.conv2d(gt_scaled, sobel_y, padding=1, groups=3)

            pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
            gt_grad = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-8)

            pred_norm = pred_grad / (pred_grad.max() + 1e-8)
            gt_norm = gt_grad / (gt_grad.max() + 1e-8)

            scale_loss = F.l1_loss(pred_norm, gt_norm)
            stroke_loss += scale_loss / scale

        return stroke_loss / len(scales)

    def optimize_generator(self):
        with self.accelerator.accumulate(self.G):
            self.unwrap_model(self.D).eval().requires_grad_(False)
            x = self.forward_generator()
            self.G_pred = x
            
            # 核心修复：使用L1损失代替MSE，减少模糊
            # L1损失比MSE更能保留边缘细节
            if hasattr(self.config, 'use_l1_loss') and self.config.use_l1_loss:
                loss_l1 = F.l1_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
                pixel_loss = loss_l1
            else:
                # 或者使用感知损失加权的MSE
                with torch.no_grad():
                    # 计算边缘重要性权重
                    # 修复：适配3通道输入的边缘检测卷积核
                    edge_kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], device=self.device, dtype=self.batch_inputs.gt.dtype)
                    edge_kernel = edge_kernel.repeat(3, 1, 1, 1)  # 扩展为3通道
                    edge_gt = F.conv2d(self.batch_inputs.gt, edge_kernel, padding=1, groups=3).abs()
                    # 边缘区域权重更高
                    edge_weight = (edge_gt / (edge_gt.max() + 1e-8)) * 2.0 + 1.0
                    loss_l2 = (F.mse_loss(x, self.batch_inputs.gt, reduction='none') * edge_weight).mean() * self.config.lambda_l2
                pixel_loss = loss_l2
                
            loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips

            loss_stroke = torch.tensor(0.0, device=self.device)
            if hasattr(self.config, 'use_stroke_loss') and self.config.use_stroke_loss:
                stroke_weight = self.get_dynamic_stroke_weight()
                loss_stroke = self.compute_stroke_loss(x, self.batch_inputs.gt) * stroke_weight
                stroke_improvement = self.compute_stroke_improvement(x, self.batch_inputs.gt)
                self.stroke_improvement_history.append(stroke_improvement)
                if len(self.stroke_improvement_history) > 50:
                    self.stroke_improvement_history.pop(0)
                self.current_stroke_weight = stroke_weight
                self.current_stroke_improvement = stroke_improvement
            
            blur_score = self.compute_blur_metric(x)
            self.blur_history.append(blur_score)
            if len(self.blur_history) > 100:
                self.blur_history.pop(0)
            
            avg_blur = sum(self.blur_history) / len(self.blur_history) if self.blur_history else 1.0
            dynamic_lambda_gan = self.config.lambda_gan * min(2.0, avg_blur)
            
            loss_disc = self.D(x, for_G=True).mean() * dynamic_lambda_gan
            
            loss_G = pixel_loss + loss_lpips + loss_stroke + loss_disc
            if hasattr(self, 'last_blur_score'):
                blur_change = abs(blur_score - self.last_blur_score)
                if blur_change > 0.5:
                    stability_loss = blur_change * 0.05
                    loss_G = loss_G + stability_loss
    
            self.last_blur_score = blur_score
            
            if hasattr(self.config, 'edge_sharpening') and self.config.edge_sharpening:
                # 鼓励生成器输出更锐利的边缘
                with torch.no_grad():
                    # 修复：适配3通道的梯度卷积核
                    grad_kernel = torch.tensor([[[[-1, 0, 1]]]], device=self.device, dtype=self.batch_inputs.gt.dtype)
                    grad_kernel = grad_kernel.repeat(3, 1, 1, 1)  # 扩展为3通道
                    edge_gt_grad = torch.abs(F.conv2d(self.batch_inputs.gt, grad_kernel, padding=(0,1), groups=3))
                # 同样修复生成图像的梯度计算
                edge_pred_grad = torch.abs(F.conv2d(x, grad_kernel, padding=(0,1), groups=3))
                edge_loss = F.mse_loss(edge_pred_grad, edge_gt_grad) * 0.1
                loss_G = loss_G + edge_loss
            
            self.accelerator.backward(loss_G)
            if self.accelerator.sync_gradients:
                # FIX: 使用当前模型的参数进行梯度裁剪
                current_G_params = [p for p in self.G.parameters() if p.requires_grad]
                self.accelerator.clip_grad_norm_(current_G_params, self.config.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()
            
            self.current_blur_score = blur_score    

        loss_dict = dict(
            G_total=loss_G, 
            G_mse=pixel_loss, 
            G_lpips=loss_lpips,
            G_disc=loss_disc, 
            G_blur=blur_score
        )
        
        if hasattr(self.config, 'use_stroke_loss') and self.config.use_stroke_loss:
            loss_dict['G_stroke'] = loss_stroke
            loss_dict['G_stroke_weight'] = torch.tensor(self.current_stroke_weight, device=self.device)
            loss_dict['G_stroke_improvement'] = torch.tensor(self.current_stroke_improvement, device=self.device)
        
        return loss_dict

    def optimize_discriminator(self):
        gt = self.batch_inputs.gt
        with torch.no_grad():
            x = self.forward_generator()
        self.G_pred = x
        
        with self.accelerator.accumulate(self.D):
            self.unwrap_model(self.D).train().requires_grad_(True)
            
            # 核心修复：减少噪声强度，让判别器关注细节
            noise_strength = 0.01  # 从0.05减少到0.01
            # 只对生成图像加噪声，真实图像保持清晰
            gt_noisy = gt  # 真实图像不加噪声
            x_noisy = x + torch.randn_like(x) * noise_strength
            
            # 屏蔽特征匹配（判别器不支持return_features）
            feature_loss = 0
            
            # FIX: 更健壮的判别器返回值处理
            real_output = self.D(gt_noisy, for_real=True)
            fake_output = self.D(x_noisy, for_real=False)
            
            if isinstance(real_output, tuple):
                loss_D_real, real_logits = real_output
            else:
                loss_D_real = real_output
                real_logits = loss_D_real.detach()   # fallback
            
            if isinstance(fake_output, tuple):
                loss_D_fake, fake_logits = fake_output
            else:
                loss_D_fake = fake_output
                fake_logits = loss_D_fake.detach()
            
            # 统一转换为标量
            loss_D_real = loss_D_real.mean() if loss_D_real.numel() > 1 else loss_D_real
            loss_D_fake = loss_D_fake.mean() if loss_D_fake.numel() > 1 else loss_D_fake
            
            if hasattr(self.config, 'use_gradient_penalty') and self.config.use_gradient_penalty:
                try:
                    alpha = torch.rand(gt.size(0), 1, 1, 1, device=gt.device)
                    interpolated = alpha * gt_noisy + (1 - alpha) * x_noisy.detach()
                    interpolated.requires_grad_(True)
                    d_interpolated = self.D(interpolated, for_real=True)
                    if isinstance(d_interpolated, tuple):
                        d_interpolated = d_interpolated[0]
                    gradients = torch.autograd.grad(
                        outputs=d_interpolated,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(d_interpolated),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    gp_weight = self.config.gp_weight if hasattr(self.config, 'gp_weight') else 10.0
                except:
                    # 梯度惩罚计算失败时兜底
                    gradient_penalty = 0
                    gp_weight = 0
            else:
                gradient_penalty = 0
                gp_weight = 0
            
            loss_D = loss_D_real + loss_D_fake + gp_weight * gradient_penalty + feature_loss
            
            self.accelerator.backward(loss_D)
            if self.accelerator.sync_gradients:
                # 温和的梯度裁剪（仅裁剪有梯度的参数）
                valid_params = [p for p in self.D_params if p.grad is not None]
                if valid_params:
                    self.accelerator.clip_grad_norm_(valid_params, max_norm=2.0)
            self.D_opt.step()
            self.D_opt.zero_grad()
            
        loss_dict = dict(D=loss_D)
        with torch.no_grad():
            # FIX: 安全计算logits均值
            real_logits_mean = real_logits.mean() if hasattr(real_logits, 'mean') else real_logits
            fake_logits_mean = fake_logits.mean() if hasattr(fake_logits, 'mean') else fake_logits
            self.last_real_logit = real_logits_mean.item() if torch.is_tensor(real_logits_mean) else real_logits_mean
            self.last_fake_logit = fake_logits_mean.item() if torch.is_tensor(fake_logits_mean) else fake_logits_mean
            
        loss_dict.update(dict(D_logits_real=real_logits_mean, D_logits_fake=fake_logits_mean))
        return loss_dict

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        self.blur_oscillation_count = 0
        self.recent_blur_scores = []
        
        d_skip_counter = 0
        warmup_steps = 500
        
        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)
                bs = len(self.batch_inputs.lq)
                
                # 动态决定训练哪个模型
                generator_step = True  # 默认训练生成器
                
                # 预热阶段：前500步主要训练生成器（避免初始值干扰）
                if self.global_step < warmup_steps:
                    generator_step = (self.batch_count % 3) != 0
                else:
                    if hasattr(self, 'current_blur_score'):
                        if self.current_blur_score > 5.0:
                            generator_step = False
                    
                    # 根据判别器性能动态调整
                    diff = abs(self.last_real_logit - self.last_fake_logit)
                    if diff < 0.2:  # 判别器区分能力太弱
                        generator_step = False
                        d_skip_counter = 0
                    elif diff > 0.6:  # 判别器太强
                        generator_step = True
                        d_skip_counter += 1
                        if d_skip_counter >= 2:
                            generator_step = False
                            d_skip_counter = 0
                    else:
                        generator_step = (self.batch_count % 2) == 0  # 1:1的比例
                
                if generator_step:
                    loss_dict = self.optimize_generator()
                else:
                    if self.accelerator.sync_gradients:
                        self.d_train_counter += 1
                    loss_dict = self.optimize_discriminator()

                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                        if k not in train_loss:
                            train_loss[k] = 0
                        train_loss[k] += avg_loss.item() / self.config.gradient_accumulation_steps

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    if generator_step:
                        self.ema_handler.update()
                    
                    # 显示更多训练信息
                    diff = abs(self.last_real_logit - self.last_fake_logit)
                    state = "Generator" if generator_step else "Discriminator"
                    blur_info = f"Blur: {self.current_blur_score:.2f}" if hasattr(self, 'current_blur_score') else ""
                    stroke_info = ""
                    if hasattr(self, 'current_stroke_weight'):
                        stroke_info = f"StrokeW: {self.current_stroke_weight:.3f}"
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"{state} Step, D_diff: {diff:.3f} {blur_info}, VRAM: {peak:.2f}GB")

                if self.accelerator.sync_gradients and not generator_step:
                    self.global_step += 1
                    self.pbar.update(1)
                    
                    # 记录额外指标
                    log_dict = {}
                    for k in train_loss.keys():
                        log_dict[f"loss/{k}"] = train_loss[k]
                    
                    # 记录判别器状态和模糊程度
                    log_dict["loss/D_diff"] = abs(self.last_real_logit - self.last_fake_logit)
                    log_dict["loss/D_real"] = self.last_real_logit
                    log_dict["loss/D_fake"] = self.last_fake_logit
                    if hasattr(self, 'current_blur_score'):
                        log_dict["loss/G_blur"] = self.current_blur_score
                    if hasattr(self, 'current_stroke_weight'):
                        log_dict["config/stroke_weight"] = self.current_stroke_weight
                    if hasattr(self, 'current_stroke_improvement'):
                        log_dict["metric/stroke_improvement"] = self.current_stroke_improvement
                    
                    train_loss = {}
                    self.accelerator.log(log_dict, step=self.global_step)
                    
                    # 精简日志频率，减少IO
                    if self.global_step % max(self.config.log_image_steps, 500) == 0 or self.global_step == 1:
                        self.log_images()
                        
                    if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 1:
                        self.save_checkpoint()

                if self.global_step >= self.config.max_train_steps:
                    break
            
        self.accelerator.end_training()

    def log_images(self):
        N = 4
        # FIX: 确保 self.G_pred 已定义
        if not hasattr(self, 'G_pred') or self.G_pred is None:
            return
        
        image_logs = dict(
            lq=(self.batch_inputs.lq[:N] + 1) / 2,
            gt=(self.batch_inputs.gt[:N] + 1) / 2,
            G=(self.G_pred[:N] + 1) / 2,
        )
        if self.config.use_ema:
            self.ema_handler.activate_ema_weights()
            with torch.no_grad():
                ema_x = self.forward_generator()
                image_logs["G_ema"] = (ema_x[:N] + 1) / 2
            self.ema_handler.deactivate_ema_weights()

        if not self.accelerator.is_main_process:
            return

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in image_logs.items():
                    tracker.writer.add_image(
                        f"image/{tag}",
                        make_grid(images.float(), nrow=4),
                        self.global_step,
                    )

        # 减少磁盘IO：只在关键步骤保存图像
        if self.global_step % max(self.config.log_image_steps, 500) == 0:
            for key, images in image_logs.items():
                image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8) \
                    .permute(0, 2, 3, 1).contiguous().cpu().numpy()
                save_dir = os.path.join(
                    self.config.output_dir, self.config.logging_dir, "log_images", f"{self.global_step:07}", key)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for i, img in enumerate(image_arrs):
                    Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))

    def log_grads(self):
        if self.global_step % max(self.config.log_grad_steps, 500) != 0:
            return
            
        self.unwrap_model(self.D).eval().requires_grad_(False)

        x = self.forward_generator()
        loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
        loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips
        loss_disc = self.D(x, for_G=True).mean() * self.config.lambda_gan
        losses = [("l2", loss_l2), ("lpips", loss_lpips), ("disc", loss_disc)]
        grad_dict = {}
        self.G_opt.zero_grad()
        for idx, (name, loss) in enumerate(losses):
            retain_graph = idx != len(losses) - 1
            loss.backward(retain_graph=retain_graph)
            lora_module_grads = {}
            for module_name, module in self.unwrap_model(self.G).named_modules():
                for suffix in self.config.log_grad_modules:
                    if module_name.endswith(suffix):
                        flat_grad = torch.cat([
                            p.grad.flatten() for p in module.parameters() if p.requires_grad and p.grad is not None
                        ])
                        lora_module_grads.setdefault(suffix, []).append(flat_grad)
                        break
            for k, v in lora_module_grads.items():
                if v:
                    grad_dict[f"grad_norm/{k}_{name}"] = torch.norm(torch.cat(v)).item()
            self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.config.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.config.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                if len(checkpoints) >= self.config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(self.config.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

            self.ema_handler.save_ema_weights(save_path)
            logger.info(f"Saved ema weights to {save_path}")
