import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler
from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings('ignore')


def freeze_for_stage2(model):
    print("second stage: CLIP remain freezed, CrossAttention and LLM LoRA are trainable.")
    for name, param in model.named_parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)

    # PEFT freezes non-adapter parameters; keep the small visual bridge and
    # cross-attention gates/norms trainable for dense VLM alignment.
    for name, param in model.named_parameters():
        if "vision_proj" in name or "cross_attn_norm" in name or "gate_alpha" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    did_optimizer_step = False
    last_step = start_step
    if start_step > 0:
        for _ in range(start_step):
            try:
                next(data_iter)
            except StopIteration:
                Logger(f'Epoch [{epoch + 1}/{args.epochs}] 可用batch不足，无法从step {start_step + 1}恢复，已跳过本轮。')
                return
    for step, (input_ids, labels, attention_mask, pixel_values) in enumerate(data_iter, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, attention_mask=attention_mask, pixel_values=pixel_values)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            did_optimizer_step = True

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items()
                if 'vision_encoder.' not in key
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}  # 半精度保存并移到CPU
            torch.save(clean_state_dict, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        did_optimizer_step = True

    return did_optimizer_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_vlm', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_i2t.parquet", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--import_path", type=str, default="../out", help="导入权重路径")
    parser.add_argument("--clip_path", type=str, default="../model/vision_model/clip-vit-base-patch16", help="CLIP视觉模型路径")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000, help="流式数据shuffle buffer大小，0表示关闭")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-SFT", help="wandb项目名")
    parser.add_argument("--alignment_type", type=str, default="cross_attn", choices=["cross_attn", "token"], help="视觉对齐方式")
    parser.add_argument("--cross_attn_layers", type=str, default=None, help="交叉注意力层: all/upper_half/every_2/逗号分隔层号")
    parser.add_argument("--cross_attn_every", type=int, default=2, help="未指定cross_attn_layers时，每隔多少层插入交叉注意力")
    parser.add_argument("--cross_attn_start_layer", type=int, default=None, help="未指定cross_attn_layers时，从第几层开始插入")
    parser.add_argument("--cross_attn_gate_init", type=float, default=0.1, help="交叉注意力sigmoid门控初值")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, 
                           max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe),
                           alignment_type=args.alignment_type,
                           cross_attn_layers=args.cross_attn_layers,
                           cross_attn_every=args.cross_attn_every,
                           cross_attn_start_layer=args.cross_attn_start_layer,
                           cross_attn_gate_init=args.cross_attn_gate_init)
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb.login(key="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")
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        device=args.device,
        vision_model_path=args.clip_path,
        import_path=args.import_path
    )
    model = freeze_for_stage2(model)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = VLMDataset(args.data_path, tokenizer, preprocess=preprocess,
                          image_special_token=vlm_config.image_special_token,
                          max_length=vlm_config.max_seq_len,
                          shuffle_buffer_size=args.shuffle_buffer_size,
                          seed=42 + (dist.get_rank() if dist.is_initialized() else 0))
    is_iterable_ds = isinstance(train_ds, IterableDataset)
    train_sampler = (DistributedSampler(train_ds) if (dist.is_initialized() and not is_iterable_ds) else None)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        setup_seed(42 + epoch)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        if is_iterable_ds:
            loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

            batch = next(iter(loader))
            input_ids, labels, attention_mask, pixel_values = batch
            print("valid ratio:", (labels != -100).float().mean().item())
            print("attention valid ratio:", attention_mask.float().mean().item())
            print("valid labels:", (labels != -100).sum().item())
            
            if skip > 0:
                Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 流式数据模式，跳过前{start_step}个step，从step {start_step + 1}开始')
                train_epoch(epoch, loader, len(loader), start_step, wandb)
            else:
                train_epoch(epoch, loader, len(loader), 0, wandb)
            continue

        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
