import os
import time
import sys
from collections import defaultdict
import random
import torch
import transformers
# import slurm
import logging
import util
import numpy as np
from tqdm.auto import tqdm

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from options import Options
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
# import evaluation

# ACFiD specific
from fidt5_ac import ACFiDT5, T5Config
import data_ac as data

logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Initialise wandb
try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False


def log_scalar(name, value, step):
    tb_logger.add_scalar(name, value, step)
    if _has_wandb:
        wandb.log({name: value, "step": step})


def train_evaluate(model, optimizer, scheduler, global_step,
                   train_dataset, dev_dataset, opt, collator_function, best_metric):
    train_sampler = (RandomSampler(train_dataset) if opt.local_rank == -1 or opt.world_size == 1
                     else DistributedSampler(train_dataset))
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=opt.per_gpu_batch_size, drop_last=True, num_workers=3,
                                  collate_fn=collator_function)

    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=opt.per_gpu_batch_size,
                                drop_last=True, num_workers=3, collate_fn=collator_function)

    # Freeze the FiD parameters and only train AC part.
    trainable_np = list(model.named_parameters())
    if opt.freeze_fid_params:
        new_np = []
        for n, p in trainable_np:
            if n.startswith("encoder.has_answer_heads") or n.startswith("ac_scheduler"):
                p.requires_grad = True
                new_np.append((n, p))
            else:
                p.requires_grad = False
        trainable_np = new_np

    if opt.freeze_has_answer_heads:
        new_np = []
        for n, p in trainable_np:
            if n.startswith("encoder.has_answer_heads"):
                p.requires_grad = False
            else:
                p.requires_grad = True
                new_np.append((n, p))
        trainable_np = new_np

    trainable_parameters = [p for n, p in trainable_np]
    # Prepare optimizer and schedule (linear warmup and decay)
    if optimizer is None or scheduler is None:
        optimizer = torch.optim.Adam(trainable_parameters, lr=opt.lr)
        scheduler = util.FixedScheduler(optimizer)

    # fp16
    if opt.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.fp16_opt_level)

    # Distributed training
    if opt.world_size > 1 and opt.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", opt.per_gpu_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        opt.train_batch_size * opt.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", opt.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", opt.total_step)
    logger.info("  Total number of training epochs = %f",
                opt.total_step * opt.train_batch_size * opt.gradient_accumulation_steps / len(train_dataset))

    loss, curr_loss = 0.0, 0.0
    epoch = 0
    step = 0
    model.train()
    model.zero_grad()
    while global_step < opt.total_step:
        epoch += 1
        if opt.world_size > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in tqdm(enumerate(train_dataloader)):
            step += 1

            # Process the inputs
            idx, answer_ids, answer_mask, context_ids, context_mask, has_answer_labels = batch
            answer_ids, answer_mask = answer_ids.cuda(), answer_mask.bool().cuda()
            has_answer_labels = has_answer_labels.cuda()
            labels = answer_ids.masked_fill(~answer_mask, -100)
            if hasattr(model, "module"):
                model.module.encoder.n_passages = context_ids.size(1)
            else:
                model.encoder.n_passages = context_ids.size(1)
            context_ids = context_ids.cuda().view(context_ids.size(0), -1)
            context_mask = context_mask.cuda().view(context_ids.size(0), -1)
            decoder_input_ids = None

            inputs = {
                'input_ids': context_ids,
                'attention_mask': context_mask,
                'decoder_attention_mask': answer_mask,
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'has_answer_labels': has_answer_labels,
            }

            # Run the model
            outputs = model(**inputs)
            train_loss = outputs[0]
            train_loss = util.average_master(train_loss, opt)

            if opt.gradient_accumulation_steps > 1:
                train_loss = train_loss / opt.gradient_accumulation_steps

            if opt.fp16:
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                train_loss.backward()

            curr_loss += train_loss.item()
            if step % opt.gradient_accumulation_steps == 0:
                # util.clip_gradients(model, opt.clip)
                if opt.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, opt.clip)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if opt.is_master and global_step % opt.log_freq == 0:
                    logger.info(
                        f"{global_step} / {opt.total_step} -- train loss = {curr_loss / opt.log_freq:.3f}"
                        f" | lr = {scheduler.get_last_lr()[0]:.5f}"
                    )
                    log_scalar("Train/Loss", curr_loss / opt.log_freq, global_step)
                    curr_loss = 0.

                if global_step % opt.eval_freq == 0:
                    results = evaluate(model, dev_dataset, dev_dataloader, tokenizer, opt)
                    dev_f1 = results["avg_f1"]  # use average F1 (across all layers) as evaluation metric
                    if opt.is_master:
                        logger.info(f"{global_step} / {opt.total_step} -- dev evaluation = {100 * dev_f1:.2f} F1")
                        for k, v in results.items():
                            log_scalar(f"Dev/{k}", v, global_step)

                    if dev_f1 > best_metric:
                        best_metric = dev_f1
                        if opt.is_master:
                            model_to_save = model.module if hasattr(model, "module") else model
                            util.save(model_to_save, optimizer, scheduler, global_step, best_metric, opt, dir_path,
                                      'best_dev')
                    model.train()

                if opt.is_master and global_step % opt.save_freq == 0:
                    model_to_save = model.module if hasattr(model, "module") else model
                    util.save(model_to_save, optimizer, scheduler, global_step, best_metric, opt, dir_path,
                              f"step-{global_step}")
                if global_step > opt.total_step:
                    break


def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module

    num_layers = model.encoder.config.num_layers
    all_results = [defaultdict(list) for _ in range(num_layers)]

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            idx, answer_ids, answer_mask, context_ids, context_mask, has_answer_labels = batch
            answer_ids, answer_mask = answer_ids.cuda(), answer_mask.bool().cuda()
            has_answer_labels = has_answer_labels.cuda()
            labels = answer_ids.masked_fill(~answer_mask, -100)
            if hasattr(model, "module"):
                model.module.encoder.n_passages = context_ids.size(1)
            else:
                model.encoder.n_passages = context_ids.size(1)
            context_ids = context_ids.cuda().view(context_ids.size(0), -1)
            context_mask = context_mask.cuda().view(context_ids.size(0), -1)
            decoder_input_ids = None

            inputs = {
                'input_ids': context_ids,
                'attention_mask': context_mask,
                'decoder_attention_mask': answer_mask,
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'has_answer_labels': has_answer_labels,
            }
            outputs = model(**inputs)
            all_has_answer_outputs = outputs[-1]  # Tuple[Tensor], shape: [bsz, n_passages]

            count = torch.numel(has_answer_labels)
            for layer_idx, logits in enumerate(all_has_answer_outputs):
                correct = torch.sum(torch.eq((logits.sigmoid() > 0.5).float(), has_answer_labels)).item()
                all_results[layer_idx]["acc"].append((correct, count))

                predictions = (logits.sigmoid() > 0.5).float()
                true_positive = torch.sum(predictions * has_answer_labels).item()
                pred_positive = torch.sum(predictions).item()
                gt_positive = torch.sum(has_answer_labels).item()

                all_results[layer_idx]["prec"].append((true_positive, pred_positive))
                all_results[layer_idx]["recall"].append((true_positive, gt_positive))

    final_results = {}
    for idx, results in enumerate(all_results):
        for metric, values in results.items():
            value_list, count_list = zip(*values)
            final_results[f"layer{idx}/{metric}"] = sum(value_list) / max(sum(count_list), 1)

    all_f1 = []
    for idx in range(num_layers):
        prec = final_results[f"layer{idx}/prec"]
        recall = final_results[f"layer{idx}/recall"]
        f1 = 2 * prec * recall / max(prec + recall, 1e-5)
        final_results[f"layer{idx}/f1"] = f1
        all_f1.append(f1)

    average_f1 = np.mean(all_f1)
    final_results["avg_f1"] = average_f1
    return final_results


if __name__ == "__main__":
    options = Options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    # slurm.init_distributed_mode(opt)
    # slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    logger.info("Distributed training")

    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    dir_path = os.path.join(opt.checkpoint_dir, opt.name)

    model_name = 't5-' + opt.model_size
    model_class = ACFiDT5
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    collator_function = data.Collator(opt, tokenizer)

    train_examples = data.load_data(opt.train_data_path, n_context=opt.n_context)
    train_dataset = data.Dataset(train_examples, opt.n_context, tokenizer, opt.max_passage_length, opt.no_title)
    dev_examples = data.load_data(opt.dev_data_path, global_rank=opt.global_rank,
                                  # use the global rank and world size attibutes to split the dev set on multiple gpus
                                  world_size=opt.world_size,
                                  n_context=opt.n_context)
    if opt.dev_data_size > 0:
        random.seed(opt.seed)
        dev_examples = random.sample(dev_examples, opt.dev_data_size)
        # dev_examples = dev_examples[:opt.dev_data_size]
    dev_dataset = data.Dataset(dev_examples, opt.n_context, tokenizer, opt.max_passage_length, opt.no_title)

    directory_exists = os.path.exists(dir_path)
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    os.makedirs(dir_path, exist_ok=True)
    if not directory_exists and opt.is_master:
        options.print_options(opt)
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    file_handler = logging.FileHandler(filename=os.path.join(dir_path, "run.log"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if opt.is_master else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()

    if opt.is_master:
        tb_logger = SummaryWriter(os.path.join(opt.checkpoint_dir, opt.name))
    # Setup wandb
    if opt.is_master and _has_wandb:
        opt_dict = vars(opt)
        # os.makedirs(os.path.join(dir_path, "wandb"))
        wandb.init(project="ACFiD", name=opt.name, dir=os.path.join(dir_path), config=opt_dict)

    global_step = 0
    best_metric = 0.

    if not directory_exists and opt.model_path == "none":
        model = model_class.from_pretrained(model_name)
        model = model.to(0 if opt.local_rank == -1 else opt.local_rank)
        # optimizer, scheduler = util.set_optim(opt, model)
        optimizer, scheduler = None, None
    elif opt.model_path == "none":  # directory exists, but model_path is none
        model, optimizer, scheduler, opt_checkpoint, global_step, best_metric = util.restore_epoch(
            model_class, dir_path, opt, reset_params=False, name="latest",
        )
        logger.info("Model loaded from %s" % dir_path)
    else:  # model_path is given
        logger.info("Loading %s" % opt.model_path)
        # model, optimizer, scheduler = util.load_model(model_class, opt.model_path, opt)
        config = T5Config.from_pretrained(opt.model_path)

        # Update config with arguments
        if opt.has_answer_pool_type != "none":
            config.has_answer_pool_type = opt.has_answer_pool_type
        if opt.scheduler_type != "none":
            config.scheduler_type = opt.scheduler_type
            config.scheduler_n_context = opt.scheduler_n_context
            config.scheduler_embed_size = opt.scheduler_embed_size

        model = model_class.from_pretrained(opt.model_path, config=config)
        model = model.to(0 if opt.local_rank == -1 else opt.local_rank)
        optimizer, scheduler = None, None
        logger.info("Model loaded from %s" % opt.model_path)
        logger.info("Model config %s", str(config))

    # Set model training configs (a hack around) which are only used during training
    model.encoder.checkpoint = opt.checkpointing_encoder
    model.decoder.checkpoint = opt.checkpointing_decoder
    model.encoder.n_passages = opt.n_context
    model.freeze_fid_params = opt.freeze_fid_params  # config for training has_answer_heads with FiD parameters froze

    # Training the scheduler
    model.encoder.budget = None  # set to None to disable the scheduler
    model.encoder.num_passages_retained = None  # set to None to disable the scheduler
    model.freeze_has_answer_heads = False
    model.use_bce_loss = True
    model.use_rl_loss = False
    # model.step_cost = opt.step_cost
    # model.discount = opt.discount

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if opt.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    logger.info("Start training")
    train_evaluate(model, optimizer, scheduler, global_step,
                   train_dataset, dev_dataset, opt, collator_function, best_metric)
