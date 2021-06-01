import os
import sys
import torch
import transformers
import logging
import util
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import evaluation

from options import Options
from fidt5_ac import ACFiDT5, T5Config
import data_ac as data

logger = logging.getLogger(__name__)


def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module

    total = 0
    ems = []
    all_layer_cost = []

    fw = None
    if opt.write_results:
        write_path = os.path.join(opt.checkpoint_dir, opt.name, 'test_results')
        fw = open(os.path.join(write_path, '%d.txt' % opt.global_rank), 'w')

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            idx, answer_ids, answer_mask, context_ids, context_mask, has_answer_labels = batch
            model.encoder.n_passages = context_ids.size(1)
            context_ids = context_ids.cuda().view(context_ids.size(0), -1)
            context_mask = context_mask.cuda().view(context_ids.size(0), -1)

            outputs, layer_cost = model.generate(
                input_ids=context_ids,
                attention_mask=context_mask,
                max_length=50,
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.get_example(idx[k])
                question = example.question
                gold = example.answers
                id = example.id
                ems_score = evaluation.ems(ans, gold)
                ems.append(ems_score)

                if fw is not None:
                    fw.write(f"{id}\t{question}\t{ans}\n")

                total += 1

            for c in layer_cost:
                all_layer_cost.append(c.item())

            if (i + 1) % opt.eval_print_freq == 0:
                logger.warning(f"{opt.global_rank}, {i + 1} / {len(dataloader)} -- average = {np.mean(ems):.3f}")

    logger.warning(f"{opt.global_rank}, total {total} -- average = {np.mean(ems):.3f}")
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    score, total = util.weighted_average(np.mean(ems), total, opt)
    logger.info('total number of example %d' % total)
    logger.info(f"average EM = {score:.5f}")
    avg_layer_cost = np.mean(all_layer_cost)
    logger.info(f"average layer cost = {avg_layer_cost:.3f}")

    # write result
    with open(os.path.join(opt.checkpoint_dir, "all_results"), "a") as f:
        f.write(f"budget = {opt.budget}, num_passages_retained = {opt.num_passages_retained}, "
                f"layer cost = {avg_layer_cost}, EM = {score}\n")

    return score


if __name__ == "__main__":
    options = Options()
    opt = options.parse()
    opt.train_batch_size = opt.per_gpu_batch_size
    logger.info("Distributed training")
    opt.is_master = True

    model_name = 't5-' + opt.model_size
    model_class = ACFiDT5
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, return_dict=False)

    collator_function = data.Collator(opt, tokenizer)
    test_examples = data.load_data(opt.test_data_path, n_context=opt.n_context)
    test_dataset = data.Dataset(test_examples, opt.n_context, tokenizer, opt.max_passage_length, opt.no_title)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=opt.per_gpu_batch_size,
                                 shuffle=False, num_workers=4, collate_fn=collator_function)

    dir_path = os.path.join(opt.checkpoint_dir, opt.name)
    directory_exists = os.path.exists(dir_path)
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    os.makedirs(dir_path, exist_ok=True)
    if opt.write_results:
        os.makedirs(os.path.join(dir_path, 'test_results'), exist_ok=True)
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

    # Load model
    logger.info("Loading %s" % opt.model_path)
    config = T5Config.from_pretrained(opt.model_path)

    # # Update AC-FiD config with arguments
    # if opt.has_answer_pool_type != "none":
    #     config.has_answer_pool_type = opt.has_answer_pool_type
    # if opt.scheduler_type != "none":
    #     config.scheduler_type = opt.scheduler_type
    #     config.scheduler_n_context = opt.scheduler_n_context
    #     config.scheduler_embed_size = opt.scheduler_embed_size
    #     config.scheduler_hidden_size = opt.scheduler_hidden_size

    model = model_class.from_pretrained(opt.model_path, config=config)
    model = model.cuda()
    logger.info("Model loaded from %s" % opt.model_path)
    logger.info("Model config %s", str(config))

    # Set model training configs (a hack around) which are only used during training
    model.encoder.checkpoint = False
    model.decoder.checkpoint = False
    model.encoder.n_passages = opt.n_context
    model.freeze_fid_params = False  # config for training has_answer_heads with FiD parameters froze

    if opt.n_context > config.scheduler_n_context:
        raise ValueError(f"n_context can not exceed scheduler_n_context={config.scheduler_n_context}")

    # Set the parameters of the AC scheduler
    model.encoder.budget = opt.budget  # config for training/evaluating AC scheduler
    model.encoder.num_passages_retained = opt.num_passages_retained  # config for training/evaluating AC scheduler
    model.freeze_has_answer_heads = False
    model.step_cost = 0.
    model.discount = 1.
    logger.warning(f"budget = {opt.budget}, num_passages_retained = {opt.num_passages_retained}")

    logger.info("Start eval")
    ems = evaluate(model, test_dataset, test_dataloader, tokenizer, opt)

    if opt.write_results and opt.is_master:
        print(opt.is_master)
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.json'
        util.write_output(glob_path, write_path)

    logger.info("EM %.6f" % (ems))
