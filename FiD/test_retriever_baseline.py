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

from evaluation import SimpleTokenizer, has_answer

simple_tokenizer = SimpleTokenizer()


def evaluate(dataset, dataloader, opt):
    all_accuracies = []

    for i, batch in enumerate(dataloader):
        idx, answer_ids, answer_mask, context_ids, context_mask, has_answer_labels = batch
        # model.encoder.n_passages = context_ids.size(1)
        # answer_ids, answer_mask = answer_ids.cuda(), answer_mask.bool().cuda()
        # context_ids = context_ids.cuda().view(context_ids.size(0), -1)
        # context_mask = context_mask.cuda().view(context_ids.size(0), -1)
        # decoder_input_ids = None
        # has_answer_labels = None
        # # labels = answer_ids.masked_fill(~answer_mask, -100)
        # labels = None
        #
        # inputs = {
        #     'input_ids': context_ids,
        #     'attention_mask': context_mask,
        #     'decoder_attention_mask': answer_mask,
        #     'decoder_input_ids': decoder_input_ids,
        #     'labels': labels,
        #     'has_answer_labels': has_answer_labels,
        # }
        # outputs = model(**inputs)
        # scheduler_outputs = outputs[-2]
        # actions, log_probs, all_skylines, retained_passages = scheduler_outputs

        # retained_passages: [bsz, num_passages_retained]
        for j, index in enumerate(idx):
            answer_acc = 0  # 1 if the selected top-k passages contain the answer, 0 otherwise
            example = dataset.get_example(index)
            answers = example.answers
            for k in range(opt.num_passages_retained):
                context = example.contexts[k]
                if has_answer(answers, context, simple_tokenizer):
                    answer_acc = 1
                    break
            all_accuracies.append(answer_acc)

    accuracy = np.mean(all_accuracies)

    logger.info('total number of example %d' % len(all_accuracies))
    logger.info(f"top-k retrieval accuracy = {accuracy:.5f}")

    # # write result
    # with open(os.path.join(opt.checkpoint_dir, "retrieval_acc"), "a") as f:
    #     f.write(f"budget = {opt.budget}, num_passages_retained = {opt.num_passages_retained}, "
    #             f"accuracy = {accuracy}\n")

    return accuracy


if __name__ == "__main__":
    options = Options()
    opt = options.parse()
    opt.train_batch_size = opt.per_gpu_batch_size
    logger.info("Distributed training")
    opt.is_master = True

    model_name = 't5-' + opt.model_size
    # model_class = ACFiDT5
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, return_dict=False)

    collator_function = data.Collator(opt, tokenizer)
    test_examples = data.load_data(opt.test_data_path, n_context=opt.n_context)
    test_dataset = data.Dataset(test_examples, opt.n_context, tokenizer, opt.max_passage_length, opt.no_title)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=opt.per_gpu_batch_size,
                                 shuffle=False, num_workers=4, collate_fn=collator_function)

    # dir_path = os.path.join(opt.checkpoint_dir, opt.name)
    # directory_exists = os.path.exists(dir_path)
    # if opt.world_size > 1 and not opt.local_rank == -1:
    #     torch.distributed.barrier()
    # os.makedirs(dir_path, exist_ok=True)
    # if opt.write_results:
    #     os.makedirs(os.path.join(dir_path, 'test_results'), exist_ok=True)
    # if not directory_exists and opt.is_master:
    #     options.print_options(opt)
    # if opt.world_size > 1 and not opt.local_rank == -1:
    #     torch.distributed.barrier()

    # file_handler = logging.FileHandler(filename=os.path.join(dir_path, "run.log"))
    # stdout_handler = logging.StreamHandler(sys.stdout)
    # handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if opt.is_master else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        # handlers=handlers,
    )

    # Load model
    # logger.info("Loading %s" % opt.model_path)
    # config = T5Config.from_pretrained(opt.model_path)

    # # Update AC-FiD config with arguments
    # if opt.has_answer_pool_type != "none":
    #     config.has_answer_pool_type = opt.has_answer_pool_type
    # if opt.scheduler_type != "none":
    #     config.scheduler_type = opt.scheduler_type
    #     config.scheduler_n_context = opt.scheduler_n_context
    #     config.scheduler_embed_size = opt.scheduler_embed_size
    #     config.scheduler_hidden_size = opt.scheduler_hidden_size

    # model = model_class.from_pretrained(opt.model_path, config=config)
    # model = model.cuda()
    # logger.info("Model loaded from %s" % opt.model_path)
    # logger.info("Model config %s", str(config))

    # # Set model training configs (a hack around) which are only used during training
    # model.encoder.checkpoint = False
    # model.decoder.checkpoint = False
    # model.encoder.n_passages = opt.n_context
    # model.freeze_fid_params = True  # config for training has_answer_heads with FiD parameters froze
    #
    # if opt.n_context > config.scheduler_n_context:
    #     raise ValueError(f"n_context can not exceed scheduler_n_context={config.scheduler_n_context}")
    #
    # # Set the parameters of the AC scheduler
    # model.encoder.budget = opt.budget  # config for training/evaluating AC scheduler
    # model.encoder.num_passages_retained = opt.num_passages_retained  # config for training/evaluating AC scheduler
    # model.freeze_has_answer_heads = True
    # model.step_cost = 0.
    # model.discount = 1.
    logger.warning(f"budget = {opt.budget}, num_passages_retained = {opt.num_passages_retained}")

    logger.info("Start eval")
    accuracy = evaluate(test_dataset, test_dataloader, opt)

    logger.info("accuracy %.6f" % (accuracy))
