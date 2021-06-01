import argparse
import os


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        parser.add_argument('--train_data_path', type=str, default='none', help='path of train data')
        parser.add_argument('--dev_data_path', type=str, default='none', help='path of dev data')
        parser.add_argument('--dev_data_size', type=int, default=-1, help='subsample dev data to speedup evaluation')
        parser.add_argument('--test_data_path', type=str, default='none', help='path of test data')
        parser.add_argument('--model_type', type=str, default='t5')
        parser.add_argument('--model_size', type=str, default='base')
        parser.add_argument('--write_results', action='store_true', help='save test results')

        # dataset parameters
        parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Accumulation step.")
        parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
        parser.add_argument('--n_context', type=int, default=1)
        parser.add_argument('--total_step', type=int, default=10000)
        parser.add_argument('--reload_step', type=int, default=-1, help='reload model at step <reload_step>')
        parser.add_argument('--max_passage_length', type=int, default=250,
                            help='maximum number of tokens in the passages (question included)')
        parser.add_argument('--checkpointing_encoder', action='store_true', help='trades memory for compute')
        parser.add_argument('--checkpointing_decoder', action='store_true', help='trades memory for compute')

        # training parameters
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='epsilon for Adam optimizer')
        parser.add_argument('--warmup_step', type=int, default=0, help='number of warmup steps')
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        parser.add_argument('--log_freq', type=int, default=10,
                            help='log model loss every <log_freq> steps during training')
        parser.add_argument('--eval_freq', type=int, default=500,
                            help='evaluate model every <eval_freq> steps during training')
        parser.add_argument('--eval_print_freq', type=int, default=1000,
                            help='print intermediate results of evaluation every <eval_print_freq> steps')
        parser.add_argument('--save_freq', type=int, default=1000,
                            help='save model every <save_freq> steps during training')

        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument("--master_port", type=int, default=-1,
                            help="Master port (for multi-node SLURM jobs)")
        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        parser.add_argument('--global_rank', type=int, default=-1)
        parser.add_argument('--world_size', type=int, default=-1)
        parser.add_argument('--is_master', action='store_true')
        parser.add_argument('--fp16', action='store_true')
        parser.add_argument('--fp16_opt_level', type=str, default="O1",
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")

        # AC (Adaptive Computation) has_answer_heads parameters
        parser.add_argument('--has_answer_pool_type', type=str, default="none",
                            help='pooling type of has_answer_heads')

        # AC (Adaptive Computation) Scheduler model parameters
        parser.add_argument('--scheduler_type', type=str, default="none",
                            help='type of the AC scheduler (default: none)')
        parser.add_argument('--scheduler_n_context', type=int, default=1,
                            help='maximum number of context for the AC scheduler')
        parser.add_argument('--scheduler_embed_size', type=int, default=10,
                            help='embedding size of the AC MLPScheduler')
        parser.add_argument('--scheduler_hidden_size', type=int, default=10,
                            help='hidden size of the AC MLPScheduler')

        # AC (Adaptive Computation) train/inference parameters
        parser.add_argument('--freeze_fid_params', action='store_true', help='freeze the FiD parameters')
        parser.add_argument('--freeze_has_answer_heads', action='store_true',
                            help='freeze the has_answer_heads parameters (used when training the AC scheduler)')
        parser.add_argument('--use_bce_loss', action='store_true',
                            help='train the has_answer_heads with Binary Cross-entropy loss')
        parser.add_argument('--use_rl_loss', action='store_true',
                            help='train the scheduler with REINFORCE loss')
        parser.add_argument('--budget', type=int, default=None, help='budget number of passage layer')
        parser.add_argument('--num_passages_retained', type=int, default=None,
                            help='number of passages retained after AC')
        parser.add_argument('--step_cost', type=float, default=0.,
                            help='cost per step when training the scheduler with REINFORCE')
        parser.add_argument('--discount', type=float, default=1.,
                            help='discount factor when training the scheduler with REINFORCE')

        return parser

    def print_options(self, opt):
        message = ''
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>40}: {:<40}{}\n'.format(str(k), str(v), comment)

        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        model_dir = os.path.join(expr_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(expr_dir, 'models'))
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        opt = self.parser.parse_args()
        return opt
