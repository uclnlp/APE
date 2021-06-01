from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions.utils import probs_to_logits
from torch.distributions import Categorical

Tower = namedtuple("Tower", "has_answer_logit, layer, rank")

LARGE_POS = 1000
LARGE_NEG = -1e5


class BaseScheduler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_priorities = nn.Parameter(self._get_init_priorities())

    def _get_init_priorities(self) -> torch.Tensor:
        """Initialize the initial priorities for all passages."""
        # # Uniform
        # self.init_priorities = nn.Parameter(torch.Tensor(self.config.scheduler_n_context))
        # bound = 1 / math.sqrt(self.config.scheduler_n_context)
        # self.init_priorities.data.uniform_(-bound, bound)
        #
        # # Zeros
        # self.init_priorities = nn.Parameter(torch.Tensor(self.config.scheduler_n_context))
        # self.init_priorities.data.fill_(0.)

        # Heuristic
        init_probs = torch.tensor([0.5 / (i + 1) for i in range(self.config.scheduler_n_context)])
        return probs_to_logits(init_probs, is_binary=True)  # shape: [n_passages]

    def forward(self, has_answer_logits, layers, ranks):
        """
        Args:
            has_answer_logits (Tensor): float Tensor with shape [bsz, n_passages]
            layers (Tensor): int Tensor with shape [bsz, n_passages]
            ranks (Tensor): int Tensor with shape [bsz, n_passages]

        Returns:
            priorities (Tensor): float Tensor with shape [bsz, n_passages]
        """
        raise NotImplementedError

    def act(self, all_has_answer_logits, layer_indices, masks, greedy=False, **kwargs):
        """ Take an action given the current status of the skyline.

        Args:
            all_has_answer_logits (torch.Tensor): float Tensor with shape [bsz, n_passages, num_layers]
            layer_indices (torch.Tensor): int Tensor with shape [bsz, n_passages]
            masks (torch.Tensor): float Tensor with shape [bsz, n_passages]
            greedy (bool): True if act greedily
            **kwargs:

        Returns:
            action (torch.Tensor): int Tensor with shape [bsz] that indicates the actions chosen
            log_probs (torch.Tensor): float Tensor with shape [bsz] that indicates the log-prob of the actions
        """
        bsz, n_passages, num_layers = all_has_answer_logits.shape
        device = all_has_answer_logits.device

        init_priors = self.init_priorities[:n_passages].unsqueeze(0).expand(bsz, -1)  # [bsz, n_passages]
        all_has_answer_logits_with_init = torch.cat((init_priors.unsqueeze(-1), all_has_answer_logits), -1)
        # shape: [bsz, n_passages, num_layers + 1]

        layer_tensor = layer_indices + 1  # shape: [bsz, n_passages], range=[0, num_layers]
        rank_tensor = torch.arange(n_passages, device=device).unsqueeze(0).expand(bsz, -1)  # shape: [bsz, n_passages]

        # Collect the has_answer logits for each tower (including the initial layers), shape: [bsz, n_passages]
        has_answer_logits = all_has_answer_logits_with_init.gather(2, layer_tensor.unsqueeze(2)).squeeze(2)
        # has_answer_logits = torch.where(layer_indices < 0, init_priors, has_answer_logits)

        priorities = self.forward(has_answer_logits, layer_tensor, rank_tensor)  # shape: [bsz, n_passages]

        # Apply the mask to avoid choosing the maximum towers again
        priorities = priorities + (1. - masks) * LARGE_NEG

        if greedy:  # select the max priority during evaluation
            action = priorities.argmax(-1)  # shape: [bsz]
            log_prob = -torch.ones(bsz, device=device, requires_grad=False)
        else:
            m = Categorical(logits=priorities)
            action = m.sample()  # shape: [bsz]
            log_prob = m.log_prob(action)  # shape: [bsz]

        return action, log_prob


class TopScheduler(BaseScheduler):
    def forward(self, has_answer_logits, layers, ranks):
        priorities = has_answer_logits * 0. - ranks.float()  # shape: [bsz, n_passages]
        return priorities


class DummyScheduler(BaseScheduler):
    """Simple scheduler that is solely based on has_answer_prob."""

    def __init__(self, config):
        super().__init__(config)
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, has_answer_logits, layers, ranks):
        priorities = self.weight * has_answer_logits  # shape: [bsz, n_passages]
        return priorities


class SimpleScheduler(BaseScheduler):
    """Scheduler that exploits has_answer_prob, rank and layer as input."""

    def __init__(self, config):
        super().__init__(config)
        self.weight = nn.Parameter(torch.tensor(1.0))

        self.layer_embeddings = nn.Embedding(config.num_layers + 1, 1)
        self.rank_embeddings = nn.Embedding(config.scheduler_n_context, 1)

    def forward(self, has_answer_logits, layers, ranks):
        # Compute the offsets
        layer_emb = self.layer_embeddings(layers)  # shape: [bsz, n_passages, 1]
        rank_emb = self.rank_embeddings(ranks)  # shape: [bsz, n_passages, 1]
        offsets = torch.squeeze(layer_emb + rank_emb, -1)  # shape: [bsz, n_passages]

        priorities = self.weight * has_answer_logits + offsets  # shape: [bsz, n_passages]
        return priorities


class MLPScheduler(BaseScheduler):
    """Scheduler that uses MLP to integrate has_answer_prob, rank and layer as input."""

    def __init__(self, config):
        super().__init__(config)
        self.weight = nn.Parameter(torch.tensor(1.0))

        embed_size = config.scheduler_embed_size
        self.layer_embeddings = nn.Embedding(config.num_layers + 1, embed_size)
        self.rank_embeddings = nn.Embedding(config.scheduler_n_context, embed_size)

        # MLP
        hidden_size = config.scheduler_hidden_size
        self.dense0 = nn.Linear(embed_size * 2 + 1, hidden_size)
        self.act_fn = F.relu
        self.dense1 = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, has_answer_logits, layers, ranks):
        """
        Args:
            has_answer_logits (Tensor): float Tensor with shape [bsz, n_passages]
            layers (Tensor): int Tensor with shape [bsz, n_passages]
            ranks (Tensor): int Tensor with shape [bsz, n_passages]

        Returns:
            priorities (Tensor): float Tensor with shape [bsz, n_passages]
        """

        layer_emb = self.layer_embeddings(layers)  # shape: [bsz, n_passages, embed_size]
        rank_emb = self.rank_embeddings(ranks)  # shape: [bsz, n_passages, embed_size]

        mlp_input = torch.cat(
            (has_answer_logits.unsqueeze(-1), layer_emb, rank_emb), -1
        )  # shape: [bsz, n_passages, embed_size * 2 + 1]
        mlp_output = self.dense1(self.act_fn(self.dense0(mlp_input)))  # shape: [bsz, n_passages, 1]

        offset_logit = torch.squeeze(mlp_output, -1)  # shape: [bsz, n_passages]
        priorities = self.weight * has_answer_logits + offset_logit  # shape: [bsz, n_passages]

        return priorities  # shape: [bsz, n_passages]


class GatedMLPScheduler(BaseScheduler):
    """Scheduler that uses MLP to integrate has_answer_prob (gated), rank and layer."""

    def __init__(self, config):
        super().__init__(config)
        self.weight = nn.Parameter(torch.tensor(1.0))

        embed_size = config.scheduler_embed_size
        self.layer_embeddings = nn.Embedding(config.num_layers + 1, embed_size)
        self.rank_embeddings = nn.Embedding(config.scheduler_n_context, embed_size)

        # MLP
        hidden_size = config.scheduler_hidden_size
        self.dense0 = nn.Linear(embed_size * 2 + 1, hidden_size)
        self.act_fn = F.relu
        self.dense1 = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, has_answer_logits, layers, ranks):
        """
        Args:
            has_answer_logits (Tensor): float Tensor with shape [bsz, n_passages]
            layers (Tensor): int Tensor with shape [bsz, n_passages]
            ranks (Tensor): int Tensor with shape [bsz, n_passages]

        Returns:
            priorities (Tensor): float Tensor with shape [bsz, n_passages]
        """

        layer_emb = self.layer_embeddings(layers)  # shape: [bsz, n_passages, embed_size]
        rank_emb = self.rank_embeddings(ranks)  # shape: [bsz, n_passages, embed_size]

        mlp_input = torch.cat(
            (has_answer_logits.unsqueeze(-1), layer_emb, rank_emb), -1
        )  # shape: [bsz, n_passages, embed_size * 2 + 1]
        mlp_output = self.dense1(self.act_fn(self.dense0(mlp_input)))  # shape: [bsz, n_passages, 2]

        offset_logit, gate_logit = torch.unbind(mlp_output, dim=-1)
        gate = torch.sigmoid(gate_logit)  # shape: [bsz, n_passages]
        priorities = self.weight * gate * has_answer_logits + offset_logit  # shape: [bsz, n_passages]

        return priorities  # shape: [bsz, n_passages]


class GatedMLPSchedulerWithPosition(GatedMLPScheduler):
    """Scheduler that uses MLP to integrate has_answer_prob (gated), rank and layer."""

    def __init__(self, config):
        super().__init__(config)
        self.weight = nn.Parameter(torch.tensor(1.0))

        embed_size = config.scheduler_embed_size
        self.layer_embeddings = nn.Embedding(config.num_layers + 1, embed_size)
        self.rank_embeddings = nn.Embedding(config.scheduler_n_context, embed_size)

        # MLP
        hidden_size = config.scheduler_hidden_size
        self.dense0 = nn.Linear(embed_size * 2 + 4, hidden_size)
        self.act_fn = F.relu
        self.dense1 = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, has_answer_logits, layers, ranks):
        """
        Args:
            has_answer_logits (Tensor): float Tensor with shape [bsz, n_passages]
            layers (Tensor): int Tensor with shape [bsz, n_passages]
            ranks (Tensor): int Tensor with shape [bsz, n_passages]

        Returns:
            priorities (Tensor): float Tensor with shape [bsz, n_passages]
        """

        layer_emb = self.layer_embeddings(layers)  # shape: [bsz, n_passages, embed_size]
        rank_emb = self.rank_embeddings(ranks)  # shape: [bsz, n_passages, embed_size]

        # Additional features
        layers_feat = (layers.float() / self.config.num_layers).unsqueeze(-1)  # shape: [bsz, n_passages, 1]
        ranks_feat = (ranks.float() / self.config.scheduler_n_context).unsqueeze(-1)  # shape: [bsz, n_passages, 1]

        mlp_input = torch.cat(
            (has_answer_logits.unsqueeze(-1), layers_feat, ranks_feat, layers_feat + ranks_feat,
             layer_emb, rank_emb), -1
        )  # shape: [bsz, n_passages, embed_size * 2 + 1]
        mlp_output = self.dense1(self.act_fn(self.dense0(mlp_input)))  # shape: [bsz, n_passages, 2]

        offset_logit, gate_logit = torch.unbind(mlp_output, dim=-1)
        gate = torch.sigmoid(gate_logit)  # shape: [bsz, n_passages]
        priorities = self.weight * gate * has_answer_logits + offset_logit  # shape: [bsz, n_passages]

        return priorities  # shape: [bsz, n_passages]


SchedulerMapping = {
    "top": TopScheduler,
    "dummy": DummyScheduler,
    "simple": SimpleScheduler,
    "mlp": MLPScheduler,
    "gated_mlp": GatedMLPScheduler,
    "gated_mlp_pos": GatedMLPSchedulerWithPosition,
}


def get_scheduler(config):
    """Construct a scheduler from the config (default: None)"""
    if hasattr(config, "scheduler_type"):
        try:
            scheduler = SchedulerMapping[config.scheduler_type](config)
        except KeyError:
            raise KeyError(f"Invalid scheduler_type: {config.scheduler_type}")
    else:
        scheduler = None
    return scheduler


def run_ac_scheduler(
    hidden_states,
    attention_mask,
    has_answer_outputs,
    ac_scheduler: BaseScheduler,
    budget: int,
    num_passages_retained: int,
    is_training: bool = True,
):
    """

    Args:
        hidden_states (torch.Tensor): float Tensor with shape [bsz (B), n_passages (N), plen (L), d_model (D)]
        attention_mask (torch.Tensor): float Tensor with shape [B, N, L]
        has_answer_outputs (torch.Tensor): float Tensor with shape [B, N, num_layers]
        ac_scheduler (BaseScheduler):
        budget (int):
        num_passages_retained (int):
        is_training (bool):

    Returns:

    """
    bsz, n_passages, plen, _ = hidden_states.shape
    num_layers = has_answer_outputs.shape[2]
    if budget > num_layers * n_passages:
        raise ValueError(f"budget={budget} should be small than num_layers * n_passages={num_layers * n_passages}")
    device = hidden_states.device

    # Run the AC prioritization algorithm
    all_actions, all_log_probs = [], []
    skyline = -torch.ones((bsz, n_passages), dtype=torch.long, device=device)  # -1 indicates the initial state
    tower_masks = torch.ones((bsz, n_passages), dtype=torch.float, device=device)  # 1.->active, 0.->inactive
    for step in range(budget):
        actions, action_log_probs = ac_scheduler.act(
            has_answer_outputs,
            skyline,
            masks=tower_masks,
            greedy=not is_training
        )  # shape: [bsz], [bsz]
        all_actions.append(actions)
        all_log_probs.append(action_log_probs)

        # Update the selected towers in the skyline
        for i, action in enumerate(actions):
            new_layer = skyline[i, action].item() + 1  # increment the layer
            if new_layer < num_layers - 1:
                skyline[i, action] = new_layer
            elif new_layer == num_layers - 1:  # reaches the last layer
                skyline[i, action] = new_layer
                tower_masks[i, action] = 0.  # mask the tower to avoid choosing it again.
            else:
                raise ValueError("Selected the tower that is at maximum height.")

    actions = torch.stack(all_actions, 1)  # shape: [bsz, budget]
    log_probs = torch.stack(all_log_probs, 1)  # shape: [bsz, budget]

    # Find the highest <num_passages_retained> towers (passages), shape: [bsz, num_passages_retained]
    retained_passages = skyline.argsort(dim=1, descending=True)[:, :num_passages_retained]

    # Update the skyline: forward the highest towers to their last layer
    skyline.scatter_(1, retained_passages, num_layers - 1)  # shape: [bsz, n_passages]

    # Acquire the hidden_states and attention_masks for the retained passages
    retained_hidden_states, retained_attention_masks = [], []
    for bi in range(bsz):
        # TODO (jimmycode): find a more efficient implementation for this indexing operation
        cur_retained_hidden_states = torch.cat(
            [hidden_states[bi, retained_passages[bi, pj]] for pj in range(num_passages_retained)], 0
        )  # shape: [num_passages_retained * p_len, d_model]
        retained_hidden_states.append(cur_retained_hidden_states)

        cur_retained_attention_masks = torch.cat(
            [attention_mask[bi, retained_passages[bi, pj]] for pj in range(num_passages_retained)], 0
        )  # shape: [num_passages_retained * p_len]
        retained_attention_masks.append(cur_retained_attention_masks)
    hidden_states = torch.stack(retained_hidden_states, 0)  # shape: [bsz, num_passages_retained * p_len, d_model]
    attention_mask = torch.stack(retained_attention_masks, 0)  # shape: [bsz, num_passages_retained * p_len]

    return hidden_states, attention_mask, (actions, log_probs, skyline, retained_passages)


def compute_REINFORCE_loss(has_answer_labels, actions, log_probs, step_cost, discount=1.0):
    """ Compute the REINFORCE loss: 1) evaluate rewards and returns 2) compute the loss """
    action_labels = torch.gather(has_answer_labels, 1, actions)  # shape: [bsz, budget]
    immediate_rewards = action_labels - step_cost  # shape: [bsz, budget]

    # Calculate return values
    # return_values = immediate_rewards.flip(1).cumsum(1).flip(1)  # shape: [bsz, budget]
    all_imd_rewards = immediate_rewards.flip(1).unbind(1)
    all_returns, acc = [], None
    for ir in all_imd_rewards:
        cur_return = ir if acc is None else ir + acc * discount  # shape: [bsz]
        all_returns.append(cur_return)
        acc = cur_return
    return_values = torch.stack(all_returns, 1).flip(1)  # shape: [bsz, budget]

    loss = -torch.sum(log_probs * return_values)  # add negative to maximize
    sum_reward = torch.mean(torch.sum(immediate_rewards, 1))  # the sum of all immediate rewards (for logging)

    return loss, sum_reward
