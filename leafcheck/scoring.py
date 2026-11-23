"""Deterministic prediction policies shared by validation, testing and the app."""

import math

import torch


class PredictionScores(torch.nn.Module):
    """Convert a CNN's raw outputs to the selected classification scores.

    The output can be used by cross-entropy and softmax. For the D4 policy it
    contains log probabilities from one network scored across eight views.
    """

    def __init__(self, network, policy=None):
        super().__init__()
        self.network = network
        self.policy = policy or {"name": "single_view"}
        name = self.policy["name"]
        bias = None
        if name == "train_prior_quarter":
            counts = self.policy["training_counts"]
            if len(counts) != 177 or any(type(n) is not int or n < 1 for n in counts) or sum(counts) != 59008:
                raise ValueError("The prior policy requires the frozen training counts in class order")
            bias = -.25 * (torch.tensor(counts, dtype=torch.float32) / sum(counts)).log()
        elif name not in {"single_view", "d4_probability_mean"}:
            raise ValueError(f"Unknown inference policy: {name}")
        self.register_buffer("class_bias", bias)

    def forward(self, x):
        if self.policy["name"] == "d4_probability_mean":
            views = (
                self.network(torch.rot90(view, turns, (-2, -1)).contiguous(memory_format=torch.channels_last)).log_softmax(1)
                for view in (x, x.flip(-1))
                for turns in range(4)
            )
            aggregate = next(views)
            for scores in views:
                aggregate = torch.logaddexp(aggregate, scores)
            return aggregate - math.log(8)
        scores = self.network(x)
        return scores + self.class_bias if self.class_bias is not None else scores
