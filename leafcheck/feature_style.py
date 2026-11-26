"""Train-only mixing of early feature statistics with an independent RNG."""

import torch


def mix_feature_statistics(features, permutation, keep):
    variance, mean = torch.var_mean(features.detach(), dim=(2, 3), correction=0, keepdim=True)
    deviation = (variance + 1e-6).sqrt()
    normalized = (features - mean) / deviation
    return normalized * deviation.lerp(deviation[permutation], 1 - keep) + mean.lerp(mean[permutation], 1 - keep)


def forward_style(model, inputs, rng, probability, alpha):
    if not model.training or len(inputs) < 2:
        return model(inputs)
    x = model.features[0](inputs)
    for index, layer in enumerate(model.features[1:], start=1):
        x = layer(x)
        if index in (1, 2) and rng.random() < probability:
            permutation = torch.as_tensor(rng.permutation(len(x)), device=x.device)
            keep = torch.as_tensor(rng.beta(alpha, alpha, len(x)), dtype=x.dtype, device=x.device)
            x = mix_feature_statistics(x, permutation, keep[:, None, None, None])
    return model.classifier(x)
