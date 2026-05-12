import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def init_class_centers(cfg: dict) -> np.ndarray:
    """
    Fixes the Gaussian blob centers once, seeded for reproducibility.
    Returns centers: ndarray of shape [n_classes, n_dims].
    """
    d = cfg["data"]
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    centers = rng.uniform(-10, 10, size=(d["n_classes"], d["n_dims"]))
    logger.info(
        f"Initialised {d['n_classes']} class centers in {d['n_dims']}D "
        f"(train classes: 0..{d['n_train_classes']-1}, "
        f"test also sees: {d['n_train_classes']}..{d['n_classes']-1})"
    )
    return centers


def _draw_samples_for_class(center: np.ndarray, n: int, noise: float) -> torch.Tensor:
    """Draws n fresh samples around `center` with Gaussian noise std=noise."""
    samples = center + np.random.randn(n, len(center)) * noise
    return torch.tensor(samples, dtype=torch.float32)


def sample_episode(centers: np.ndarray, cfg: dict, device: str):
    """
    Samples one prototypical training episode using KNOWN classes only
    (class indices 0 .. n_train_classes-1).
    Returns:
        support_x: [n_train_classes * k_shot, n_dims]
        support_y: [n_train_classes * k_shot]   integer labels (0-indexed within known set)
        query_x:   [n_train_classes * n_query, n_dims]
        query_y:   [n_train_classes * n_query]
    """
    d = cfg["data"]
    pc = cfg["prototypical"]
    k = pc["k_shot"]
    q = pc["n_query"]
    noise = d["noise"]
    n_train = d["n_train_classes"]

    support_xs, support_ys, query_xs, query_ys = [], [], [], []
    for label_idx in range(n_train):
        center = centers[label_idx]
        samples = _draw_samples_for_class(center, k + q, noise)
        support_xs.append(samples[:k])
        query_xs.append(samples[k:])
        support_ys.append(torch.full((k,), label_idx, dtype=torch.long))
        query_ys.append(torch.full((q,), label_idx, dtype=torch.long))

    dev = torch.device(device)
    return (
        torch.cat(support_xs).to(dev),
        torch.cat(support_ys).to(dev),
        torch.cat(query_xs).to(dev),
        torch.cat(query_ys).to(dev),
    )


def sample_test_batch(centers: np.ndarray, cfg: dict, device: str):
    """
    Draws a fresh test batch covering ALL classes (known + unknown).
    Also draws a small support set from known classes to build prototypes.
    Returns:
        support_x, support_y:   from known classes only (for prototype computation)
        test_x:                 [n_classes * n_test_per_class, n_dims]
        test_y:                 [n_classes * n_test_per_class]  global integer labels
    """
    d = cfg["data"]
    pc = cfg["prototypical"]
    noise = d["noise"]
    n_test = d["n_test_samples_per_class"]
    n_train = d["n_train_classes"]
    n_classes = d["n_classes"]
    k = pc["k_shot"]

    # Support from known classes
    sup_xs, sup_ys = [], []
    for label_idx in range(n_train):
        sup_xs.append(_draw_samples_for_class(centers[label_idx], k, noise))
        sup_ys.append(torch.full((k,), label_idx, dtype=torch.long))

    # Test from ALL classes
    test_xs, test_ys = [], []
    for label_idx in range(n_classes):
        test_xs.append(_draw_samples_for_class(centers[label_idx], n_test, noise))
        test_ys.append(torch.full((n_test,), label_idx, dtype=torch.long))

    dev = torch.device(device)
    return (
        torch.cat(sup_xs).to(dev),
        torch.cat(sup_ys).to(dev),
        torch.cat(test_xs).to(dev),
        torch.cat(test_ys).to(dev),
    )


def get_class_names(cfg: dict) -> list:
    return [f"class_{i}" for i in range(cfg["data"]["n_classes"])]
