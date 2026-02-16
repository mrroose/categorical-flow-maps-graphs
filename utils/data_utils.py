import json
import os
from typing import Literal

import torch
from torch_geometric.data import download_url
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.loader import DataLoader as GraphDataLoader

from utils.general_utils import logger
from utils.molecule_utils import process_graph_qm9, process_graph_zinc, process_graphs


def _maybe_slice(graphs, limit: int | None):
    if limit is None:
        return graphs
    return graphs[: min(limit, len(graphs))]


def _limit_from_dataset_size(dataset_size: str) -> int | None:
    """
    Returns the maximum number of examples to use per split.
    None means "use full split".
    """
    if dataset_size == "one":
        return 1
    if dataset_size == "tiny":
        return 100
    if dataset_size == "small":
        return 1000
    if dataset_size == "full":
        return None
    raise ValueError(f"Invalid dataset_size: {dataset_size}")


def get_loaders(task, dataset_size, batch_size):
    """
    Always caches/loads the FULL processed dataset.
    dataset_size only controls how much of the loaded dataset is used in the loaders.
    """
    os.makedirs("data", exist_ok=True)

    if task == "qm9_wo_H":
        max_nodes = 9
        edge_feats = 4
        node_feats = 4
    elif task == "zinc":
        max_nodes = 38
        edge_feats = 4
        node_feats = 9
    else:
        raise ValueError(f"Invalid task: {task}")

    filename_dense = lambda split: f"data/{task}_{split}_graphs_full.pt"
    filename_smiles = lambda split: f"data/{task}_{split}_smiles_full.pt"

    have_cache = all(
        os.path.exists(filename_dense(split)) for split in ("train", "val", "test")
    ) and all(
        os.path.exists(filename_smiles(split)) for split in ("train", "val", "test")
    )

    if have_cache:
        train_graphs = torch.load(filename_dense("train"), weights_only=False)
        val_graphs = torch.load(filename_dense("val"), weights_only=False)
        test_graphs = torch.load(filename_dense("test"), weights_only=False)

        train_smiles = torch.load(filename_smiles("train"), weights_only=False)
        val_smiles = torch.load(filename_smiles("val"), weights_only=False)
        test_smiles = torch.load(filename_smiles("test"), weights_only=False)

    else:
        # Generate FULL processed datasets and cache them once.
        train_dataset, val_dataset, test_dataset = generate_datasets(
            task=task, n_max_nodes=max_nodes, dataset_size="full"
        )

        # Note: process_graphs returns (graphs, smiles); we only cache graphs here.
        train_graphs, train_smiles = process_graphs(train_dataset, max_nodes)
        val_graphs, val_smiles = process_graphs(val_dataset, max_nodes)
        test_graphs, test_smiles = process_graphs(test_dataset, max_nodes)

        torch.save(train_graphs, filename_dense("train"))
        torch.save(val_graphs, filename_dense("val"))
        torch.save(test_graphs, filename_dense("test"))

        torch.save(train_smiles, filename_smiles("train"))
        torch.save(val_smiles, filename_smiles("val"))
        torch.save(test_smiles, filename_smiles("test"))

    # Slice AFTER loading full cache
    limit = _limit_from_dataset_size(dataset_size)
    train_graphs = _maybe_slice(train_graphs, limit)
    val_graphs = _maybe_slice(val_graphs, limit)
    test_graphs = _maybe_slice(test_graphs, limit)

    train_smiles = _maybe_slice(train_smiles, limit)
    val_smiles = _maybe_slice(val_smiles, limit)
    test_smiles = _maybe_slice(test_smiles, limit)

    train_loader = GraphDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        node_feats,
        edge_feats,
        max_nodes,
        train_smiles,
        val_smiles,
        test_smiles,
    )


def generate_datasets(
    task, n_max_nodes, dataset_size: Literal["one", "tiny", "small", "full"] = "full"
):
    """
    Generates the FULL underlying datasets (train/val/test).
    dataset_size is kept for backwards-compat but ignored (always full).
    """
    if dataset_size != "full":
        logger.info(
            "generate_datasets(dataset_size=%s) ignored; generating FULL dataset. "
            "Use get_loaders(..., dataset_size=...) to limit via slicing.",
            dataset_size,
        )

    if task[:3] == "qm9":
        dataset = QM9(root="data/QM9")

        dataset = dataset.shuffle()
        dataset.shuffle()

        train_dataset = dataset[:100000]
        val_dataset = dataset[100000:120000]
        test_dataset = dataset[120000:]

        # FULL: no truncation here; truncation is done in get_loaders
        train_dataset = [
            process_graph_qm9(mol, max_nodes=n_max_nodes) for mol in train_dataset
        ]
        val_dataset = [
            process_graph_qm9(mol, max_nodes=n_max_nodes) for mol in val_dataset
        ]
        test_dataset = [
            process_graph_qm9(mol, max_nodes=n_max_nodes) for mol in test_dataset
        ]

    elif task == "zinc":
        train_dataset = ZINC(root="data/ZINC", subset=False, split="train")
        val_dataset = ZINC(root="data/ZINC", subset=False, split="val")
        test_dataset = ZINC(root="data/ZINC", subset=False, split="test")
        full_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset, test_dataset]
        )
        val_index_url = "https://raw.githubusercontent.com/harryjo97/GruM/master/GruM_2D/data/valid_idx_zinc250k.json"
        val_index_path = "data/ZINC/valid_idx_zinc250k.json"
        if not os.path.exists(val_index_path):
            download_url(val_index_url, "data/ZINC")

        with open(val_index_path, "r") as f:
            valid_indices = json.load(f)

        valid_set = set(valid_indices)
        train_indices = [i for i in range(len(full_dataset)) if i not in valid_set]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, valid_indices)

        processed_graphs = {}
        datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        for k, v in datasets.items():
            gs = []
            for graph in v:
                processed_graph = process_graph_zinc(graph, n_max_nodes)
                gs.append(processed_graph)
            processed_graphs[k] = gs

        train_dataset = processed_graphs["train"]
        val_dataset = processed_graphs["val"]
        test_dataset = processed_graphs["test"]
    else:
        raise ValueError

    return train_dataset, val_dataset, test_dataset
