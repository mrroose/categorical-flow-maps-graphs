import logging
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

from utils.graph_utils import to_dense

logger = logging.getLogger(__name__)

_ATOM_SYMBOLS: List[str] = ["C", "N", "O", "F", "Br", "Cl", "I", "P", "S"]
_BOND_TYPE_BY_LABEL = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


def make_molecule(x: Tensor, e: Tensor, size: int) -> Chem.RWMol:
    """
    Build an RDKit RWMol from dense node and edge tensors.

    Expected encodings:
      - x: [N, num_atom_types] logits/one-hot where argmax gives atom type index
      - e: either
          * [N, N, num_bond_types] logits/one-hot where argmax gives bond label, or
          * [N, N] integer bond labels directly
        Bond labels: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic

    Notes:
      - Adds bonds only for i<j
      - Marks aromatic atoms/bonds as aromatic (helps RDKit sanitization/kekulization)
    """
    if size <= 0:
        return Chem.RWMol()

    # Be defensive about tensor sizes.
    n_x = int(x.size(0))
    n_e0 = int(e.size(0)) if e.dim() >= 2 else 0
    n_e1 = int(e.size(1)) if e.dim() >= 2 else 0
    size = int(min(size, n_x, n_e0, n_e1))

    mol = Chem.RWMol()

    # --- atoms ---
    atom_type_idx = x[:size].argmax(dim=-1).to(torch.long).cpu().tolist()
    for idx in atom_type_idx:
        # clamp unknown indices to carbon to avoid KeyErrors
        if idx < 0 or idx >= len(_ATOM_SYMBOLS):
            raise ValueError("Unknown atom type found.")
        mol.AddAtom(Chem.Atom(_ATOM_SYMBOLS[idx]))

    if size <= 1:
        return mol

    # --- bonds (upper triangular only) ---
    if e.dim() == 3:
        bond_labels = e[:size, :size].argmax(dim=-1).to(torch.long)
    elif e.dim() == 2:
        bond_labels = e[:size, :size].to(torch.long)
    else:
        # Unexpected edge tensor shape; return atom-only mol.
        return mol

    iu = torch.triu_indices(size, size, offset=1, device=bond_labels.device)
    i_idx = iu[0].cpu().tolist()
    j_idx = iu[1].cpu().tolist()
    b_lab = bond_labels[iu[0], iu[1]].cpu().tolist()

    for i, j, lab in zip(i_idx, j_idx, b_lab):
        if lab == 0:
            continue
        bond_type = _BOND_TYPE_BY_LABEL.get(int(lab))
        if bond_type is None:
            continue

        mol.AddBond(int(i), int(j), bond_type)

        # Ensure aromatic flags are consistent for RDKit
        # if int(lab) == 4:
        #     mol.GetAtomWithIdx(int(i)).SetIsAromatic(True)
        #     mol.GetAtomWithIdx(int(j)).SetIsAromatic(True)
        #     b = mol.GetBondBetweenAtoms(int(i), int(j))
        #     if b is not None:
        #         b.SetIsAromatic(True)

    return mol


def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Standard "graph-gen paper" validity gate:
      1) RDKit sanitization must succeed
      2) (optional) molecule must be connected (recommended True)
    No Kekulize() here by design.
    """
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return mol


def mol2smiles(mol: Chem.Mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def _canonical_smiles_from_mol(mol: Chem.Mol) -> Optional[str]:
    """
    Returns canonical SMILES for the largest fragment after sanitization/kekulization.
    Returns None if RDKit sanitization fails.
    """
    try:
        Chem.SanitizeMol(mol)
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
        return Chem.MolToSmiles(largest, canonical=True)
    except Exception:
        return None


def _smiles_from_dense_graph(X: Tensor, E: Tensor, size: int) -> Optional[str]:
    """
    Builds an RDKit molecule from dense node/edge tensors and returns canonical SMILES.
    """
    if size <= 1:
        return None
    mol = make_molecule(X, E, size)
    return _canonical_smiles_from_mol(mol)


def _iter_batch_graphs_and_smiles(
    batch: Data, n_max_nodes: int
) -> Tuple[List[Data], List[str], int]:
    """
    Converts a (possibly batched) PyG Batch/Data into:
      - list of per-sample Data objects
      - list of corresponding canonical SMILES
      - number of sanitization failures
    Filters out samples with <=1 node or invalid SMILES.
    """
    # Ensure we can map dense index `b` back to the correct sample object.
    data_list = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]

    # Quick filter: if batch is somehow malformed, just skip gracefully.
    if not hasattr(batch, "x") or batch.x is None or batch.x.size(0) <= 1:
        return [], [], 0

    graphs, node_mask, _ = to_dense(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, n_max_nodes
    )

    kept_graphs: List[Data] = []
    kept_smiles: List[str] = []
    errors_sanitizing = 0

    B = graphs.X.size(0)
    # data_list length should match B, but be defensive.
    B_eff = min(B, len(data_list))

    for b in range(B_eff):
        size = int(node_mask[b].sum().item())
        if size <= 1:
            continue

        smile = _smiles_from_dense_graph(graphs.X[b], graphs.E[b], size)
        if smile is None:
            errors_sanitizing += 1
            continue

        kept_smiles.append(smile)
        kept_graphs.append(data_list[b])

    return kept_graphs, kept_smiles, errors_sanitizing


def process_graphs(dataset, n_max_nodes: int):
    """
    Processes a dataset of PyG Data objects:
      - filters invalid graphs (<=1 node) and molecules that RDKit can't sanitize
      - returns (list_of_graphs, list_of_smiles) aligned by index
    """
    all_graphs: List[Data] = []
    smiles: List[str] = []

    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)

    errors_sanitizing_total = 0
    for batch in loader:
        gs, ss, errs = _iter_batch_graphs_and_smiles(batch, n_max_nodes)
        all_graphs.extend(gs)
        smiles.extend(ss)
        errors_sanitizing_total += errs

    if errors_sanitizing_total > 0:
        logger.warning(
            f"{errors_sanitizing_total} errors during sanitizing of training data."
        )

    return all_graphs, smiles


def process_graphs_old(dataset, n_max_nodes: int):
    all_graphs, smiles = [], []
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)

    errors_sanitizing = 0
    for data in loader:
        if data.x.size(0) <= 1:  # Max nr. of nodes
            continue

        graphs, node_mask, _ = to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch, n_max_nodes
        )

        # support B>=1 (even though loader uses batch_size=1)
        B = graphs.X.size(0)
        for b in range(B):
            size = int(node_mask[b].sum().item())
            if size <= 1:
                continue

            mol = make_molecule(graphs.X[b], graphs.E[b], size)
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol)

                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())

                smile = Chem.MolToSmiles(largest_mol, canonical=True)
                smiles.append(smile)
                all_graphs.append(data)  # original Data object for this sample
            except Exception:
                errors_sanitizing += 1
                continue

    if errors_sanitizing > 0:
        logger.warning(
            f"{errors_sanitizing} errors during sanitizing of training data."
        )

    return all_graphs, smiles


def get_smiles_old(loader, n_max_nodes):
    test_smiles = []
    # iterate via a DataLoader so we always have a `.batch`
    dl = GraphDataLoader(loader.dataset, batch_size=1, shuffle=False)
    for batch in dl:
        dense, _, _ = to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, n_max_nodes
        )
        mol = make_molecule(dense.X.squeeze(), dense.E.squeeze(), n_max_nodes)
        smiles = Chem.MolToSmiles(mol)
        test_smiles.append(smiles)

    return test_smiles


def get_smiles(loader: GraphDataLoader, n_max_nodes: int) -> List:
    # Edited to match eval_and_log, so that we get novelty=0 on the training set
    smiles_out = []

    # Batch size 1
    dl = GraphDataLoader(loader.dataset, batch_size=1, shuffle=False)

    for batch in dl:
        dense, mask, _ = to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, n_max_nodes
        )

        # mask: [B,N]
        size = int(mask.squeeze(0).sum().item())
        if size <= 1:
            continue

        mol = make_molecule(dense.X.squeeze(0), dense.E.squeeze(0), size)

        try:
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol)

            frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            mol = max(frags, default=mol, key=lambda m: m.GetNumAtoms())

            smiles_out.append(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            continue

    return smiles_out


def process_graph_qm9(mol, max_nodes):
    # get non-hydrogen nodes
    X = mol.x[:, 1:5]
    # remove all rows that where hydrogen
    non_H_nodes = ~X.eq(0).all(dim=1)
    num_non_H_nodes = int(torch.sum(non_H_nodes).item())
    X = X[non_H_nodes]
    # pad x with zeros to max_nodes
    # X = torch.cat((X, torch.zeros(size=(max_nodes - num_non_H_nodes, 4))), dim=0)
    # dict that maps each edge to its edge attribute

    # loop over edges in index
    edge_index = mol.edge_index
    edge_attr = mol.edge_attr

    e = edge_index.T.tolist()
    f = edge_attr.tolist()

    edges, features = [], []
    d = {tuple(ex): ef for ex, ef in zip(e, f)}

    # edge index to list of edges

    # list of edges to set of edges
    for i, j in product(range(num_non_H_nodes), range(num_non_H_nodes)):

        if (i, j) in d:
            edges.append([i, j])

            fea = d[(i, j)]
            fea = np.argmax(fea) + 1

            features.append(fea)
        #
        # else:
        #     features.append(0)

    edge_index = torch.tensor(edges).T
    edge_attr = torch.tensor(features)
    edge_attr = torch.nn.functional.one_hot(edge_attr.long(), num_classes=4).squeeze()

    if X.shape[0] == 1:
        edge_attr = edge_attr.unsqueeze(0)

    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr)


def process_graph_zinc(graph: Data, max_nodes: int) -> Data:
    """
    ZINC (PyG) typically has:
      - graph.x: atom types as integer labels (shape [num_nodes, 1] or [num_nodes])
      - graph.edge_attr: bond types as integer labels (shape [num_edges, 1] or [num_edges])
        with 4 bond types: {1..4} or sometimes {0..3} depending on source/version.

    We convert to:
      - x: one-hot with 9 classes (your remapping)
      - edge_attr: one-hot with 5 classes: 0=no-bond, 1..4=bond types
    """
    # ---- node features (9 classes) ----
    inds = torch.tensor(
        [
            0,
            2,
            1,
            3,
            0,
            8,
            5,
            2,
            1,
            4,
            1,
            1,
            1,
            1,
            8,
            6,
            7,
            2,
            1,
            2,
            8,
            7,
            7,
            0,
            7,
            8,
            0,
            7,
        ],
        device=graph.x.device,
    )

    atom = graph.x
    atom = atom.view(-1).long()
    atom = inds[atom].clamp(0, 8)
    x = torch.nn.functional.one_hot(atom, num_classes=9).float()

    # ---- edge features (3 classes: 0..3) ----
    ea = graph.edge_attr
    if ea is None:
        ea = torch.zeros(
            graph.edge_index.size(1), dtype=torch.long, device=graph.edge_index.device
        )
    else:
        # if already one-hot, convert to scalar labels
        if ea.dim() == 2 and ea.size(-1) > 1:
            ea = ea.argmax(dim=-1)
        ea = ea.view(-1).long()

    ea = ea.clamp(0, 3)
    edge_attr = torch.nn.functional.one_hot(ea, num_classes=4).float()

    return Data(x=x, edge_index=graph.edge_index, edge_attr=edge_attr)
