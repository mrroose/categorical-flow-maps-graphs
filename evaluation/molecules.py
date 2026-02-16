import logging
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
from tqdm import tqdm

import wandb
from utils.molecule_utils import mol2smiles

logger = logging.getLogger(__name__)

_FCD_REF_MODEL = None


def _tqdm_safe_log(msg: str, enable: bool = True):
    if not enable:
        return
    # avoids breaking tqdm progress bars in slurm logs
    tqdm.write(msg)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.2f}"


def _get_fcd_ref_model():
    global _FCD_REF_MODEL
    if _FCD_REF_MODEL is None:
        from fcd import load_ref_model

        _FCD_REF_MODEL = load_ref_model()
    return _FCD_REF_MODEL


def evaluate_molecules(
    mols: List,
    train_smiles: List,
    val_smiles: List,
) -> Dict[str, Any]:
    valid_mols = []
    valid_smiles = []

    # Process molecules
    for mol in mols:
        smiles = mol2smiles(mol)  # Sanitize + to smiles, None if fail
        if smiles is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                valid_mols.append(largest_mol)
                smiles_largest = mol2smiles(largest_mol)
                valid_smiles.append(smiles_largest)
            except Exception:
                pass

    unique_smiles = list(set(valid_smiles))
    denom = len(mols) if len(mols) > 0 else 1
    percentage_valid = len(valid_smiles) / denom
    percentage_unique = (
        (len(unique_smiles) / len(valid_smiles)) if len(valid_smiles) > 0 else 0.0
    )
    percentage_valid_and_unique = len(unique_smiles) / denom
    FCD_DEFAULT_VALUE = None

    if len(unique_smiles) > 0:
        from fcd import canonical_smiles, get_predictions

        model = _get_fcd_ref_model()

        can_val = [w for w in canonical_smiles(val_smiles, njobs=2) if w is not None]
        can_gen = [w for w in canonical_smiles(unique_smiles, njobs=2) if w is not None]
        can_train = [
            w for w in canonical_smiles(train_smiles, njobs=2) if w is not None
        ]

        # novelty (guard against empty)
        can_novel = set(can_gen) - set(can_train)
        novelty = (len(can_novel) / len(can_gen)) if len(can_gen) > 0 else 0.0

        # FCD needs >=2 samples to form covariances
        if len(can_val) < 2 or len(can_gen) < 2:
            fcd_score = FCD_DEFAULT_VALUE
        else:
            test_fc = get_predictions(model, can_val)
            gen_fc = get_predictions(model, can_gen)

            if len(test_fc) < 2 or len(gen_fc) < 2:
                fcd_score = FCD_DEFAULT_VALUE
            else:
                mu_real, sigma_real = np.mean(test_fc, axis=0), np.cov(
                    test_fc, rowvar=False
                )
                mu_gen, sigma_gen = np.mean(gen_fc, axis=0), np.cov(
                    gen_fc, rowvar=False
                )

                def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
                    from scipy.linalg import sqrtm

                    mu_diff = mu1 - mu2
                    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
                    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
                    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
                    if np.iscomplexobj(covmean):
                        covmean = covmean.real
                    trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)
                    return (mu_diff.dot(mu_diff) + trace_term).real

                try:
                    fcd_score = calculate_frechet_distance(
                        mu_real, sigma_real, mu_gen, sigma_gen
                    )
                except Exception:
                    logger.warning("Error calculating FCD score.")
                    fcd_score = FCD_DEFAULT_VALUE
    else:
        novelty = 0.0
        fcd_score = FCD_DEFAULT_VALUE

    return {
        "valid_mols": valid_mols,
        "percentage_valid": percentage_valid,
        "percentage_unique": percentage_unique,
        "percentage_valid_and_unique": percentage_valid_and_unique,
        "novelty": novelty,
        "fcd_score": fcd_score,
    }


def eval_and_log(
    mols: List,
    log: bool,
    train_smiles: List,
    val_smiles: List,
    n_steps: int,
    global_step: Optional[int],
    prefix: str,
) -> List:
    metrics = evaluate_molecules(
        mols=mols,
        train_smiles=train_smiles,
        val_smiles=val_smiles,
    )

    steps_key = f"eval/{prefix}/{n_steps}_steps"
    msg = (
        f"[{prefix} {n_steps} steps] valid={_fmt(metrics['percentage_valid'])}, "
        f"unique={_fmt(metrics['percentage_unique'])}, "
        f"novelty={_fmt(metrics['novelty'])}, fcd={_fmt(metrics['fcd_score'])}"
    )
    _tqdm_safe_log(msg, enable=log)
    logger.info(msg)

    if log:
        wandb.log(
            {
                f"{steps_key}/valid": metrics["percentage_valid"],
                f"{steps_key}/unique": metrics["percentage_unique"],
                f"{steps_key}/v&u": metrics["percentage_valid_and_unique"],
                f"{steps_key}/novelty": metrics["novelty"],
                f"{steps_key}/fcd": metrics["fcd_score"],
            },
            step=global_step,
        )

    return metrics["valid_mols"]
