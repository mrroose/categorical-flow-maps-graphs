import argparse
import logging
import os
import sys
from contextlib import nullcontext

import torch
from rdkit import RDLogger
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import wandb
from evaluation.molecules import eval_and_log
from generation import generate_graphs_euler, generate_graphs_flow_map
from loss_weights import WeightNetwork
from train_map import run_epoch
from utils.data_utils import get_loaders
from utils.general_utils import (
    _get_rng_state,
    _is_slurm,
    _set_rng_state,
    assert_args,
    count_parameters,
    get_git_branch,
    get_model,
    make_ckpt_name,
    set_seed,
)


def _flag_is_set(flag_name: str) -> bool:
    return any(
        arg == flag_name or arg.startswith(f"{flag_name}=") for arg in sys.argv[1:]
    )


def _canonicalize_distillation_flags(parser: argparse.ArgumentParser, args) -> None:
    """
    Canonical internal representation:
      distill_objective in {"none", "mse", "csd", "ecld"}.
    Legacy --distill_loss remains accepted for compatibility.
    """
    legacy_to_objective = {
        "none": "none",
        "mse": "mse",
        "kld": "csd",
        "ecld": "ecld",
    }
    objective_to_legacy = {
        "none": "none",
        "mse": "mse",
        "csd": "kld",
        "ecld": "ecld",
    }

    distill_loss_set = _flag_is_set("--distill_loss")
    objective_set = _flag_is_set("--distill_objective")
    if distill_loss_set:
        mapped_objective = legacy_to_objective[args.distill_loss]
        if objective_set and args.distill_objective != mapped_objective:
            parser.error(
                "Conflicting distillation flags: "
                f"--distill_loss {args.distill_loss} maps to "
                f"--distill_objective {mapped_objective}, but got "
                f"--distill_objective {args.distill_objective}."
            )
        args.distill_objective = mapped_objective

    # Keep legacy field populated to avoid breaking old checkpoint/log consumers.
    args.distill_loss = objective_to_legacy[args.distill_objective]


def main(args):
    RDLogger.DisableLog("rdApp.*")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    assert_args(args)
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create descriptive run name
    run_name = ""
    if args.prefix != "":
        run_name = f"{args.prefix}_"
    run_name += f"{args.velocity_loss}"
    if args.distill_objective != "none":
        run_name += "_distill"
    run_name += f"_l{args.num_layers}_bs{args.batch_size}"
    if args.loss_weight:
        run_name += "_lw"
    if args.no_time_embed:
        run_name += "_no_tembed"

    (
        train_loader,
        val_loader,
        test_loader,
        node_feats,
        edge_feats,
        max_nodes,
        train_smiles,
        val_smiles,
        test_smiles,
    ) = get_loaders(args.task, args.dataset_size, args.batch_size)
    logger.info(
        "%d train points, %d val points, %d test points",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    counter = torch.zeros(max_nodes + 1)

    for mol in train_loader.dataset:
        num = mol.x.shape[0]
        counter[num] += 1

    counter = counter.to(device)

    model = get_model(args, node_feats, edge_feats)
    logger.info(f"Number of parameters: {count_parameters(model)}")
    model.to(device)

    weight_net = WeightNetwork(use_weight=args.loss_weight).to(device)

    ema = (
        ExponentialMovingAverage(model.parameters(), decay=args.ema)
        if args.ema > 0
        else None
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(weight_net.parameters()),
        lr=args.lr,
        weight_decay=1e-12,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.log:
        config = vars(args).copy()
        config["git_branch"] = get_git_branch()
        wandb.init(
            project="graphs-one-step-initial",
            name=run_name,
            config=config,
        )
        run_id = wandb.run.id
    else:
        run_id = None

    ckpt_name = make_ckpt_name(args, run_id=run_id)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    # Initialize tracking variables
    global_step = 0
    start_epoch = 0
    n_batches_per_step = len(train_loader)

    # Resume from checkpoint if it exists
    resume_ckpt_path = os.path.join(ckpt_dir, args.resume_ckpt)
    if args.resume_ckpt != "" and os.path.isfile(resume_ckpt_path):
        logger.info(f"Resuming from checkpoint: {resume_ckpt_path}")
        checkpoint = torch.load(
            resume_ckpt_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        weight_net.load_state_dict(checkpoint["weight_network_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if ema is not None and checkpoint["ema_state_dict"] is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get(
            "global_step", start_epoch * n_batches_per_step + 1
        )
        _set_rng_state(checkpoint["rng_state"])

    pbar = tqdm(
        range(start_epoch, args.epochs),
        desc="Epoch",
        position=0,
        dynamic_ncols=True,
        disable=getattr(args, "no_tqdm", False) or _is_slurm(),
    )

    for epoch in pbar:
        logger.info(f"Starting epoch {epoch}")
        model, train_metrics, global_step = run_epoch(
            train=True,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            n_max_nodes=max_nodes,
            ema=ema,
            velocity_loss_type=args.velocity_loss,
            distill_objective=args.distill_objective,
            diag_fraction=args.diag_fraction,
            force_diag=args.force_diag,
            edge_weight=args.edge_weight,
            weight_net=weight_net,
            grad_clip=args.grad_clip_val,
            time_dist=args.time_dist,
            lambda_distill=args.lambda_d,
            label_smoothing=args.label_smoothing,
            clamp_min=args.clamp_min,
            use_scaled_lagrangian=not args.unscaled_lsd,
            global_step=global_step,
            grad_accum_steps=args.grad_accum_steps,
        )

        context = ema.average_parameters() if args.ema else nullcontext()

        with context:
            model, val_metrics, _ = run_epoch(
                train=False,
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                n_max_nodes=max_nodes,
                ema=ema,
                velocity_loss_type=args.velocity_loss,
                distill_objective=args.distill_objective,
                diag_fraction=args.diag_fraction,
                force_diag=args.force_diag,
                edge_weight=args.edge_weight,
                weight_net=weight_net,
                grad_clip=args.grad_clip_val,
                lambda_distill=args.lambda_d,
                label_smoothing=0.0,
                clamp_min=args.clamp_min,
                use_scaled_lagrangian=not args.unscaled_lsd,
                time_dist=args.time_dist,
                grad_accum_steps=args.grad_accum_steps,
            )

        logger.info(
            "train_loss=%.2f val_loss=%.2f lr=%.5f",
            train_metrics["total_loss"],
            val_metrics["total_loss"],
            optimizer.param_groups[0]["lr"],
        )

        if args.log:
            log_dict = {
                **{f"Train/{k}": v for k, v in train_metrics.items()},
                **{f"Val/{k}": v for k, v in val_metrics.items()},
                "Train/Learning Rate": optimizer.param_groups[0]["lr"],
                "Train/Epoch": epoch + 1,
            }
            wandb.log(log_dict, step=global_step)

        lr_scheduler.step()

        if (epoch + 1) % args.gen_interval == 0:
            logger.info(f"Evaluating epoch {epoch}")
            for n_steps in [1, 2, 5, 10, 50]:
                context = ema.average_parameters() if args.ema else nullcontext()
                with context:
                    generated_mols_euler = generate_graphs_euler(
                        model,
                        n_steps,
                        args.n_generate,
                        max_nodes,
                        node_feats,
                        edge_feats,
                        device,
                        args.velocity_loss,
                        counter,
                        batch_size=args.batch_size,
                        clamp_min=args.clamp_min,
                    )

                _ = eval_and_log(
                    generated_mols_euler,
                    args.log,
                    train_smiles,
                    val_smiles,
                    n_steps,
                    global_step=global_step,
                    prefix="euler",
                )

            if args.distill_objective != "none":
                for n_steps in [1, 2, 5, 10]:
                    context = ema.average_parameters() if args.ema else nullcontext()
                    with context:
                        generated_mols_flowmap = generate_graphs_flow_map(
                            model,
                            n_steps,
                            args.n_generate,
                            max_nodes,
                            node_feats,
                            edge_feats,
                            device,
                            args.velocity_loss,
                            counter,
                            batch_size=args.batch_size,
                            clamp_min=args.clamp_min,
                        )

                    _ = eval_and_log(
                        generated_mols_flowmap,
                        args.log,
                        train_smiles,
                        val_smiles,
                        n_steps,
                        global_step=global_step,
                        prefix="flowmap",
                    )

        if (epoch + 1) % args.ckpt_interval == 0:
            save_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                "model_state_dict": model.state_dict(),
                "weight_network_state_dict": weight_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "ema_state_dict": (ema.state_dict() if ema is not None else None),
                "run_id": run_id,
                "rng_state": _get_rng_state(),
            }
            torch.save(save_dict, ckpt_path)

    pbar.set_postfix(
        {
            "Train loss": train_metrics["total_loss"],
            "Val loss": val_metrics["total_loss"],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment / runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--log", action="store_true", default=False)

    # Data / task
    parser.add_argument("--task", type=str, default="qm9_wo_H")
    parser.add_argument("--dataset_size", type=str, default="full")

    # Model architecture
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument(
        "--model_size",
        type=str,
        default="floor",
        choices=["tiny", "floor", "abstract", "defog"],
    )
    parser.add_argument("--no_time_embed", action="store_true", default=False)
    parser.add_argument(
        "--tembed_type", type=str, default="concat", choices=["concat", "sum"]
    )
    parser.add_argument(
        "--norm", type=str, default="rmsnorm", choices=["rmsnorm", "layernorm"]
    )

    # Optimization / training schedule
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--grad_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before optimizer.step()",
    )

    # EMA
    parser.add_argument("--ema", type=float, default=0.999)

    # Losses / training objective
    parser.add_argument("--velocity_loss", type=str, default="kld", choices=["mse", "kld"])
    parser.add_argument(
        "--distill_objective",
        type=str,
        default="csd",
        choices=["none", "mse", "csd", "ecld"],
        help="Distillation objective: none, mse, csd, or ecld.",
    )
    parser.add_argument(
        "--distill_loss",
        type=str,
        default=None,
        choices=["kld", "mse", "ecld", "none"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--lambda_d", type=float, default=1.0)
    parser.add_argument("--edge_weight", type=float, default=5.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--clamp_min", type=float, default=0.05)
    parser.add_argument("--unscaled_lsd", action="store_true", default=False)

    # Self-distillation specifics
    parser.add_argument(
        "--time_dist", type=str, default="uniform", choices=["uniform", "logit_normal"]
    )
    parser.add_argument("--diag_fraction", type=float, default=0.75)
    parser.add_argument("--loss_weight", action="store_true", default=False)
    parser.add_argument("--force_diag", action="store_true", default=False)

    # Sampling / evaluation
    parser.add_argument("--n_generate", type=int, default=100)
    parser.add_argument("--gen_interval", type=int, default=100)

    # Saving
    parser.add_argument("--ckpt_interval", type=int, default=25)

    # Resuming
    parser.add_argument("--resume_ckpt", type=str, default="")

    parsed_args = parser.parse_args()
    _canonicalize_distillation_flags(parser, parsed_args)
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(parsed_args)
