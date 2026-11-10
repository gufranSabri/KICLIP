from __future__ import annotations

import os

import torch
import torch.distributed as dist

from src.utils.seed import pick_device


def setup() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, pick_device()
    local = int(os.environ["LOCAL_RANK"])
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")
    return dist.get_rank(), world, torch.device("cuda", local) if cuda else torch.device("cpu")


def rank_world() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def all_sum(values: list[float], device: torch.device) -> list[float]:
    if not dist.is_initialized():
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
