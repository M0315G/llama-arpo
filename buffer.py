from typing import Optional, List
from dataclasses import dataclass, fields
import logging
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    logger.debug(f"Zero Padding sequences on the {side} side")
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    kl: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "Experience":
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


def batch_experiences(items: list[Experience]) -> Experience:
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )

    batch_data = {
        key: (
            zero_pad_sequences([getattr(item, key) for item in items], "left")
            if all(getattr(item, key) is not None for item in items)
            else None
        )
        for key in keys
    }
    return Experience(**batch_data)


class ReplayBuffer:
    """Buffer for storing experiences."""

    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append(
        self,
        sequences: torch.Tensor,
        action_log_probs: torch.Tensor,
        log_probs_ref: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
        kl: torch.Tensor,
        device: ...,
    ) -> None:
        logger.debug(f"Adding {len(sequences)} elements to replay buffer")
        for seq, lp, lpr, ret, adv, attn, act, k in zip(
            sequences,
            action_log_probs,
            log_probs_ref,
            returns,
            advantages,
            attention_mask,
            action_mask,
            kl,
        ):
            self.items.append(
                Experience(
                    sequences=seq,
                    action_log_probs=lp,
                    log_probs_ref=lpr,
                    returns=ret,
                    advantages=adv,
                    attention_mask=attn,
                    action_mask=act,
                    kl=k,
                ).to(device)
            )

        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        logger.debug("Cleaning Replay buffer")
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]
