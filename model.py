import torch
import torch.nn.functional as F
import logging
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
)


logger = logging.getLogger(__name__)


def load_model(
    model_name_or_path: str,
    # filename: str,
    # tokenizer_name_or_path: str,
    use_bfloat16: bool = True,
    device_map: ... = None
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    logger.debug("Loading model tokenizer from hub.")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    logger.info("Loaded tokenizer.")
    tokenizer.pad_token = tokenizer.eos_token
    logger.debug("Loading model from hub.")
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bfloat16 else "auto",
        # attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    logger.info("Loaded model.")
    return model, tokenizer


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    logger.debug("Computing log probs from logits.")
    # logits:        [B, seq_len, vocab_size]
    # output_ids:    [B, seq_len]

    # log_prob:      [B, seq_len, vocab_size] --> just the difference being that the values in -1'th 
    # dimension now sum upto 1 (i.e. are normalized and represent probabilities in the log space)
    log_prob = F.log_softmax(logits, dim=-1)
    # final outpit:  [B, seq_len]
    # The steps are as below:
    # 1. unsqueeze the output_ids, making the dimension [B, seq_len, 1]
    # 2. Now gather the log probability corresponding to the token given in output_ids,
    #       specifically if the output ids are: [1, 3, 5] then it takes the 1st, 3rd and 5th
    #       values of the log_prob
    # 3. This returns a tensor of [B, seq_len, 1]
    # 4. Finally applying squeeze gives us [B, seq_len]
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def get_sequence_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor :
    logger.debug("Generating log probabilities for the output sequence.")
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    # position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    # Shape: [B, seq_len, vocab_size]
    # This is the raw, un-normalized values for each of the output tokens
    logits = output["logits"]
    # We get the corresponding log probabilities for each output token in the
    # shape: [B, seq_len]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs