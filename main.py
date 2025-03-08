import torch
import re
import random
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from transformers import (
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
import config
import model as model_lib
import buffer as replay_buffer
from loss import GRPOLoss, approx_kl_divergence 

ExperimentConfig = config.ExperimentConfig
writer = SummaryWriter(log_dir=ExperimentConfig.log_dir)


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def compute_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes advantages based on the group outputs."""
    # The input is of shape: [G, 1] i.e. one value for each output of the group.
    # and the output is again of shape: [G, 1] with the value calculated as:
    #      (return - mean)
    #       -------------
    #         std + eps (small balancing factor)
    return (returns - returns.mean()) / (returns.std() + eps)


@torch.no_grad
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    ground_truth: str,
    group_size: int,
    max_len: int = 1024,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Generate responses to the input task and calculate returns."""

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": config.SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # 2. Duplicate prompt group_size times
    # Since we need to generate `group_size` num of responses for each input, we
    # duplicate it along the 1st axis.
    # The size here is: [G, seq_len]
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        group_size, 1
    )
    input_ids = model_inputs["input_ids"].repeat(group_size, 1)
    model_inputs["input_ids"] = input_ids

    # 3. Generate sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_len,
        pad_token_id=pad_token_id,
    )
    # Output is of size: [G, seq_len] where each element is seq_len is a token
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    # 4. Generate Action mask
    # We want to compute losses/rewards for the model's actions, thus this mask.
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 5. Compute Rewards
    returns = torch.zeros(group_size, 1, dtype=torch.float)
    numbers = [int(x) for x in ground_truth.split(", ")]
    for idx, completion in enumerate(completions):
        reward = 0

        # Getting the answer
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )
        if answer_match:
            answer = answer_match.group(1)
            model_nums = [int(x) for x in answer.split(", ")]
            # Ensure lengths match (if model output is malformed, penalize heavily)
            if len(model_nums) == len(numbers):
                # Fraction of correctly placed elements
                correct_count = sum(1 for m, c in zip(model_nums, numbers) if m == c)
                total = len(numbers)
                reward = correct_count / total
        
        returns[idx] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask


def read_data(path: str, max_limit: int = 1024):
    data = []
    with open(path) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    
    return data[:max_limit]


def run():
    config = ExperimentConfig()

    device = torch.device("cuda", config.device_index)
    device_cpu = torch.device("cpu")

    init_rng(config.seed)
    reference_model, _ = model_lib.load_model(config.model_name, device_map=device)
    reference_model.eval()  # Setting the model in eval state.

    model, tokenizer = model_lib.load_model(config.model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Enable gradient checkpointing to reduce memory footprint during back-prop
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    prompts = read_data("data/math_tasks.jsonl", max_limit=128*1024)
    data_loader = DataLoader(
        prompts,
        batch_size=config.num_episodes_per_step
        * config.batch_size,  # See config for details
        shuffle=True,
        drop_last=True,  # To make sure each batch is of equal length
        pin_memory=False,  # Since the data is not so huge, we defer from using the page-locked memory to transfer data CPU -> GPU
    )
    buffer = replay_buffer.ReplayBuffer()
    objective = GRPOLoss(clip_eps=config.clip_eps, kl_weight=config.kl_weight)

    global_step = 0
    for k, prompt_batch in enumerate(data_loader):
        rollout_returns = []
        buffer.clear()

        # Size: [B, seq_len] ideally but the q & a are not padded here yet so
        # the might have variable seq_len but before going into the model for prediction
        # we'll pad them via the tokenizer.
        tasks = prompt_batch["task"]
        ground_truths = prompt_batch["ground_truth"]

        with torch.no_grad():
            for t, gt in zip(tasks, ground_truths):
                sequence_ids, returns, action_mask = rollout(
                    model=model,
                    tokenizer=tokenizer,
                    task=t,
                    ground_truth=gt,
                    group_size=config.group_size,
                    max_len=config.max_len,
                    top_p=config.top_p,
                    temperature=config.temperature,
                )
                rollout_returns.append(returns.cpu())

                # Compute group advantages for the rollouts
                advantages = compute_advantages(returns)

                # Compute the attention mask and log probs for the sequences
                attention_mask = sequence_ids != tokenizer.eos_id
                log_probs = model_lib.get_sequence_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                log_probs_ref = model_lib.get_sequence_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                # Storing the data points for computation over the policy
                buffer.append(
                    sequence_ids,
                    log_probs,
                    log_probs_ref,
                    returns,
                    advantages,
                    attention_mask,
                    action_mask,
                    kl,
                    device_cpu,
                )

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")

        # Creating a separate dataloader to train the policy on the current experiences.
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=replay_buffer.batch_experiences,
        )

        # We train for epochs_per_step for the current gathered data.
        # Since we dont want model to overfit the data, you'll see this loop is
        # different than in normal SFT. Here instead of showing model the same data again and
        # again, we just take 1 epoch over this data and then discard it.
        for step_epoch in range(config.epochs_per_step):
            model.train()

            for idx, exp in enumerate(experience_sampler):
                exp: replay_buffer.Experience
                exp = exp.to(device)

                optimizer.zero_grad()
                # Get the current log_probs based on the updated model and then we compute the loss
                # between this and the old log probs.
                log_probs = model_lib.get_sequence_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )
                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={exp.advantages}")
                    continue

                writer.add_scalar(f"GRPOLoss/train@{config.group_size}", loss, global_step)
                writer.add_scalar(f"KL-divergence/train@{config.group_size}", kl, global_step)
                global_step += 1

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                optimizer.step()

        if (
            config.checkpoint_path is not None
            and config.checkpoint_interval is not None
            and (k + 1) % config.checkpoint_interval == 0
        ):
            model.save_pretrained(config.checkpoint_path / f"step_{k}")

    if config.checkpoint_path is not None:
        model.save_pretrained(config.checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    run()