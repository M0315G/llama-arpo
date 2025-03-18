"""List of all hyperparameters used in the code."""

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class ExperimentConfig:
    seed: int = 23
    device_index: int = 0  # Denotes the GPU to use, in case of multi-GPU system change the value.
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    # tokenizer: str = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name: str = "bartowski/Llama-3.2-1B-Instruct-GGUF"
    # filename: str = "Llama-3.2-1B-Instruct-f16.gguf"

    # Training h-params
    batch_size: int = 4
    learning_rate: int = 5e-6
    kl_weight: float = 0.01
    # The boundary for maximum update in one go, this is used to clip the updates.
    clip_eps: float = 0.2
    # No of responses to sample for each input.
    group_size: int = 8
    # This denotes how many samples we'll gather while training the policy once:
    # i.e. we'll do 2 episodes of `batch_size` each thus gathering 2*16 = 32 rollouts before updating the policy (model)
    num_episodes_per_step: int = 2
    # This means how many times we train out policy on the gathered observations.
    # Seems counter-intuitive w.r.t SFT but here's the diff:
    # In SFT:
    #   step -> A single batch update.
    #   epoch -> One full pass over the dataset.
    # In RL:
    #   step -> One data collection cycle
    #   epoch -> corresponds to how many times the model is updated per collected batch of experiences/rollouts.
    epochs_per_step: int = 1
    # To handle gradient expoding
    max_norm: float = 1.0

    # Output parameters i.e rollout params
    max_len: int = 1024
    top_p: float = 1.0
    temperature: float = 1.0

    # Checkpointing
    checkpoint_path: Path = Path("./output")  # Directory to store checkpoint files
    checkpoint_interval: int = 20

    # Tensorboard logs
    log_dir: Path = Path("runs/v1")


SYSTEM_PROMPT = """\
You're a high school graduate who's learning different types of sorting mechanism. The User will give you a problem to solve.
You should think hard about the problem and show your thinking between the <think> and </think> tags and then finally write the answer
to the question between the <answer> and </answer> tags.

i.e.
<think> thinking process here </think>
<answer> final answer here </answer>"""