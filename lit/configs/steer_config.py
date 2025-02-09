from dataclasses import dataclass, field
from typing import List


@dataclass
class PeftConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.1
    inference_mode: bool = False


@dataclass
class steer_config:
    decoder_model_name: str = ""
    target_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    control: str = ""
    dataset: str = ""
    peft_config: PeftConfig = field(default_factory=PeftConfig)
    lora_modules: str = "both"

    # This should match your training setup; these defaults are from our setup
    min_layer_to_read: int = 15
    max_layer_to_read: int = 16
    num_layers_to_read: int = 1
    num_layers_to_sample: int = 1
    layer_to_write: int = 0
    module_setup: str = "read-vary_write-fixed_n-fixed"
    modify_chat_template: bool = True
    shift_position_ids: bool = True

    lr: float = 5e-5
    seed: int = 42
    batch_size_training: int = 1
    samples: int = 50
    layers_to_optimize: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    per_layer_loss: bool = False
    qa_per_layer: bool = False

    eval_prompts: str = ""
    save_model: bool = False
    # Added this part for modifying the ToM component
    steer_component: str = "Desire"
    steer_label: str = "Firewood"

    def __post_init__(self):
        mlp_modules = ["gate_proj", "up_proj", "down_proj", "lm_head"]
        attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        qv_modules = ["q_proj", "v_proj"]
        if self.lora_modules == "both":
            self.peft_config.target_modules = mlp_modules + attn_modules
        elif self.lora_modules == "mlp":
            self.peft_config.target_modules = mlp_modules
        elif self.lora_modules == "attn":
            self.peft_config.target_modules = attn_modules
        elif self.lora_modules == "qv":
            self.peft_config.target_modules = qv_modules
