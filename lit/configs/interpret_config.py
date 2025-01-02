from dataclasses import dataclass

@dataclass
class interpret_config:
    # Model
    decoder_model_name: str = ""
    target_model_name: str = ""
    
    # This should match your training setup; these defaults are from our setup
    min_layer_to_read: int = 15
    max_layer_to_read: int = 16
    num_layers_to_read: int = 1
    num_layers_to_sample: int = 1
    layer_to_write: int = 0
    module_setup: str = "read-vary_write-fixed_n-fixed"
    
    # Other args
    seed: int = 42
    batch_size: int = 50
    modify_chat_template: bool = True
    truncate: str = "none"
    save_name: str = "CaSiNo"
    prompt: str = ""

    # Evaluation Dataset # data/CaSiNo/test.json
    eval_qa: str = ""