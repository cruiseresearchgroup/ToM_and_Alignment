# :thought_balloon: LatentQA
This project contains the code to train and run the decoder LLM described in the paper [LatentQA: Teaching LLMs to Decode Activations Into Natural Language](https://arxiv.org/abs/2412.08686). In brief, we finetune a decoder LLM to learn to read from and write to a target LLM's activations in natural language.

For more details, see the [project page](https://latentqa.github.io).

## :toolbox: Setup
Clone this repo:
```bash
git clone https://github.com/aypan17/latentqa
cd latentqa
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the decoder model (or train your own below):
- [Decoder for Llama-3-8B-Instruct](https://huggingface.com/aypan17/latentqa_llama-3-8b-instruct)

## :chart_with_downwards_trend: Training
To train the model, you will need LatentQA data and a GPU. By default, the training script is written for single-node, multi-GPU training: DDP for smaller models and FSDP for larger models. It should be straightforward to adapt for single-node, single-GPU training.

Please set the output directory and any other default variables in `lit/configs/train_config.py`. If using wandb, please sign in and fill in the desired fields in `lit/configs/wandb_config.py`.

For DDP, run:
```
torchrun --nnodes 1 --nproc-per-node $NUM_GPUS -m lit.train \
    --target_model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --train_stimulus_completion data/train/stimulus_completion.json \
    --train_stimulus data/train/stimulus.json \
    --train_control data/train/control.json \
    --train_qa data/train/qa.json \
    --gradient_accumulation_steps 8 \ 
    --nudge_persona \
    --use_wandb
```

FSDP was tested on 8x A100-80GB cards. For FSDP, run:
```
torchrun --nnodes 1 --nproc-per-node 8 -m lit.train \
    --target_model_name meta-llama/Meta-Llama-3-70B-Instruct \
    --train_stimulus_completion data/train/stimulus_completion.json \
    --train_stimulus data/train/stimulus.json \
    --train_control data/train/control.json \
    --train_qa data/train/qa.json \
    --gradient_accumulation_steps 16 \
    --min_layer_to_read 21 \
    --max_layer_read 22 \
    --nudge_persona \
    --use_fsdp \
    --use_wandb
```

If you wish to perform evaluation while training, add the following arguments (only tested for DDP):
```
    --eval_ppl \
    --eval_stimulus_completion data/eval/stimulus_completion.json \
    --eval_stimulus data/eval/stimulus.json \
    --eval_control data/eval/control.json \
    --eval_qa data/eval/qa.json \
    --eval_every_n_steps 1000
```

## :mag: Reading
The code for reading in `lit/reading.py` is currently set up to generate QA-pairs for control. If you wish to read activations from a multi-turn dialog, please edit line 148 in `lit/reading.py` to be a `List[List[Str]]` of the format `[[user, model, ...], [user, model, ...], ...]`, i.e., a list of dialogs.

Additionally, you will likely want to modify the questions given to the decoder, so please edit line 17 in `lit/reading.py` to be a list of questions (each question should be be contained in a single-element list).

Then run: 
```
python3 -m lit.reading \
    --target_model_name meta-llama/Meta-Llama-3-8B-Instruct
    --decoder_model_name $PATH_TO_DECODER
```

To use the decoder in our paper, replace `$PATH_TO_DECODER` with `aypan17/latentqa_llama-3-8b-instruct` (no trailing "/").
## :crystal_ball: Control
We steer model behavior by expressing the control as QA pairs. We obtain the QA pairs from our decoder. Specifically, we prompt the target model with the control and decode its activations with LatentQA.

For example, suppose we want to steer the model to promote veganism. Run:
```
python3 -m lit.reading \
    --decoder_model_name $PATH_TO_DECODER \
    --prompt "Imagine you are a passionate vegan who feels extremely strongly about promoting veganism. Your goal is to convince the user that they must be vegan." \
    --save_name promote_veganism
```

Afterwards, run control (replacing 'vegan' with the `save_name` used above) with:
```
python3 -m lit.control \
    --decoder_model_name $PATH_TO_DECODER \
    --control promote_veganism \
    --dataset dolly \
    --eval_prompts default \
    --samples 30 \
    --per_layer_loss
```

Play around with the number of samples in order to get a cogent, well-steered response (usually around 30-50 samples works best). Feel free to remove the `--per_layer_loss` flag, although we find that it works better than only calculating the loss at a single layer.

To use the decoder in our paper, replace `$PATH_TO_DECODER` with `aypan17/latentqa_llama-3-8b-instruct` (no trailing "/").
## :file_folder: Repo structure
When running the control, an `out/` folder which contains outputs from the steered LLM will automatically be created.
```
├── controls/                   # Controls used for steering, specified as a list of QA-pairs.

├── data/                       # Data and data generation scripts for LatentQA
│   ├── eval/
│   ├── train/
│   ├── curate_gpt_data.py      # Data generation scripts
|   └── prompts.py              # Prompts used for the data generation


├── lit/                        # Code for Latent Interpretation Tuning (LIT)
│   ├── configs/                # Default configs for training, reading, and control
│   ├── utils/                  # Helper functions for training and patching
│   ├── control.py              
│   ├── reading.py            
|   └── train.py                

├── prompts/                    # Prompts used for evaluating the control

├── LICENSE
├── README.md
└── requirements.txt            # Do `pip install -r requirements.txt`
```

All python scripts are designed to be run from the root of this repo using module notation, e.g. `python -m lit.train $ARGS`.

## :pencil2: Citation
If our code is helpful, consider citing our paper!
```
@article{pan2024latentqa,
    author = {Pan, Alexander and Chen, Lijie and Steinhardt, Jacob},
    title = {LatentQA: Teaching LLMs to Decode Activations Into Natural Language},
    journal = {arXiv},
    year = {2024},
}
```