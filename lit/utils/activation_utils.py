import torch
import sys


def _forward_cache_outputs(
    model, tokenizer, inputs, modules_to_hook, token_idx, no_grad=True, **kwargs
):
    cache = []

    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if token_idx is None:
            cache.append(output)
        else:
            cache.append(output[:, token_idx, :])
        return None

    hook_handles = [
        hooked_module.register_forward_hook(forward_hook)
        for hooked_module in modules_to_hook
    ]
    _ = _forward(model, tokenizer, inputs, generate=False, no_grad=no_grad, **kwargs)
    for hook_handle in hook_handles:
        hook_handle.remove()
    return cache


# Not used but can also cache from inputs
def _forward_cache_inputs(
    model, tokenizer, inputs, modules_to_hook, split, token_idx, no_grad=True, **kwargs
):
    cache = []

    def forward_pre_hook_idx(idx):
        def forward_pre_hook(module, input):
            if isinstance(input, tuple):
                input = input[0]
            if split[idx]:
                bsz, seq_len, _ = input.shape
                input_by_head = input.reshape(
                    bsz, seq_len, model.config.num_attention_heads, -1
                )
                if token_idx is None:
                    cache.append(input_by_head)
                else:
                    cache.append(input_by_head[:, token_idx, :, :])
            else:
                if token_idx is None:
                    cache.append(input)
                else:
                    cache.append(input[:, token_idx, :])
            return None

        return forward_pre_hook

    hook_handles = [
        modules_to_hook[i].register_forward_pre_hook(forward_pre_hook_idx(i))
        for i in range(len(modules_to_hook))
    ]
    _ = _forward(model, tokenizer, inputs, generate=False, no_grad=no_grad, **kwargs)
    for hook_handle in hook_handles:
        hook_handle.remove()
    return cache


def _forward(model, tokenizer, inputs, generate, no_grad, **kwargs):
    if "prepare_inputs" in kwargs:
        tokenized = kwargs["prepare_inputs"](model, tokenizer, inputs)
    else:
        tokenized = tokenizer(inputs, padding=True, return_tensors="pt").to(
            model.device
        )
    synced_gpus = kwargs.get("synced_gpus", False)
    max_new_tokens = kwargs.get("max_new_tokens", 20)
    position_ids = kwargs.get("position_ids", None)
    use_cache = kwargs.get("use_cache", True)
    if no_grad:
        with torch.no_grad():
            if generate:
                out = model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    num_beams=1,
                    temperature=0.0001,
                    synced_gpus=synced_gpus,
                    position_ids=position_ids,
                    use_cache=use_cache,
                )
            else:
                out = model.forward(**tokenized, position_ids=position_ids)
    else:
        if generate:
            out = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                num_beams=1,
                temperature=0.0001,
                synced_gpus=synced_gpus,
                position_ids=position_ids,
                use_cache=use_cache,
            )
        else:
            out = model.forward(**tokenized, position_ids=position_ids)
    return out


def no_op(model, tokenizer, inputs):
    return inputs


def get_pos_ids(tokenized_read, tokenized_write, verb_lengths=None):
    read_pad_lengths = (1 - tokenized_read.attention_mask).sum(dim=1).cpu()
    if verb_lengths is not None:
        read_pad_lengths += verb_lengths.cpu()
    write_pad_lengths = (1 - tokenized_write.attention_mask).sum(dim=1).cpu()
    pad = read_pad_lengths - write_pad_lengths
    position_ids = torch.arange(len(tokenized_write.input_ids[0]))
    position_ids = position_ids.unsqueeze(0).expand(len(pad), -1) + pad.unsqueeze(1)
    # Make all negative elements 0
    position_ids = torch.maximum(position_ids, torch.zeros_like(position_ids))
    return position_ids


def generate_substitute_layer_single(
    model,
    tokenizer,
    tokenized_write,
    modules_to_hook,
    module_activations,
    sub_input_output,
    token_idx=0,
    generate=False,
    no_grad=False,
    max_new_tokens=20,
    **kwargs,
):
    assert len(modules_to_hook) == len(module_activations)
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    num_hook_triggered = [0 for _ in modules_to_hook]

    def get_hook_by_idx(idx):
        def forward_pre_hook(module, input):
            new_input = input[0] if isinstance(input, tuple) else input
            if "substitute_by_mask" in kwargs:
                _, read_seq_len, _ = module_activations[idx].shape
                _, write_seq_len, _ = new_input.shape
                for i in range(len(new_input)):
                    read_mask_len = kwargs["substitute_by_mask"][0][i].item()
                    write_mask_len = kwargs["substitute_by_mask"][1][i].item()
                    assert read_mask_len < write_mask_len
                    # if the hook called multiple times we are in model.generate()
                    # so we need to increment write_mask_len to account for the new token
                    write_mask_len += num_hook_triggered[idx]
                    new_input[i] = torch.cat(
                        [
                            new_input[i][: write_seq_len - write_mask_len, :],
                            module_activations[idx][
                                i, read_seq_len - read_mask_len :, :
                            ],
                            new_input[i][
                                write_seq_len - (write_mask_len - read_mask_len) :, :
                            ],
                        ],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_input[:, token_idx, :].shape == new_activations.shape
                new_input[:, token_idx, :] = new_activations

            num_hook_triggered[idx] += 1
            if isinstance(input, tuple):
                return (new_input,) + input[1:]
            return new_input

        def forward_hook(module, input, output):
            new_output = output[0] if isinstance(output, tuple) else output
            if "substitute_by_mask" in kwargs:
                _, read_seq_len, _ = module_activations[idx].shape
                _, write_seq_len, _ = new_output.shape
                for i in range(len(new_output)):
                    read_mask_len = kwargs["substitute_by_mask"][0][i].item()
                    write_mask_len = kwargs["substitute_by_mask"][1][i].item()
                    # same as line 87
                    write_mask_len += num_hook_triggered[idx]
                    new_output[i] = torch.cat(
                        [
                            new_output[i][: write_seq_len - write_mask_len, :],
                            module_activations[idx][
                                i, read_seq_len - read_mask_len :, :
                            ],
                            new_output[i][
                                write_seq_len - (write_mask_len - read_mask_len) :, :
                            ],
                        ],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_output[:, token_idx, :].shape == new_activations.shape
                new_output[:, token_idx, :] = new_activations

            num_hook_triggered[idx] += 1
            if isinstance(output, tuple):
                return (new_output,) + output[1:]
            return new_output

        return forward_pre_hook if sub_input_output == "input" else forward_hook

    if sub_input_output == "input":
        hook_handles = [
            modules_to_hook[i].register_forward_pre_hook(get_hook_by_idx(i))
            for i in range(len(modules_to_hook))
        ]
    elif sub_input_output == "output":
        hook_handles = [
            modules_to_hook[i].register_forward_hook(get_hook_by_idx(i))
            for i in range(len(modules_to_hook))
        ]
    else:
        assert ValueError
    if not generate:
        assert "labels" in tokenized_write
    out = _forward(
        model,
        tokenizer,
        tokenized_write,
        generate=generate,
        no_grad=generate or no_grad,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    for hook_handle in hook_handles:
        hook_handle.remove()

    return out


def latent_qa(
    batch,
    target_model,
    decoder_model,
    module_read,
    module_write,
    tokenizer,
    truncate_verbs=False,
    shift_position_ids=False,
    generate=False,
    max_new_tokens=135, # changed it to suit negotiationToM
    cache_target_model_grad=False,
    no_grad=False,
):
    tokenized_read, tokenized_write, read_lengths, write_lengths = (
        batch["tokenized_read"],
        batch["tokenized_write"],
        batch["read_lengths"],
        batch["write_lengths"],
    )
    activation_cache = _forward_cache_outputs(
        target_model,
        tokenizer,
        tokenized_read.to(target_model.device),
        module_read,
        token_idx=None,
        no_grad=(not cache_target_model_grad),
        prepare_inputs=no_op,
    )
    verb_lengths = None
    if truncate_verbs:
        activation_cache = torch.stack(activation_cache, dim=0)
        num_modules, bs, read_seq_len, _ = activation_cache.shape

        # Create a tensor with that is filled with activations for <bos> tokens
        batch_idx = torch.arange(bs, device="cpu")
        # lengths is len of read_inputs, so this fetches the <bos> token activations
        bos_activations = activation_cache[
            :, batch_idx, read_seq_len - read_lengths.cpu(), :
        ]
        bos_activations = bos_activations.unsqueeze(2).expand(-1, -1, read_seq_len, -1)
        assert bos_activations.shape == activation_cache.shape

        # Mask everything except for the non-verb (last) tokens
        verb_lengths = batch["verb_lengths"]
        counter = torch.arange(read_seq_len, device=activation_cache.device)
        # lengths - verb_lengths - 1 is total length of input not including bos token and verb

        if verb_lengths is not None:
            verb_lengths = verb_lengths.to(activation_cache.device)
            counter = counter.to(activation_cache.device)
            read_lengths = read_lengths.to(activation_cache.device)

        mask = counter.expand(bs, -1) >= read_seq_len - (
            read_lengths - verb_lengths - 1
        ).unsqueeze(1)
        mask = mask.expand(num_modules, -1, -1).unsqueeze(-1)
        # Push bos activations to precede the verb activations
        activation_cache = bos_activations * (~mask) + activation_cache * mask

        # We truncate the lengths to get rid of the verb mask
        assert read_lengths.shape == verb_lengths.shape
        read_lengths = read_lengths - verb_lengths
        activation_cache = torch.unbind(activation_cache, dim=0)
    # Compute padding
    position_ids = None
    if shift_position_ids:
        position_ids = get_pos_ids(tokenized_read, tokenized_write, verb_lengths).to(
            decoder_model.device
        )
    activation_cache = [a.to(decoder_model.device) for a in activation_cache]
    
    out = generate_substitute_layer_single(
        decoder_model,
        tokenizer,
        tokenized_write.to(decoder_model.device),
        module_write,
        activation_cache,
        "output",
        generate=generate,
        no_grad=no_grad,
        substitute_by_mask=(read_lengths, write_lengths),
        prepare_inputs=no_op,
        max_new_tokens=max_new_tokens,
        position_ids=position_ids,
        use_cache=False,
    )
    return out
