from __future__ import annotations

from .formal_model import require_torch, torch


DEFAULT_EVALUATION_BATCH_SIZE = 256


def _validated_batch_size(batch_size: int) -> int:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("evaluation batch_size must be a positive integer")
    return batch_size


def _model_device_and_dtype(model):
    require_torch()
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise RuntimeError("cannot batch evaluation for a parameterless model") from error
    if not parameter.is_floating_point():
        raise RuntimeError("evaluation model parameters must use a floating dtype")
    return parameter.device, parameter.dtype


def _floating_tensor(model, values, name: str, dimensions: int):
    """Validate a floating bank and keep its storage on the host."""

    _device, dtype = _model_device_and_dtype(model)
    tensor = torch.as_tensor(values)
    if tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    return tensor.detach().to(device="cpu", dtype=dtype)


def _boolean_tensor(model, values, name: str, dimensions: int):
    _model_device_and_dtype(model)
    tensor = torch.as_tensor(values)
    if tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if tensor.dtype != torch.bool:
        raise ValueError(f"{name} must use boolean values")
    return tensor.detach().to(device="cpu")


def _index_tensor(values, name: str, row_count: int, upper_bound: int):
    tensor = torch.as_tensor(values)
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if tensor.ndim != 1 or tensor.shape[0] != row_count:
        raise ValueError(f"{name} must have shape [row]")
    if tensor.dtype not in integer_dtypes:
        raise ValueError(f"{name} must use integer values")
    tensor = tensor.detach().to(device="cpu", dtype=torch.long)
    if row_count and bool(torch.any((tensor < 0) | (tensor >= upper_bound)).item()):
        raise ValueError(f"{name} contains an out-of-range index")
    return tensor


def _bank_indices(
    values,
    name: str,
    row_count: int,
    bank_count: int,
):
    if values is not None:
        return _index_tensor(values, name, row_count, bank_count)
    if row_count == 0:
        return torch.empty(0, dtype=torch.long)
    if bank_count == 1:
        return torch.zeros(row_count, dtype=torch.long)
    if bank_count == row_count:
        return torch.arange(row_count, dtype=torch.long)
    raise ValueError(
        f"{name} is required unless its bank has one row or matches the input rows"
    )


def _batch_bounds(row_count: int, batch_size: int):
    for start in range(0, row_count, batch_size):
        yield start, min(start + batch_size, row_count)


def encode_context_bank(
    model,
    maps,
    radio,
    *,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    """Encode each unique map/radio context once in bounded batches."""
    batch = _validated_batch_size(batch_size)
    device, dtype = _model_device_and_dtype(model)
    map_values = _floating_tensor(model, maps, "maps", 4)
    radio_values = _floating_tensor(model, radio, "radio", 2)
    if map_values.shape[0] != radio_values.shape[0]:
        raise ValueError("maps and radio context banks must share a row count")
    if map_values.shape[0] == 0:
        token_count = int(model.map_encoder.fixed_position.shape[1]) + 1
        return torch.empty(
            (0, token_count, int(model.state_dim)),
            dtype=dtype,
            device="cpu",
        )

    encoded = []
    with torch.no_grad():
        for start, stop in _batch_bounds(map_values.shape[0], batch):
            encoded_chunk = model.encode_context(
                map_values[start:stop].to(device=device, dtype=dtype),
                radio_values[start:stop].to(device=device, dtype=dtype),
            )
            encoded.append(encoded_chunk.detach().to(device="cpu"))
    return torch.cat(encoded, dim=0)


def batched_model_states_from_context(
    model,
    visible_patches,
    masks,
    encoded_contexts,
    *,
    context_indices=None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    """Evaluate F in input order while reusing an encoded context bank."""
    batch = _validated_batch_size(batch_size)
    device, dtype = _model_device_and_dtype(model)
    visible = _floating_tensor(model, visible_patches, "visible_patches", 3)
    mask_values = _boolean_tensor(model, masks, "masks", 2)
    contexts = _floating_tensor(model, encoded_contexts, "encoded_contexts", 3)
    row_count = int(visible.shape[0])
    if visible.shape[1:] != (int(model.patch_count), int(model.patch_dim)):
        raise ValueError("visible_patches must match the model patch layout")
    if mask_values.shape != visible.shape[:2]:
        raise ValueError("masks must be boolean [row, patch]")
    if contexts.shape[2] != int(model.state_dim):
        raise ValueError("encoded_contexts must use the model state dimension")
    indices = _bank_indices(
        context_indices,
        "context_indices",
        row_count,
        int(contexts.shape[0]),
    )
    if row_count == 0:
        return torch.empty(
            (0, int(model.patch_count), int(model.state_dim)),
            dtype=dtype,
            device="cpu",
        )

    states = []
    with torch.no_grad():
        for start, stop in _batch_bounds(row_count, batch):
            state_chunk = model.state_from_context(
                visible[start:stop].to(device=device, dtype=dtype),
                mask_values[start:stop].to(device=device),
                contexts[indices[start:stop]].to(device=device, dtype=dtype),
            )
            states.append(state_chunk.detach().to(device="cpu"))
    return torch.cat(states, dim=0)


def batched_masked_states_from_context(
    model,
    patch_bank,
    masks,
    encoded_contexts,
    *,
    patch_indices=None,
    context_indices=None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    """Mask indexed patch rows and evaluate F in bounded input-order batches."""
    batch = _validated_batch_size(batch_size)
    device, dtype = _model_device_and_dtype(model)
    patches = _floating_tensor(model, patch_bank, "patch_bank", 3)
    mask_values = _boolean_tensor(model, masks, "masks", 2)
    contexts = _floating_tensor(model, encoded_contexts, "encoded_contexts", 3)
    row_count = int(mask_values.shape[0])
    if patches.shape[1:] != (int(model.patch_count), int(model.patch_dim)):
        raise ValueError("patch_bank must match the model patch layout")
    if mask_values.shape[1] != int(model.patch_count):
        raise ValueError("masks must be boolean [row, patch]")
    if contexts.shape[2] != int(model.state_dim):
        raise ValueError("encoded_contexts must use the model state dimension")
    patch_rows = _bank_indices(
        patch_indices,
        "patch_indices",
        row_count,
        int(patches.shape[0]),
    )
    context_rows = _bank_indices(
        context_indices,
        "context_indices",
        row_count,
        int(contexts.shape[0]),
    )
    if row_count == 0:
        return torch.empty(
            (0, int(model.patch_count), int(model.state_dim)),
            dtype=dtype,
            device="cpu",
        )

    states = []
    with torch.no_grad():
        for start, stop in _batch_bounds(row_count, batch):
            chunk_masks = mask_values[start:stop]
            visible = patches[patch_rows[start:stop]].clone()
            visible[chunk_masks] = 0.0
            state_chunk = model.state_from_context(
                visible.to(device=device, dtype=dtype),
                chunk_masks.to(device=device),
                contexts[context_rows[start:stop]].to(device=device, dtype=dtype),
            )
            states.append(state_chunk.detach().to(device="cpu"))
    return torch.cat(states, dim=0)


def batched_model_states(
    model,
    visible_patches,
    maps,
    radio,
    masks,
    *,
    context_indices=None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    contexts = encode_context_bank(model, maps, radio, batch_size=batch_size)
    return batched_model_states_from_context(
        model,
        visible_patches,
        masks,
        contexts,
        context_indices=context_indices,
        batch_size=batch_size,
    )


def gather_query_states(states, queries):
    """Select each row's requested patch state without changing row order."""
    require_torch()
    values = torch.as_tensor(states).detach().to(device="cpu")
    if values.ndim != 3:
        raise ValueError("states must have shape [row, patch, state]")
    query_values = _index_tensor(
        queries,
        "queries",
        int(values.shape[0]),
        int(values.shape[1]),
    )
    rows = torch.arange(values.shape[0], dtype=torch.long)
    return values[rows, query_values]


def batched_query_states_from_context(
    model,
    visible_patches,
    masks,
    queries,
    encoded_contexts,
    *,
    context_indices=None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    states = batched_model_states_from_context(
        model,
        visible_patches,
        masks,
        encoded_contexts,
        context_indices=context_indices,
        batch_size=batch_size,
    )
    return gather_query_states(states, queries)


def batched_query_states(
    model,
    visible_patches,
    maps,
    radio,
    masks,
    queries,
    *,
    context_indices=None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    contexts = encode_context_bank(model, maps, radio, batch_size=batch_size)
    return batched_query_states_from_context(
        model,
        visible_patches,
        masks,
        queries,
        contexts,
        context_indices=context_indices,
        batch_size=batch_size,
    )


def encode_action_bank(
    model,
    actions,
    *,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    """Encode each unique signed action once in bounded batches."""
    batch = _validated_batch_size(batch_size)
    device, dtype = _model_device_and_dtype(model)
    action_values = _floating_tensor(model, actions, "actions", 4)
    if action_values.shape[0] == 0:
        return torch.empty(
            (0, int(model.action_token_count) + 1, int(model.map_dim)),
            dtype=dtype,
            device="cpu",
        )

    encoded = []
    with torch.no_grad():
        for start, stop in _batch_bounds(action_values.shape[0], batch):
            encoded_chunk = model.encode_action(
                action_values[start:stop].to(device=device, dtype=dtype)
            )
            encoded.append(encoded_chunk.detach().to(device="cpu"))
    return torch.cat(encoded, dim=0)


def batched_action_predictions_from_encoded(
    model,
    states,
    encoded_actions,
    queries,
    *,
    action_indices=None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    """Predict in input order while reusing an encoded action bank."""
    batch = _validated_batch_size(batch_size)
    device, dtype = _model_device_and_dtype(model)
    state_values = _floating_tensor(model, states, "states", 3)
    actions = _floating_tensor(model, encoded_actions, "encoded_actions", 3)
    row_count = int(state_values.shape[0])
    if state_values.shape[1:] != (
        int(model.patch_count),
        int(model.state_dim),
    ):
        raise ValueError("states must match the model patch and state dimensions")
    if actions.shape[1:] != (
        int(model.action_token_count) + 1,
        int(model.map_dim),
    ):
        raise ValueError("encoded_actions must match the model action layout")
    query_values = _index_tensor(
        queries,
        "queries",
        row_count,
        int(model.patch_count),
    )
    indices = _bank_indices(
        action_indices,
        "action_indices",
        row_count,
        int(actions.shape[0]),
    )
    if row_count == 0:
        return (
            torch.empty(
                (0, int(model.latent_dim)),
                dtype=dtype,
                device="cpu",
            ),
            torch.empty(
                (0, int(model.patch_dim)),
                dtype=dtype,
                device="cpu",
            ),
        )

    latent = []
    physical = []
    with torch.no_grad():
        for start, stop in _batch_bounds(row_count, batch):
            latent_chunk, physical_chunk = model.predict_from_action(
                state_values[start:stop].to(device=device, dtype=dtype),
                actions[indices[start:stop]].to(device=device, dtype=dtype),
                query_values[start:stop].to(device=device),
            )
            latent.append(latent_chunk.detach().to(device="cpu"))
            physical.append(physical_chunk.detach().to(device="cpu"))
    return torch.cat(latent, dim=0), torch.cat(physical, dim=0)


def batched_action_predictions(
    model,
    states,
    actions,
    queries,
    *,
    action_indices=None,
    batch_size: int = DEFAULT_EVALUATION_BATCH_SIZE,
):
    encoded = encode_action_bank(model, actions, batch_size=batch_size)
    return batched_action_predictions_from_encoded(
        model,
        states,
        encoded,
        queries,
        action_indices=action_indices,
        batch_size=batch_size,
    )
