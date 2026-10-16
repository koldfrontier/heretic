# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Verifies that the attention block's output projection - the Linear that writes
the attention output back to the residual stream - is discovered as an
abliterable component under every naming scheme a model may use for it.

The motivating case is a hybrid architecture whose full attention layers spell
it `self_attn.o_proj` while its linear attention layers spell it
`linear_attn.out_proj` (Qwen3.5 MoE, Qwen4-Exp). Both write to the residual
stream, which is what abliteration subtracts from, so both must be discovered
and both are reported under the single component name `attn.o_proj` (see
get_layer_modules in src/heretic/model.py).

The script always imports heretic from this repository's src directory,
regardless of any installed copy.

Usage (with heretic's dependencies installed):

    python tests/check_attention_out_proj_discovery.py [hybrid-model-dir] [dense-model-dir]

Both models are built on the meta device from their config.json alone, so no
weight files and no network access are required. The default paths are the
local Hugging Face cache snapshots:

    %HF_HOME%/hub/models--ukisai--Swift1.5-Qwen3.8-Flash-Next/snapshots/<revision>
    %HF_HOME%/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision>

Section 3 additionally exercises naming schemes with synthetic decoder layers
that need no model files at all.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import Linear, Module, ModuleList

project_directory = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_directory / "src"))

from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from heretic.model import Model  # noqa: E402

HOME = Path.home()

# Component counts recorded on master (1b27afc) with the same meta-device
# builds. They must not change: the attention-output discovery was generalized,
# not altered.
EXPECTED_DENSE_COUNTS = {"attn.o_proj": 28, "mlp.down_proj": 28}
EXPECTED_HYBRID_COUNTS = {"attn.o_proj": 48, "mlp.down_proj": 24624}

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix: str = f" ({detail})" if detail else ""
    print(f"* {status}: {label}{suffix}")
    if not condition:
        failures.append(label)


def build_on_meta(model_directory: Path):
    """Builds a model from a local config.json on the meta device."""
    config = AutoConfig.from_pretrained(model_directory, local_files_only=True)

    # Same choice heretic's get_model_class makes: a vision_config means the
    # checkpoint is multimodal and its language model lives under
    # model.language_model.
    model_class = (
        AutoModelForImageTextToText
        if getattr(config, "vision_config", None) is not None
        else AutoModelForCausalLM
    )

    with torch.device("meta"):
        model = model_class.from_config(config)

    return config, model


def discover(hf_model) -> tuple[dict[str, int], dict[str, list[str]], Model]:
    """
    Runs the discovery for every layer and returns the per-component module
    counts, the per-layer list of `component <- module full name` entries, and
    the Model wrapper used for the discovery.
    """
    model = Model.__new__(Model)
    model.model = hf_model

    counts: dict[str, int] = {}
    per_layer: dict[str, list[str]] = {}

    for layer_index in range(len(model.get_layers())):
        layer_modules = model.get_layer_modules(layer_index)

        # The name map is built after the layer's discovery, because the MoE
        # discovery registers new modules (one Linear view per routed expert).
        names = {id(module): name for name, module in hf_model.named_modules()}

        entries: list[str] = []
        for component, modules in sorted(layer_modules.items()):
            counts[component] = counts.get(component, 0) + len(modules)
            for module in modules:
                entries.append(f"{component} <- {names.get(id(module), '?')}")
        per_layer[str(layer_index)] = sorted(entries)

    return counts, per_layer, model


def attention_modules(per_layer: dict[str, list[str]]) -> list[str]:
    return sorted(
        entry
        for entries in per_layer.values()
        for entry in entries
        if entry.startswith("attn.o_proj <- ")
    )


def layer_path(full_name: str) -> str:
    """
    Strips the wrapper path: multimodal checkpoints nest the language model
    under model.language_model, text-only ones directly under model, and only
    the layer path is what the discovery is about.
    """
    marker = "layers."
    return full_name[full_name.index(marker) :] if marker in full_name else full_name


def check_hybrid_model(model_directory: Path) -> None:
    print(f"Hybrid model (config only, meta device): {model_directory}")

    if not (model_directory / "config.json").is_file():
        check(
            "hybrid model config.json found",
            False,
            f"{model_directory} does not contain config.json; "
            f"pass the snapshot directory as the first argument",
        )
        return

    config, hf_model = build_on_meta(model_directory)
    counts, per_layer, model = discover(hf_model)

    text_config = config.get_text_config()
    layer_types = list(text_config.layer_types)
    linear_layers = [
        index for index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]

    print(f"  * {len(layer_types)} layers, {len(linear_layers)} with linear attention")
    print(f"  * component counts: {counts}")

    attention = attention_modules(per_layer)
    attention_paths = [layer_path(entry.split(" <- ", 1)[1]) for entry in attention]

    print(
        f"  * example: {attention_paths[0] if attention_paths else '(none)'}"
        f" on a {layer_types[0]} layer"
    )

    check(
        "hybrid model has both attention naming schemes",
        0 < len(linear_layers) < len(layer_types),
        f"{len(layer_types) - len(linear_layers)} full/indexed attention layers, "
        f"{len(linear_layers)} linear attention layers",
    )

    check(
        "every layer's attention output projection is discovered",
        len(attention) == len(layer_types),
        f"{len(attention)} attn.o_proj modules for {len(layer_types)} layers",
    )

    expected = [
        f"layers.{index}."
        + (
            "linear_attn.out_proj"
            if layer_type == "linear_attention"
            else "self_attn.o_proj"
        )
        for index, layer_type in enumerate(layer_types)
    ]

    check(
        "each layer's projection is the one its layer type declares",
        sorted(attention_paths) == sorted(expected),
        f"{sum(1 for path in attention_paths if 'linear_attn.out_proj' in path)} "
        f"from linear_attn.out_proj, "
        f"{sum(1 for path in attention_paths if 'self_attn.o_proj' in path)} "
        f"from self_attn.o_proj",
    )

    check(
        "both schemes are reported under the same component name",
        set(counts) == {"attn.o_proj", "mlp.down_proj"},
        f"components: {sorted(counts)}",
    )

    check(
        "component counts match master (no regression)",
        counts == EXPECTED_HYBRID_COUNTS,
        f"{counts}",
    )

    check(
        "the residual stream width is available for the shape-based fallback",
        model.get_hidden_size() == text_config.hidden_size,
        f"hidden_size={model.get_hidden_size()}",
    )

    # Downstream consumer: apply_lora() must wrap every discovered attention
    # output projection, so that abliteration (lora_A = v @ W,
    # lora_B = -weight * v) actually reaches both naming schemes.
    model.apply_lora(lora_rank=1)

    wrapped: dict[str, int] = {}
    names = {id(module): name for name, module in model.model.named_modules()}
    for layer_index in range(len(model.get_layers())):
        for component, modules in model.get_layer_modules(layer_index).items():
            if component != "attn.o_proj":
                continue
            for module in modules:
                if hasattr(module, "lora_A"):
                    leaf = names.get(id(module), "?").rsplit(".", 1)[-1]
                    wrapped[leaf] = wrapped.get(leaf, 0) + 1

    check(
        "apply_lora wraps every attention output projection",
        wrapped.get("out_proj", 0) == len(linear_layers)
        and wrapped.get("o_proj", 0) == len(layer_types) - len(linear_layers),
        f"LoRA-wrapped: {wrapped}",
    )
    print()


def check_dense_model(model_directory: Path) -> None:
    print(f"Dense model (config only, meta device): {model_directory}")

    if not (model_directory / "config.json").is_file():
        check(
            "dense model config.json found",
            False,
            f"{model_directory} does not contain config.json; "
            f"pass the snapshot directory as the second argument",
        )
        return

    _, hf_model = build_on_meta(model_directory)
    counts, per_layer, model = discover(hf_model)

    print(f"  * component counts: {counts}")

    attention = attention_modules(per_layer)
    attention_paths = [layer_path(entry.split(" <- ", 1)[1]) for entry in attention]

    print(
        f"  * example: {attention_paths[0] if attention_paths else '(none)'}"
    )

    check(
        "dense discovery is unchanged",
        counts == EXPECTED_DENSE_COUNTS,
        f"{counts}",
    )

    check(
        "every dense layer's self_attn.o_proj is discovered",
        sorted(attention_paths)
        == sorted(
            f"layers.{index}.self_attn.o_proj"
            for index in range(len(model.get_layers()))
        ),
        f"{len(attention)} modules for {len(model.get_layers())} layers",
    )
    print()


def synthetic_layer(
    block_name: str,
    block_children: dict[str, Linear],
) -> Module:
    """A decoder layer with one attention/mixer block and an MLP."""
    layer = Module()

    block = Module()
    for child_name, child in block_children.items():
        setattr(block, child_name, child)
    setattr(layer, block_name, block)

    mlp = Module()
    mlp.down_proj = Linear(16, 32, bias=False)
    layer.mlp = mlp

    return layer


def discovery_for_layer(layer: Module, hidden_size: int | None) -> dict[str, list[Module]]:
    stub = Module()
    stub.config = SimpleNamespace(hidden_size=hidden_size)
    stub.model = Module()
    stub.model.layers = ModuleList([layer])

    model = Model.__new__(Model)
    model.model = stub

    return model.get_layer_modules(0)


def check_naming_schemes() -> None:
    print("Naming schemes (synthetic layers, no model files)")

    hidden_size = 64
    in_features = 16

    # (block attribute, output projection attribute). The first four are the
    # spellings Heretic already supported before this check was added; the rest
    # are schemes that were not discovered before the generalization.
    discovered_schemes = [
        ("self_attn", "o_proj"),
        ("self_attn", "out_proj"),
        ("linear_attn", "out_proj"),
        ("conv", "out_proj"),
        ("self_attention", "dense"),
        ("attention", "output_linear"),
        ("attn", "c_proj"),
        ("mixer", "out_proj"),
    ]

    for block_name, out_name in discovered_schemes:
        out_proj = Linear(in_features, hidden_size, bias=False)
        layer = synthetic_layer(block_name, {out_name: out_proj})

        modules = discovery_for_layer(layer, hidden_size)

        check(
            f"{block_name}.{out_name} is discovered as attn.o_proj",
            modules.get("attn.o_proj") == [out_proj],
            f"{len(modules.get('attn.o_proj', []))} module(s)",
        )

    # Shape-based fallback: a scheme whose output projection has a name no
    # architecture list can know, recognized by writing to the residual stream.
    out_proj = Linear(in_features, hidden_size, bias=False)
    layer = synthetic_layer("attention", {"proj_out": out_proj})
    modules = discovery_for_layer(layer, hidden_size)
    check(
        "an unknown output projection name is found by shape",
        modules.get("attn.o_proj") == [out_proj],
        f"{len(modules.get('attn.o_proj', []))} module(s)",
    )

    # Negative control: an input-side projection whose width happens to equal
    # the residual stream width must not be mistaken for the output projection.
    layer = synthetic_layer(
        "attention",
        {"q_proj": Linear(in_features, hidden_size, bias=False)},
    )
    modules = discovery_for_layer(layer, hidden_size)
    check(
        "an input-side projection is not discovered by shape",
        "attn.o_proj" not in modules,
        f"components: {sorted(modules)}",
    )

    # Negative control: two equally plausible shape matches are ambiguous, and
    # ambiguity is not resolved by guessing.
    layer = synthetic_layer(
        "attention",
        {
            "proj_out": Linear(in_features, hidden_size, bias=False),
            "proj_mix": Linear(in_features, hidden_size, bias=False),
        },
    )
    modules = discovery_for_layer(layer, hidden_size)
    check(
        "an ambiguous shape match is not discovered",
        "attn.o_proj" not in modules,
        f"components: {sorted(modules)}",
    )

    # Negative control: without a known residual stream width, the shape-based
    # fallback must stay out of the way.
    layer = synthetic_layer(
        "attention",
        {"proj_out": Linear(in_features, hidden_size, bias=False)},
    )
    modules = discovery_for_layer(layer, None)
    check(
        "the shape fallback requires a known hidden size",
        "attn.o_proj" not in modules,
        f"components: {sorted(modules)}",
    )

    print()


def main() -> None:
    hybrid_model = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else HOME
        / ".cache/huggingface/hub/models--ukisai--Swift1.5-Qwen3.8-Flash-Next"
        / "snapshots/0bd4fe22431372cdad1979267d3ab45aa7e6150a"
    )
    dense_model = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else HOME
        / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B"
        / "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    )

    check_hybrid_model(hybrid_model)
    check_dense_model(dense_model)
    check_naming_schemes()

    if failures:
        print(f"{len(failures)} check(s) FAILED")
        sys.exit(1)

    print("All checks passed")


if __name__ == "__main__":
    main()
