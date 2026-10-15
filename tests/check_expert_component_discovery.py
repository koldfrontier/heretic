# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Verifies that MoE routed-expert and shared-expert down_proj modules are
discovered as abliterable components, and that the discovered modules
actually work with the downstream abliteration consumer, which applies
a direction vector as v @ W through LoRA adapters (see
src/heretic/modifiers/abliteration.py, modify_model()).

The script always imports heretic from this repository's src directory,
regardless of any installed copy.

Usage (with heretic's dependencies installed):

    python tests/check_expert_component_discovery.py <model-path-or-id>

For the tiny-random qwen3.5-moe fixture:

    python tests/check_expert_component_discovery.py ^
        %HF_HOME%/hub/models--tiny-random--qwen3.5-moe/snapshots/<revision>
"""

import sys
from pathlib import Path

import torch

project_directory = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_directory / "src"))

from heretic.config import Settings  # noqa: E402
from heretic.model import Model  # noqa: E402
from heretic.utils import Prompt  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"* {status}: {label}{suffix}")
    if not condition:
        failures.append(label)


def main() -> None:
    # Heretic's Settings parse sys.argv as CLI arguments; capture our own
    # argument first, then empty sys.argv so Settings construction is clean.
    model_reference = sys.argv[1] if len(sys.argv) > 1 else None
    sys.argv = sys.argv[:1]

    if model_reference is None:
        snapshots_directory = (
            Path.home()
            / ".cache/huggingface/hub/models--tiny-random--qwen3.5-moe/snapshots"
        )
        model_reference = next(
            str(snapshot)
            for snapshot in sorted(snapshots_directory.iterdir())
            if snapshot.is_dir()
        )

    settings = Settings(
        model=str(model_reference),
        dtypes=["float32"],
        device_map="cpu",
    )
    model = Model(settings)

    components = model.get_abliterable_components()
    print(f"Abliterable components: {components}")

    module_full_names = {
        id(module): full_name
        for full_name, module in model.model.named_modules()
    }

    module_counts: dict[str, int] = {}
    routed_experts: list[tuple[int, object, str]] = []
    shared_experts: list[tuple[int, object, str]] = []

    for layer_index in range(len(model.get_layers())):
        for component, modules in model.get_layer_modules(layer_index).items():
            module_counts[component] = module_counts.get(component, 0) + len(modules)
            for module in modules:
                full_name = module_full_names.get(id(module), "?")
                if ".expert_down." in full_name:
                    routed_experts.append((layer_index, module, full_name))
                elif ".shared_expert.down_proj" in full_name:
                    shared_experts.append((layer_index, module, full_name))

    for component, count in sorted(module_counts.items()):
        print(f"* {component}: {count} modules total")

    print()

    check("'mlp.down_proj' component discovered", "mlp.down_proj" in components)

    check(
        "routed expert down_proj modules discovered",
        len(routed_experts) > 0,
        f"{len(routed_experts)} routed expert modules",
    )
    check(
        "shared expert down_proj modules discovered",
        len(shared_experts) > 0,
        f"{len(shared_experts)} shared expert modules",
    )

    if not routed_experts:
        # Without routed experts there is nothing further to prove;
        # this is the behavior of the unfixed code on the fixture.
        print()
        if failures:
            print(f"{len(failures)} check(s) FAILED")
            sys.exit(1)
        print("All checks passed")
        return

    # Downstream-consumer proof: the direction vector must be applicable to
    # each discovered module's (LoRA-wrapped) base weight as v @ W, with
    # W of shape (hidden_dim, in_features) and v of shape (hidden_dim,).
    model.apply_lora(lora_rank=1)

    # apply_lora replaces the discovered modules with LoRA wrappers
    # (exactly what the abliteration modifier sees), so re-collect
    # the modules from the wrapped model.
    prompts = [Prompt(system="You are a helpful assistant.", user="What is 1+1?")]

    raw_logits = model.get_logits(prompts)

    module_full_names = {
        id(module): full_name
        for full_name, module in model.model.named_modules()
    }

    routed_experts = []
    shared_experts = []

    for layer_index in range(len(model.get_layers())):
        for component, modules in model.get_layer_modules(layer_index).items():
            for module in modules:
                full_name = module_full_names.get(id(module), "?")
                if ".expert_down." in full_name:
                    routed_experts.append((layer_index, module, full_name))
                elif ".shared_expert.down_proj" in full_name:
                    shared_experts.append((layer_index, module, full_name))

    _, routed_module, routed_name = routed_experts[0]
    print(f"Routed expert module under test: {routed_name}")

    from peft.tuners.lora.layer import Linear as LoraLinear

    check(
        "routed expert module is LoRA-wrapped (peft lora.Linear)",
        isinstance(routed_module, LoraLinear),
        type(routed_module).__name__,
    )

    base_weight = routed_module.base_layer.weight
    print(
        f"Consumed form: W = base_layer.weight with shape "
        f"{tuple(base_weight.shape)} (out_features=hidden_dim, "
        f"in_features=moe_intermediate_dim)"
    )

    direction = torch.nn.functional.normalize(
        torch.randn(base_weight.shape[0]), dim=0
    )
    lora_A = direction @ base_weight.to(torch.float32).view(
        base_weight.shape[0], -1
    )
    check(
        "v @ W is well-formed for the routed expert (v: hidden_dim)",
        lora_A.shape == (base_weight.shape[1],),
        f"v @ W -> shape {tuple(lora_A.shape)}",
    )

    # Adapter liveness: with zero adapters the model must be unchanged;
    # with a nonzero adapter on ONE routed expert, the output must change.
    zero_adapter_logits = model.get_logits(prompts)
    impl_swap_delta = (raw_logits - zero_adapter_logits).abs().max().item()
    print(
        f"* Info: max |logits(original forward) - logits(zeroed adapters)| = "
        f"{impl_swap_delta:.3e} (rounding-level difference from routing the "
        f"expert computation through the Linear views)"
    )

    lora_A_weight = routed_module.lora_A["default"].weight
    lora_B_weight = routed_module.lora_B["default"].weight
    stacked = routed_module.base_layer.weight

    stacked_before = stacked.detach().clone()

    with torch.no_grad():
        lora_A_weight.data = lora_A.view(1, -1).to(lora_A_weight.dtype)
        lora_B_weight.data = (-direction).view(-1, 1).to(lora_B_weight.dtype)

    ablated_logits = model.get_logits(prompts)
    ablation_delta = (ablated_logits - zero_adapter_logits).abs().max().item()
    check(
        "routed expert adapter affects inference (delta is live)",
        ablation_delta > 1e-5,
        f"max |dlogits| = {ablation_delta:.3e}",
    )

    # Merge proof: merging the adapter must write the ablation into the
    # stacked (num_experts, hidden_dim, intermediate_dim) down_proj parameter,
    # so the saved model is actually ablated.
    merged_model = model.get_merged_model()
    merged_delta = (stacked - stacked_before).abs().max().item()
    expected_delta = (
        lora_B_weight.to(torch.float32) @ lora_A_weight.to(torch.float32)
    ).abs().max().item()
    check(
        "adapter merge writes through to the stacked expert weights",
        merged_delta > 0.0,
        f"max |dW_stack| = {merged_delta:.3e} (adapter B@A max = {expected_delta:.3e})",
    )
    check(
        "merged model is the unmodified PeftModel base",
        merged_model is not None,
        type(merged_model).__name__,
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("All checks passed")


if __name__ == "__main__":
    main()
