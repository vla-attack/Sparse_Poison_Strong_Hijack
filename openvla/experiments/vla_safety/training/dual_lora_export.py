"""Export serial dual LoRA checkpoints as ordinary OpenVLA + action-head weights."""

import json
from pathlib import Path

import torch

from experiments.vla_safety.training.dual_lora import (
    inject_serial_lora, merge_and_unload, merge_auxiliary_state, read_manifest,
)
from experiments.vla_safety.training.inference_assets import copy_inference_assets


def register_openvla():
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def export_checkpoint(checkpoint, output_dir=None, base_checkpoint=None, clean_scale=1.0,
                      trigger_scale=1.0, manifest=None, trigger_group_scales=None, calibration=None):
    # An in-progress training save passes its manifest directly; external callers
    # must go through read_manifest's INCOMPLETE check.
    manifest = read_manifest(checkpoint) if manifest is None else manifest
    checkpoint = Path(checkpoint)
    output = checkpoint if output_dir is None else Path(output_dir)
    base = manifest["base_model_path"]
    if base_checkpoint and str(base_checkpoint) != base and Path(base_checkpoint).resolve() != Path(base).resolve():
        raise ValueError("base_checkpoint differs from the recorded adapter base; refusing an incorrect merge")
    if output.resolve() == Path(base).resolve():
        raise ValueError("Export would overwrite the original base model")
    if not all(torch.isfinite(torch.tensor(value)).item() for value in (clean_scale, trigger_scale)):
        raise ValueError("Merge scales must be finite")
    if trigger_group_scales is not None:
        from experiments.vla_safety.training.grouped_lora_fusion import group_for_layer, validate_group_scales
        trigger_group_scales = validate_group_scales(trigger_group_scales)
        if clean_scale != 1.0 or trigger_scale != 1.0:
            raise ValueError("Grouped fusion fixes Clean and global Trigger scales at 1")
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForVision2Seq
    register_openvla()
    model = AutoModelForVision2Seq.from_pretrained(
        base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    output_embeddings = model.get_output_embeddings()
    continuation = manifest.get("clean_continuation")
    targets = inject_serial_lora(model, manifest["rank"], manifest["alpha"], manifest["dropout"],
                                 exclude=() if continuation else (output_embeddings,),
                                 conv_targets=continuation["conv_targets"] if continuation else (),
                                 clean_uses_trigger=manifest.get('model_config', {}).get('clean_uses_trigger', True))
    if continuation:
        from experiments.vla_safety.training.clean_lora_continuation import apply_reference
        reference = torch.load(checkpoint / "clean_reference.pt", map_location="cpu", weights_only=True)
        apply_reference({"vla": model}, {"vla": reference["vla"]})
    if targets != manifest["targets"]["vla"]:
        raise ValueError("Base model's linear targets differ from the training checkpoint")
    # load_adapters validates the full component set. Load only VLA tensors here;
    # auxiliary modules are merged directly in their canonical state-dict format.
    with torch.no_grad():
        expected_components = set(manifest["targets"])
        for branch in ("clean", "trigger"):
            state = torch.load(checkpoint / "dual_lora" / f"{branch}.pt", map_location="cpu", weights_only=True)
            if set(state) != expected_components:
                raise ValueError("Adapter components differ from the manifest")
            parameters = dict(model.named_parameters())
            expected = {f"{target}.{branch}{suffix}" for target in targets for suffix in ("_A", "_B")}
            if set(state["vla"]) != expected:
                raise ValueError("VLA adapter target mismatch")
            for name, value in state["vla"].items():
                if parameters[name].shape != value.shape:
                    raise ValueError(f"VLA adapter shape mismatch: {name}")
                parameters[name].copy_(value)
    scales = ({name: trigger_group_scales[group_for_layer("vla", name)] for name in targets}
              if trigger_group_scales is not None else None)
    model = merge_and_unload(model, clean_scale, trigger_scale, trigger_scales=scales)
    model.save_pretrained(output)
    del model
    bases = torch.load(checkpoint / "auxiliary_base.pt", map_location="cpu", weights_only=True)
    for component, base_state in bases.items():
        scales = ({name: trigger_group_scales[group_for_layer(component, name)]
                   for name in manifest["targets"][component]} if trigger_group_scales is not None else None)
        state = merge_auxiliary_state(base_state, component, manifest, checkpoint / "dual_lora",
                                      clean_scale, trigger_scale, trigger_scales=scales)
        torch.save(state, output / f"{component}--{manifest['step']}_checkpoint.pt")
    copy_inference_assets(checkpoint, output)
    report = {
        "format": manifest["format"], "base_model_path": base, "step": manifest["step"],
        "adapter_checkpoint": str(checkpoint.resolve()),
        "model_config": manifest.get("model_config", {}),
        "clean_scale": clean_scale, "trigger_scale": trigger_scale,
        "formula": "W'=(I+trigger_scale*T)(W+clean_scale*C); b'=(I+trigger_scale*T)b",
        "data_sampling": manifest["data_sampling"],
        "note": "Both LoRA deltas use scale 1 by default. Data fractions are not merge scales.",
    }
    if continuation:
        report["clean_continuation"] = continuation
        report["clean_reference_sha256"] = manifest["clean_reference_sha256"]
        report["formula"] = "W'=(I+trigger_scale*T)(W150k-C150k+clean_scale*Ccurrent); b'=(I+trigger_scale*T)b"
    if trigger_group_scales is not None:
        report["trigger_group_scales"] = trigger_group_scales
        report["calibration"] = calibration
        report["formula"] = "W_l'=(I+beta_group(l)*T_l)(W150k_l-C150k_l+Ccurrent_l); b_l'=(I+beta_group(l)*T_l)b_l"
        report["note"] = "Clean scale is 1; each Trigger group uses its recorded beta, not the global placeholder 1"
    if 'language_policy' in manifest:
        report['language_policy'] = manifest['language_policy']
        report['language_policy_sha256'] = manifest['language_policy_sha256']
        report['training_objective'] = manifest['training_objective']
    if 'trigger_objective' in manifest:
        report['trigger_objective'] = manifest['trigger_objective']
    if 'gradient_surgery' in manifest:
        report['gradient_surgery']=manifest['gradient_surgery']
    if 'sampling_policy' in manifest:
        report['sampling_policy']=manifest['sampling_policy']
    if 'training_forward' in manifest:
        report['training_forward'] = manifest['training_forward']
        report['inference_forward'] = 'One fully merged serial model for both Clean and Trigger evaluation; no source routing'
    (output / "dual_lora_merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return output
