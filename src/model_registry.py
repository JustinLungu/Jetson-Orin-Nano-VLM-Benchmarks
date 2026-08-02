"""Supported native multi-frame small VLMs."""

MODEL_REPOSITORIES = {
    "smolvlm2-256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "smolvlm2-500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "phi-3.5-vision": "microsoft/Phi-3.5-vision-instruct",
}


def select_models(arguments: list[str]) -> list[str]:
    """Resolve one, several, or all selectors in registry order."""
    if not arguments:
        raise ValueError("Select at least one model")
    if "all" in arguments:
        if len(arguments) != 1:
            raise ValueError("Use 'all' alone, or provide individual model selectors")
        return list(MODEL_REPOSITORIES)

    unknown = [name for name in arguments if name not in MODEL_REPOSITORIES]
    if unknown:
        raise ValueError(f"Unknown model selector: {unknown[0]}")
    return list(dict.fromkeys(arguments))
