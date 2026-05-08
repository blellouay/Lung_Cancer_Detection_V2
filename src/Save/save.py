import os
import json
import torch
from torchview import draw_graph


DEFAULT_CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]

DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]


def save_evaluation_results(
    results: dict,
    model,
    save_dir: str,
    config: dict = None,
    filename: str = "metrics.json",
    input_size=(1, 3, 224, 224)
):
    os.makedirs(save_dir, exist_ok=True)
    config = dict(config or {})
    config["class_names"] = config.get("class_names") or DEFAULT_CLASS_NAMES
    config["normalize_mean"] = config.get("normalize_mean") or DEFAULT_NORMALIZE_MEAN
    config["normalize_std"] = config.get("normalize_std") or DEFAULT_NORMALIZE_STD
    config["image_size"] = config.get("image_size") or input_size[2]

    # =========================
    # Save model weights
    # =========================
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # =========================
    # Parameter counts
    # =========================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # =========================
    # Architecture image
    # =========================
    architecture_filename = "architecture"
    architecture_image_path = None

    try:
        graph = draw_graph(
            model,
            input_size=input_size,
            expand_nested=True,
            graph_dir="TB",
            save_graph=False
        )

        architecture_image_path = os.path.join(
            save_dir,
            architecture_filename + ".svg"
        )

        graph.visual_graph.render(
            filename=architecture_filename,
            directory=save_dir,
            format="svg",
            cleanup=True
        )

    except Exception as e:
        print(f"Architecture visualization skipped: {e}")

    # =========================
    # Layer breakdown
    # =========================
    def extract_layers(model):
        layers = []

        for name, module in model.named_modules():
            if name == "":
                continue

            params = sum(
                p.numel() for p in module.parameters(recurse=False)
            )

            trainable = sum(
                p.numel()
                for p in module.parameters(recurse=False)
                if p.requires_grad
            )

            layers.append({
                "name": name,
                "type": module.__class__.__name__,
                "parameters": params,
                "trainable_parameters": trainable,
                "config": repr(module)
            })

        return layers

    # =========================
    # Input/output shape probing
    # =========================
    def probe_shapes(model, input_size):
        try:
            device = next(model.parameters()).device
            dummy = torch.zeros(input_size).to(device)

            shape_map = {}
            hooks = []

            def make_hook(name):
                def hook(module, inp, out):
                    in_shape = list(inp[0].shape) if inp else None

                    if hasattr(out, "shape"):
                        out_shape = list(out.shape)
                    elif isinstance(out, (list, tuple)) and len(out) > 0:
                        out_shape = (
                            list(out[0].shape)
                            if hasattr(out[0], "shape")
                            else None
                        )
                    else:
                        out_shape = None

                    shape_map[name] = {
                        "input_shape": in_shape,
                        "output_shape": out_shape
                    }

                return hook

            for name, module in model.named_modules():
                if name == "":
                    continue

                hooks.append(
                    module.register_forward_hook(make_hook(name))
                )

            model.eval()

            with torch.no_grad():
                model(dummy)

            for h in hooks:
                h.remove()

            return shape_map

        except Exception as e:
            print(f"Shape probing skipped: {e}")
            return {}

    layers = extract_layers(model)
    shape_map = probe_shapes(model, input_size)

    for layer in layers:
        shapes = shape_map.get(layer["name"], {})
        layer["input_shape"] = shapes.get("input_shape")
        layer["output_shape"] = shapes.get("output_shape")

    # =========================
    # Model info
    # =========================
    model_info = {
        "model_name": model.__class__.__name__,
        "architecture": str(model),
        "architecture_image": architecture_image_path,
        "architecture_image_format": "svg",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_size": list(input_size),
        "layers": layers
    }

    # =========================
    # Deployment info
    # =========================
    deployment_info = {
        "model_path": model_path,

        "input_size": [
            input_size[1],
            input_size[2],
            input_size[3]
        ],

        "image_size": config["image_size"],
        "class_names": config["class_names"],
        "normalize_mean": config["normalize_mean"],
        "normalize_std": config["normalize_std"],

        "deployment_ready": True,

        "onnx_exported": False,

        "gradcam_available": (
            "gradcam_paths" in results
            and results["gradcam_paths"] is not None
        ),

        "medical_thresholds": {
            "minimum_recall": 0.85,
            "current_recall": results.get("recall_macro"),
            "passes_recall_threshold": (
                results.get("recall_macro", 0) >= 0.85
            )
        }
    }

    # =========================
    # Final JSON object
    # =========================
    full_results = {
        "model_info": model_info,
        "evaluation_metrics": results,
        "deployment_info": deployment_info
    }

    full_results["training_config"] = config

    # =========================
    # Save JSON
    # =========================
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "w") as f:
        json.dump(full_results, f, indent=4)

    print(f"Evaluation results saved to: {save_path}")
    print(f"Model weights saved to: {model_path}")
    print(f"Architecture image saved to: {architecture_image_path}")
