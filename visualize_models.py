"""Visualize model architectures for all three configs."""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchinfo import summary
from torchview import draw_graph

CONFIG_DIR = str(Path(__file__).parent / "config" / "train")
OUTPUT_DIR = Path(__file__).parent / "figures"

MODELS = {
    "lewm": "ViT-Tiny + ARPredictor",
    "gru_predictor": "MobileNet + GRUPredictor",
    "mobilenet_encoder": "MobileNet + ARPredictor",
}


def visualize(model_name, label):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="lewm", overrides=[
            f"model={model_name}",
            "model.action_encoder.input_dim=2",
        ])

    model = instantiate(cfg.model)
    model.eval()

    print("=" * 70)
    print(f"  {label}  (config: model/{model_name}.yaml)")
    print("=" * 70)

    B, T, C, H, W = 1, cfg.history_size, 3, cfg.img_size, cfg.img_size
    action_dim = 2

    pixels = torch.randn(B, T, C, H, W)
    actions = torch.randn(B, T, action_dim)

    info = {"pixels": pixels, "action": actions}
    info = model.encode(info)
    emb, act_emb = info["emb"], info["act_emb"]

    print("\n-- Full JEPA --")
    summary(model, depth=3, col_names=["num_params", "params_percent", "kernel_size"])

    print("\n-- Predictor --")
    summary(model.predictor, input_data=(emb, act_emb), depth=2,
            col_names=["num_params", "params_percent", "output_size"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        draw_graph(
            model.predictor,
            input_data=(emb, act_emb),
            expand_nested=True,
            save_graph=True,
            filename=f"{model_name}_predictor",
            directory=str(OUTPUT_DIR),
        )
        print(f"\n  -> Predictor graph saved to figures/{model_name}_predictor.png")
    except Exception as e:
        print(f"\n  -> Could not render predictor graph: {e}")

    print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = list(MODELS.keys())

    for name in targets:
        label = MODELS.get(name, name)
        visualize(name, label)
