import argparse

import gradio as gr
import torch

from gradio_tabs.convert_onnx import create_onnx_app
from gradio_tabs.dataset import create_dataset_app
from gradio_tabs.inference import create_inference_app
from gradio_tabs.merge import create_merge_app
from gradio_tabs.style_vectors import create_style_vectors_app
from gradio_tabs.train import create_train_app
from style_bert_vits2.constants import GRADIO_THEME, VERSION
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers
from style_bert_vits2.utils.paths import get_paths_config


# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--autolaunch", action="store_true", default=False)
parser.add_argument("--share", action="store_true")
# parser.add_argument("--skip_default_models", action="store_true")

args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

# if not args.skip_default_models:
#     download_default_models()

paths_config = get_paths_config()
model_holder = TTSModelHolder(
    paths_config.assets_root,
    device,
    torch_device_to_onnx_providers(device),
    ignore_onnx=False,
)

with gr.Blocks(theme=GRADIO_THEME) as app:
    gr.Markdown(f"# Style-Bert-VITS2 WebUI (version {VERSION})")
    with gr.Tabs():
        with gr.Tab("音声合成"):
            create_inference_app(model_holder=model_holder)
        with gr.Tab("データセット作成"):
            create_dataset_app()
        with gr.Tab("学習"):
            create_train_app()
        with gr.Tab("スタイル作成"):
            create_style_vectors_app()
        with gr.Tab("マージ"):
            create_merge_app(model_holder=model_holder)
        with gr.Tab("ONNX変換"):
            create_onnx_app(model_holder=model_holder)

app.launch(
    server_name=args.host,
    server_port=args.port,
    inbrowser=args.autolaunch,
    share=args.share,
)
