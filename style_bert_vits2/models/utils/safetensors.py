import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from style_bert_vits2.logging import logger


def _build_partially_loadable_embedding_tensor(
    parameter_key: str,
    checkpoint_tensor: torch.Tensor,
    current_tensor: torch.Tensor,
    allow_partial_load_embedding_keys: tuple[str, ...],
) -> torch.Tensor | None:
    """
    明示 opt-in された embedding key のみ、語彙拡張時の prefix 行を部分ロードする。

    Args:
        parameter_key (str): state_dict のキー
        checkpoint_tensor (torch.Tensor): checkpoint 側の tensor
        current_tensor (torch.Tensor): 現在 model 側の tensor
        allow_partial_load_embedding_keys (tuple[str, ...]): 部分ロードを許可する key suffix

    Returns:
        torch.Tensor | None: 部分ロード後の tensor。適用できない場合は None
    """

    if not any(
        parameter_key.endswith(suffix) for suffix in allow_partial_load_embedding_keys
    ):
        return None
    if checkpoint_tensor.ndim != current_tensor.ndim or checkpoint_tensor.ndim < 2:
        return None
    if checkpoint_tensor.shape[1:] != current_tensor.shape[1:]:
        return None
    if current_tensor.shape[0] < checkpoint_tensor.shape[0]:
        return None

    shared_rows = min(checkpoint_tensor.shape[0], current_tensor.shape[0])
    if shared_rows <= 0:
        return None

    merged_tensor = current_tensor.detach().clone()
    merged_tensor[:shared_rows] = checkpoint_tensor[:shared_rows].to(
        device=current_tensor.device,
        dtype=current_tensor.dtype,
    )
    logger.warning(
        f"Partially loaded embedding tensor. key: {parameter_key}, checkpoint_shape: {tuple(checkpoint_tensor.shape)}, model_shape: {tuple(current_tensor.shape)}, copied_rows: {shared_rows}"
    )
    return merged_tensor


def load_safetensors(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    for_infer: bool = False,
    device: str | torch.device = "cpu",
    allow_partial_load_embedding_keys: tuple[str, ...] = (),
) -> tuple[torch.nn.Module, int | None]:
    """
    指定されたパスから safetensors モデルを読み込み、モデルとイテレーションを返す。

    Args:
        checkpoint_path (str | Path): モデルのチェックポイントファイルのパス
        model (torch.nn.Module): 読み込む対象のモデル
        for_infer (bool): 推論用に読み込むかどうかのフラグ
        allow_partial_load_embedding_keys (tuple[str, ...]): 語彙拡張時に prefix 行の部分ロードを許可するための embedding key suffix

    Returns:
        tuple[torch.nn.Module, int | None]: 読み込まれたモデルとイテレーション回数（存在する場合）
    """

    tensors: dict[str, Any] = {}
    iteration: int | None = None
    with safe_open(str(checkpoint_path), framework="pt", device=device) as f:  # type: ignore
        for key in f.keys():
            if key == "iteration":
                iteration = f.get_tensor(key).item()
            tensors[key] = f.get_tensor(key)

    if hasattr(model, "module"):
        current_state_dict = model.module.state_dict()  # type: ignore
    else:
        current_state_dict = model.state_dict()

    new_state_dict: dict[str, torch.Tensor] = dict(tensors)
    for key, current_tensor in current_state_dict.items():
        if key not in tensors:
            continue

        if tensors[key].shape == current_tensor.shape:
            continue

        partially_loadable_tensor = _build_partially_loadable_embedding_tensor(
            parameter_key=key,
            checkpoint_tensor=tensors[key],
            current_tensor=current_tensor,
            allow_partial_load_embedding_keys=allow_partial_load_embedding_keys,
        )
        if partially_loadable_tensor is not None:
            new_state_dict[key] = partially_loadable_tensor
            continue

        if key.startswith("enc_q") and for_infer:
            new_state_dict.pop(key, None)
            continue

    if hasattr(model, "module"):
        result = model.module.load_state_dict(new_state_dict, strict=False)  # type: ignore
    else:
        result = model.load_state_dict(new_state_dict, strict=False)

    for key in result.missing_keys:
        if key.startswith("enc_q") and for_infer:
            continue
        logger.warning(f"Missing key: {key}")
    for key in result.unexpected_keys:
        if key == "iteration":
            continue
        logger.warning(f"Unexpected key: {key}")
    if iteration is None:
        logger.info(f"Loaded '{checkpoint_path}'")
    else:
        logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, iteration


def save_safetensors(
    model: torch.nn.Module,
    iteration: int,
    checkpoint_path: str | Path,
    is_half: bool = False,
    for_infer: bool = False,
) -> None:
    """
    モデルを safetensors 形式で保存する。

    Args:
        model (torch.nn.Module): 保存するモデル
        iteration (int): イテレーション回数
        checkpoint_path (str | Path): 保存先のパス
        is_half (bool): モデルを半精度で保存するかどうかのフラグ
        for_infer (bool): 推論用に保存するかどうかのフラグ
    """

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()  # type: ignore
    else:
        state_dict = model.state_dict()
    keys = []
    # 推論時に不要かつ pc_variance_monitor_k に依存して形状が変わるバッファ群
    ## これらは SpeakerAdapter の分布診断専用で、推論経路では参照されない
    ## また学習側 pc_variance_monitor_k と推論側既定値 (8) が異なると形状不一致でロード失敗するため、
    ## 推論用 safetensors からは確実に除外する
    speaker_adapter_diagnostic_buffers = {
        "emb_g_mean",
        "emb_g_var",
        "emb_g_pca_components",
        "emb_g_pca_variances",
        "emb_g_pca_k_actual",
        "emb_g_statistics_initialized",
    }
    for k in state_dict:
        if "enc_q" in k and for_infer:
            continue
        # state_dict key に prefix (例: net_g.emb_g_mean) が付く場合も診断バッファとして除外する (完全一致だと漏れる)
        if for_infer:
            state_dict_key_tail = k.rsplit(".", 1)[-1]
            if state_dict_key_tail in speaker_adapter_diagnostic_buffers:
                continue
        keys.append(k)

    new_dict = (
        {k: state_dict[k].half() for k in keys}
        if is_half
        else {k: state_dict[k] for k in keys}
    )
    new_dict["iteration"] = torch.LongTensor([iteration])
    logger.info(f"Saved safetensors to {checkpoint_path}")

    # 一時ファイルへ書き終えてから差し替え、書き込み途中のファイルを正規の名前で読まれないようにする
    temporary_path = f"{checkpoint_path}.tmp"
    save_file(new_dict, temporary_path)
    os.replace(temporary_path, checkpoint_path)
