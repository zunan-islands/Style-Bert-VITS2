import glob
import os
import re
from pathlib import Path
from typing import Any, overload

import torch

from style_bert_vits2.logging import logger


@overload
def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    skip_optimizer: bool = False,
    for_infer: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, torch.optim.Optimizer, float, int]: ...


@overload
def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: None = None,
    skip_optimizer: bool = False,
    for_infer: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, None, float, int]: ...


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    skip_optimizer: bool = False,
    for_infer: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, float, int]:
    """
    指定されたパスからチェックポイントを読み込み、モデルとオプティマイザーを更新する。

    Args:
        checkpoint_path (str | Path): チェックポイントファイルのパス
        model (torch.nn.Module): 更新するモデル
        optimizer (torch.optim.Optimizer | None): 更新するオプティマイザー。None の場合は更新しない
        skip_optimizer (bool): オプティマイザーの更新をスキップするかどうかのフラグ
        for_infer (bool): 推論用に読み込むかどうかのフラグ

    Returns:
        tuple[torch.nn.Module, torch.optim.Optimizer | None, float, int]: 更新されたモデルとオプティマイザー、学習率、イテレーション回数
    """

    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    logger.info(
        f"Loading model and optimizer at iteration {iteration} from {checkpoint_path}"
    )
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()  # type: ignore
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)  # type: ignore

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()  # type: ignore
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            if k not in saved_state_dict:
                raise KeyError(k)
            if saved_state_dict[k].shape != v.shape:
                raise ValueError((saved_state_dict[k].shape, v.shape))
            new_state_dict[k] = saved_state_dict[k]
        except Exception:
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warning(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            elif "enc_q" in k and for_infer:
                continue
            else:
                logger.error(f"{k} is not in the checkpoint {checkpoint_path}")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)  # type: ignore
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(f"Loaded '{checkpoint_path}' (iteration {iteration})")

    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | torch.optim.AdamW,
    learning_rate: float,
    iteration: int,
    checkpoint_path: str | Path,
) -> None:
    """
    モデルとオプティマイザーの状態を指定されたパスに保存する。

    Args:
        model (torch.nn.Module): 保存するモデル
        optimizer (torch.optim.Optimizer | torch.optim.AdamW): 保存するオプティマイザー
        learning_rate (float): 学習率
        iteration (int): イテレーション回数
        checkpoint_path (str | Path): 保存先のパス
    """
    logger.info(
        f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()  # type: ignore
    else:
        state_dict = model.state_dict()
    save_torch_atomically(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def save_torch_atomically(obj: Any, path: str | Path) -> None:
    """
    一時ファイルへ書き終えてから差し替えて保存する。
    書き込みの途中で学習が止まっても、壊れたチェックポイントが正規の名前で残らないようにするため。

    Args:
        obj (Any): 保存するオブジェクト
        path (str | Path): 保存先のパス
    """

    temporary_path = f"{path}.tmp"
    torch.save(obj, temporary_path)
    os.replace(temporary_path, path)


def save_resume_marker(
    model_dir_path: str | Path, global_step: int, state: dict[str, Any]
) -> None:
    """
    そのステップの再開用チェックポイント一式 (G・D・DUR・WD・EMA) を書き終えた印と、続きから学習するための情報を保存する。
    一式の保存の最後に呼ぶことで、印のあるステップは全ての部品が揃っていることを保証する。

    Args:
        model_dir_path (str | Path): チェックポイントの保存先
        global_step (int): チェックポイントのファイル名に使ったステップ数
        state (dict[str, Any]): 次に学習するステップ数・エポック・エポック内で済んだバッチ数・乱数の状態
    """

    save_torch_atomically(
        state, os.path.join(str(model_dir_path), f"RESUME_{global_step}.pth")
    )


def load_resume_state(
    model_dir_path: str | Path,
    components: list[tuple[str, torch.nn.Module, torch.optim.Optimizer | None]],
    skip_optimizer: bool = False,
) -> tuple[int, int, dict[str, float], dict[str, Any] | None]:
    """
    再開に使うチェックポイント一式を、揃っていて読める最新のステップから読み込む。
    再開の印 (RESUME_*.pth) があるステップを優先し、印の無い以前の学習は全部品が揃った最新のステップを選ぶ。
    読み込みに失敗したステップは飛ばして1つ前へ戻り、どのステップも読めなければ例外を出す (ゼロからの学習へ黙って落ちない)。

    Args:
        model_dir_path (str | Path): チェックポイントの保存先
        components (list[tuple[str, torch.nn.Module, torch.optim.Optimizer | None]]): (ファイル名の接頭辞, モデル, オプティマイザー) の並び。先頭は G
        skip_optimizer (bool): オプティマイザーの状態を読まないか

    Returns:
        tuple[int, int, dict[str, float], dict[str, Any] | None]: (ステップ, エポック, 接頭辞ごとの学習率, 再開の印の中身。印が無ければ None)

    Raises:
        RuntimeError: 読み込めるステップが1つも無い場合
    """

    def steps_of(prefix: str) -> set[int]:
        steps = set()
        for path in glob.glob(os.path.join(str(model_dir_path), f"{prefix}_*.pth")):
            match = re.fullmatch(rf"{prefix}_(\d+)\.pth", os.path.basename(path))
            if match is not None:
                steps.add(int(match.group(1)))
        return steps

    prefixes = [prefix for prefix, _, _ in components]
    complete_steps = set.intersection(*(steps_of(prefix) for prefix in prefixes))
    marked_steps = complete_steps & steps_of("RESUME")
    # 印のあるステップを新しい順に試し、その後で印の無い (以前の版で保存した) ステップを新しい順に試す
    candidates = sorted(marked_steps, reverse=True) + sorted(
        complete_steps - marked_steps, reverse=True
    )
    errors: list[str] = []
    for step in candidates:
        try:
            marker_path = os.path.join(str(model_dir_path), f"RESUME_{step}.pth")
            marker = (
                torch.load(marker_path, map_location="cpu", weights_only=False)
                if step in marked_steps
                else None
            )
            epoch = 0
            learning_rates: dict[str, float] = {}
            for prefix, model, optimizer in components:
                _, _, learning_rate, epoch_of_component = load_checkpoint(
                    os.path.join(str(model_dir_path), f"{prefix}_{step}.pth"),
                    model,
                    optimizer,
                    skip_optimizer=skip_optimizer,
                )
                learning_rates[prefix] = learning_rate
                if prefix == prefixes[0]:
                    epoch = epoch_of_component
            return step, epoch, learning_rates, marker
        except Exception as ex:
            # 書き込みの途中で止まった版などで読めないステップは飛ばし、1つ前の一式から再開する
            logger.warning(
                f"Failed to load the checkpoints of step {step}, trying an older step: {ex}"
            )
            errors.append(f"step {step}: {ex}")
    raise RuntimeError(
        f"No loadable checkpoint set ({', '.join(prefixes)}) was found in {model_dir_path}. "
        + "; ".join(errors)
    )


def clean_checkpoints(
    model_dir_path: str | Path = "logs/44k/",
    n_ckpts_to_keep: int = 2,
    sort_by_time: bool = True,
) -> None:
    """
    指定されたディレクトリから古いチェックポイントを削除して空き容量を確保する

    Args:
        model_dir_path (str | Path): モデルが保存されているディレクトリのパス
        n_ckpts_to_keep (int): 保持するチェックポイントの数（G_0.pth と D_0.pth を除く）
        sort_by_time (bool): True の場合、時間順に削除。False の場合、名前順に削除
    """

    ckpts_files = [
        f
        for f in os.listdir(model_dir_path)
        if os.path.isfile(os.path.join(model_dir_path, f))
    ]

    def name_key(_f: str) -> int:
        # 接頭辞は G・D の1文字だけでなく WD・DUR・EMA・RESUME もあるので、英大文字の並びとして読む
        return int(re.compile("[A-Z]+_(\\d+)\\.pth").match(_f).group(1))  # type: ignore

    def time_key(_f: str) -> float:
        return os.path.getmtime(os.path.join(model_dir_path, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x: str) -> list[str]:
        return sorted(
            [
                f
                for f in ckpts_files
                if f.startswith(_x) and f.endswith(".pth") and not f.endswith("_0.pth")
            ],
            key=sort_key,
        )

    to_del = [
        os.path.join(model_dir_path, fn)
        for fn in (
            x_sorted("G_")[:-n_ckpts_to_keep]
            + x_sorted("D_")[:-n_ckpts_to_keep]
            + x_sorted("WD_")[:-n_ckpts_to_keep]
            + x_sorted("DUR_")[:-n_ckpts_to_keep]
            + x_sorted("EMA_")[:-n_ckpts_to_keep]
            + x_sorted("RESUME_")[:-n_ckpts_to_keep]
        )
    ]
    # 書き込みの途中で止まった一時ファイルも残さない
    to_del += [
        os.path.join(model_dir_path, f) for f in ckpts_files if f.endswith(".pth.tmp")
    ]

    def del_info(fn: str) -> None:
        return logger.info(f"Free up space by deleting ckpt {fn}")

    def del_routine(x: str) -> list[Any]:
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_latest_checkpoint_path(
    model_dir_path: str | Path, regex: str = "G_*.pth"
) -> str:
    """
    指定されたディレクトリから最新のチェックポイントのパスを取得する

    Args:
        model_dir_path (str | Path): モデルが保存されているディレクトリのパス
        regex (str): チェックポイントのファイル名の正規表現

    Returns:
        str: 最新のチェックポイントのパス
    """

    f_list = glob.glob(os.path.join(str(model_dir_path), regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    try:
        x = f_list[-1]
    except IndexError:
        raise ValueError(f"No checkpoint found in {model_dir_path} with regex {regex}")

    return x
