"""
BERT 特徴量生成スクリプト。

train.list / val.list から読み込んだ音声ファイルに対応する
BERT 特徴量を生成し、.bert.pt ファイルとして保存する。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import (
    cleaned_text_to_sequence,
    convert_unsupported_phones_for_current_model,
    extract_bert_feature,
)
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.paths import (
    TrainingModelPaths,
    add_dataset_root_argument,
    add_model_argument,
)
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()


def process_line(
    x: tuple[str, bool, bool, str, bool, Path],
) -> None:
    """
    1行のデータを処理し、BERT 特徴量を生成する。

    Args:
        x (tuple[str, bool, bool, str, bool, Path]):
            (行データ, add_blank フラグ, use_nanairo フラグ, デバイス, マルチデバイス使用フラグ, wavs_dir) のタプル
    """

    line, add_blank, use_nanairo, device, use_multi_device, wavs_dir = x

    if use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"

    fields = line.strip().split("|")
    if len(fields) not in (7, 8):
        raise ValueError(f"Invalid train.list line format: {line.strip()}")

    # 属性 ID は学習時の条件入力なので、BERT 特徴量の生成では7列目までを使う
    wav_path, _, language_str, text, phones, tone, word2ph = fields[:7]
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]

    # g2p 処理では対応しているが現行モデルでは対応していない特定音素を、対応する音素にフォールバックする
    # 変更は引数で与えられた phone / tone / word2ph に in-place で適用される
    convert_unsupported_phones_for_current_model(
        phone, tone, word2ph, Languages[language_str]
    )
    phone, tone, language = cleaned_text_to_sequence(
        phone,
        tone,
        Languages[language_str],
        use_nanairo=use_nanairo,
    )

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    # wav_path は wavs_dir からの相対パスなので、拡張子非依存で派生パスを構築
    bert_relative_path = Path(wav_path).with_suffix(".bert.pt")
    bert_path = wavs_dir / bert_relative_path

    try:
        bert = torch.load(bert_path)
        assert bert.shape[-1] == len(phone)
    except Exception:
        bert = extract_bert_feature(
            text,
            word2ph,
            Languages(language_str),
            device,
            use_nanairo=use_nanairo,
        )
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate BERT features for audio files.",
    )
    add_model_argument(parser)
    add_dataset_root_argument(parser)
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for BERT feature extraction (default: cuda if available)",
    )
    parser.add_argument(
        "--use_multi_device",
        action="store_true",
        default=False,
        help="Use multiple GPUs for parallel processing",
    )
    args = parser.parse_args()

    # TrainingModelPaths を使ってパスを解決
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name, dataset_root=Path(args.dataset_root))
    config_path = paths.config_path
    wavs_dir = paths.wavs_dir

    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []

    train_files_path = Path(hps.data.training_files)
    val_files_path = Path(hps.data.validation_files)

    with open(train_files_path, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(val_files_path, encoding="utf-8") as f:
        lines.extend(f.readlines())

    add_blank = hps.data.add_blank
    use_nanairo = hps.is_nanairo_like_model()
    device: str = args.device
    use_multi_device: bool = args.use_multi_device
    num_processes: int = args.num_processes

    if len(lines) != 0:
        # pyopenjtalk の別ワーカー化により、並列処理でエラーが出る模様なので、
        # 一旦シングルスレッド強制にする
        if num_processes != 1:
            logger.warning(
                f"--num_processes={num_processes} was specified, but forcing to 1 "
                "due to pyopenjtalk worker compatibility issues."
            )
        num_processes = 1
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(
                        process_line,
                        [
                            (
                                line,
                                add_blank,
                                use_nanairo,
                                device,
                                use_multi_device,
                                wavs_dir,
                            )
                            for line in lines
                        ],
                    ),
                    total=len(lines),
                    file=SAFE_STDOUT,
                    dynamic_ncols=True,
                )
            )

    logger.info(f"bert.pt is generated! total: {len(lines)} bert.pt files.")
