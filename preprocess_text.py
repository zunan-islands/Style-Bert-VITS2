"""
テキスト前処理スクリプト。

esd.list から読み込んだ書き起こしデータを処理し、
音素・トーン情報を付与した train.list / val.list を生成する。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import sample

from tqdm import tqdm

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    normalize_japanese_text,
)
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.paths import (
    TrainingModelPaths,
    add_dataset_root_argument,
    add_model_argument,
)
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()


def count_lines(file_path: Path) -> int:
    """
    ファイルの行数をカウントする。

    Args:
        file_path (Path): カウント対象のファイルパス

    Returns:
        int: ファイルの行数
    """

    with file_path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def write_error_log(error_log_path: Path, line: str, error: Exception) -> None:
    """
    エラーログを書き込む。

    Args:
        error_log_path (Path): エラーログファイルのパス
        line (str): エラーが発生した行
        error (Exception): 発生した例外
    """

    with error_log_path.open("a", encoding="utf-8") as error_log:
        error_log.write(f"{line.strip()}\n{error}\n\n")


def process_line(
    line: str,
    wavs_dir: Path,
    use_jp_extra: bool,
    use_nanairo: bool,
    yomi_error: str,
) -> str:
    """
    1行のデータを処理し、音素・トーン情報を付与する。

    Args:
        line (str): 処理対象の行
            - 4 列: `utt|spk|language|text`
            - 6 列: `utt|spk|language|text|phones|tones`
        wavs_dir (Path): wavs ディレクトリのパス
        use_jp_extra (bool): JP-Extra モードを使用するかどうか
        use_nanairo (bool): Nanairo モードを使用するかどうか
        yomi_error (str): 読みエラー時の挙動（"raise", "skip", "use"）

    Returns:
        str: 処理済みの行（utt|spk|language|norm_text|phones|tones|word2ph 形式）
            日本語の norm_text の列には、g2p が形態素解析した解析用テキストを保存する

    Raises:
        ValueError: 行のフォーマットが不正な場合
    """

    splitted_line = line.strip().split("|")
    given_phone: list[str] | None = None
    given_tone: list[int] | None = None
    if len(splitted_line) == 4:
        utt, spk, language, text = splitted_line
    elif len(splitted_line) == 6:
        utt, spk, language, text, given_phone_string, given_tone_string = splitted_line
        given_phone = given_phone_string.split()
        given_tone = [int(tone) for tone in given_tone_string.split()]
    else:
        raise ValueError(f"Invalid line format: {line.strip()}")

    norm_text, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text=text,
        language=language,  # type: ignore
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        use_nanairo=use_nanairo,
        raise_yomi_error=(yomi_error != "use"),
    )
    # BERT の特徴量の生成 (bert_gen.py) は、保存したテキストを形態素解析し直して word2ph と同じ単語に分ける
    ## g2p は句読点を「、」「。」のまま残した解析用テキストを形態素解析するので、日本語は解析用テキストを保存する
    ## 「.」に置き換えた正規化済みテキストを保存すると、「1234.」を「千二百三十四」と解析し直して word2ph と文字数が食い違う
    if language == Languages.JP:
        _, norm_text = normalize_japanese_text(text, use_nanairo=use_nanairo)

    # パスを正規化: wavs/ からの相対パスに統一、スラッシュで統一
    utt_path = Path(utt)
    try:
        # wavs_dir の子孫パスの場合は相対パスに変換
        normalized_utt_path = utt_path.relative_to(wavs_dir)
    except ValueError:
        # wavs_dir の子孫でない場合（Data/model_name/wavs/ 形式など）
        # パス文字列を正規化してから処理
        utt_posix = utt_path.as_posix()
        if "/wavs/" in utt_posix:
            # Data/model_name/wavs/file.ogg 形式などの場合、wavs/ 以降を取得
            normalized_utt_path = Path(utt_posix.split("/wavs/", 1)[1])
        elif utt_posix.startswith("wavs/"):
            # wavs/file.ogg 形式などの場合、wavs/ プレフィックスを除去
            normalized_utt_path = Path(utt_posix[5:])  # "wavs/" は 5 文字
        else:
            # 既に wavs/ からの相対パスの場合はそのまま使用
            normalized_utt_path = utt_path
    # スラッシュ区切りの文字列に変換
    # Windows/Unix に関わらずスラッシュで統一
    utt = normalized_utt_path.as_posix()

    return "{}|{}|{}|{}|{}|{}|{}\n".format(
        utt,
        spk,
        language,
        norm_text,
        " ".join(phones),
        " ".join([str(i) for i in tones]),
        " ".join([str(i) for i in word2ph]),
    )


def preprocess(
    transcription_path: Path,
    cleaned_path: Path,
    train_path: Path,
    val_path: Path,
    config_path: Path,
    wavs_dir: Path,
    val_per_lang: int,
    max_val_total: int,
    use_jp_extra: bool,
    use_nanairo: bool,
    yomi_error: str,
) -> None:
    """
    テキスト前処理のメイン処理。

    esd.list を読み込み、音素・トーン情報を付与した後、
    train.list と val.list に分割して出力する。
    また、config.json に話者ID情報を書き込む。

    Args:
        transcription_path (Path): 書き起こしファイル（esd.list）のパス
        cleaned_path (Path): 前処理済みファイル（esd.list.cleaned）のパス
        train_path (Path): 学習用リスト（train.list）のパス
        val_path (Path): 検証用リスト（val.list）のパス
        config_path (Path): モデル設定ファイル（config.json）のパス
        wavs_dir (Path): wavs ディレクトリのパス
        val_per_lang (int): 話者ごとの検証データ数
        max_val_total (int): 検証データの最大数
        use_jp_extra (bool): JP-Extra モードを使用するかどうか
        use_nanairo (bool): Nanairo モードを使用するかどうか
        yomi_error (str): 読みエラー時の挙動（"raise", "skip", "use"）
    """

    assert yomi_error in ["raise", "skip", "use"]

    error_log_path = transcription_path.parent / "text_error.log"
    if error_log_path.exists():
        error_log_path.unlink()
    error_count = 0

    total_lines = count_lines(transcription_path)

    # transcription_path から 1行ずつ読み込んで文章処理して cleaned_path に書き込む
    with (
        transcription_path.open("r", encoding="utf-8") as trans_file,
        cleaned_path.open("w", encoding="utf-8") as out_file,
    ):
        for line in tqdm(
            trans_file, file=SAFE_STDOUT, total=total_lines, dynamic_ncols=True
        ):
            try:
                processed_line = process_line(
                    line,
                    wavs_dir,
                    use_jp_extra,
                    use_nanairo,
                    yomi_error,
                )
                out_file.write(processed_line)
            except Exception as ex:
                logger.error(f"An error occurred at line:\n{line.strip()}", exc_info=ex)
                write_error_log(error_log_path, line, ex)
                error_count += 1

    transcription_path = cleaned_path

    # 各話者ごとの line の辞書
    spk_utt_map: dict[str, list[str]] = defaultdict(list)

    # 話者から ID への写像
    spk_id_map: dict[str, int] = {}

    # 話者 ID
    current_sid: int = 0

    # 音源ファイルのチェックや、spk_id_map の作成
    # utt は process_line() で wavs_dir からの相対パスに正規化されているため、
    # ファイルの存在チェックは wavs_dir を基準に行う
    with transcription_path.open("r", encoding="utf-8") as f:
        audio_paths: set[str] = set()
        count_same = 0
        count_not_found = 0
        for line in f.readlines():
            utt, spk = line.strip().split("|")[:2]
            if utt in audio_paths:
                logger.warning(f"Same audio file appears multiple times: {utt}")
                count_same += 1
                continue
            # utt は wavs_dir からの相対パスなので、フルパスを構築してチェック
            audio_full_path = wavs_dir / utt
            if not audio_full_path.is_file():
                logger.warning(f"Audio not found: {utt} (checked: {audio_full_path})")
                count_not_found += 1
                continue
            audio_paths.add(utt)
            spk_utt_map[spk].append(line)

            # 新しい話者が出てきたら話者 ID を割り当て、current_sid を 1 増やす
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1
        if count_same > 0 or count_not_found > 0:
            logger.warning(
                f"Total repeated audios: {count_same}, Total number of audio not found: {count_not_found}"
            )

    train_list: list[str] = []
    val_list: list[str] = []

    # 各話者ごとに発話リストを処理
    for spk, utts in spk_utt_map.items():
        if val_per_lang == 0:
            train_list.extend(utts)
            continue
        # ランダムに val_per_lang 個のインデックスを選択
        val_indices = set(sample(range(len(utts)), val_per_lang))
        # 元の順序を保ちながらリストを分割
        for index, utt in enumerate(utts):
            if index in val_indices:
                val_list.append(utt)
            else:
                train_list.append(utt)

    # バリデーションリストのサイズ調整
    if len(val_list) > max_val_total:
        extra_val = val_list[max_val_total:]
        val_list = val_list[:max_val_total]
        # 余剰のバリデーション発話をトレーニングリストに追加（元の順序を保持）
        train_list.extend(extra_val)

    with train_path.open("w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with val_path.open("w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    with config_path.open("r", encoding="utf-8") as f:
        json_config = json.load(f)

    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)

    if error_count > 0:
        if yomi_error == "skip":
            logger.warning(
                f"An error occurred in {error_count} lines. Proceed with lines without errors. Please check {error_log_path} for details."
            )
        else:
            # yom_error == "raise" と "use" の場合。
            # "use" の場合は、そもそも yomi_error = False で処理しているので、
            # ここが実行されるのは他の例外のときなので、エラーを raise する。
            logger.error(
                f"An error occurred in {error_count} lines. Please check {error_log_path} for details."
            )
            raise Exception(
                f"An error occurred in {error_count} lines. Please check `Data/your_model_name/text_error.log` file for details."
            )
            # 何故か {error_log_path} を raise すると文字コードエラーが起きるので上のように書いている
    else:
        logger.info(
            "Training set and validation set generation from texts is complete!"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess transcription files and generate train.list / val.list.",
    )
    add_model_argument(parser)
    add_dataset_root_argument(parser)

    # 「話者ごと」のバリデーションデータ数、言語ごとではない！
    # 元のコードや設定ファイルで val_per_lang となっていたので名前をそのままにしている
    parser.add_argument(
        "--val-per-lang",
        type=int,
        default=0,
        help="Number of validation data per SPEAKER, not per language (due to compatibility with the original code).",
    )
    parser.add_argument(
        "--max-val-total",
        type=int,
        default=12,
        help="Maximum number of validation data (default: 12)",
    )
    parser.add_argument(
        "--yomi_error",
        default="raise",
        choices=["raise", "skip", "use"],
        help="Behavior when yomi error occurs: raise (default), skip, or use",
    )

    args = parser.parse_args()

    # TrainingModelPaths を使ってパスを解決
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name, dataset_root=Path(args.dataset_root))
    hyper_parameters = HyperParameters.load_from_json(paths.config_path)
    use_jp_extra = hyper_parameters.is_jp_extra_like_model()
    use_nanairo = hyper_parameters.is_nanairo_like_model()

    transcription_path = paths.esd_list_path
    cleaned_path = paths.esd_list_cleaned_path
    train_path = paths.train_list_path
    val_path = paths.val_list_path
    config_path = paths.config_path
    wavs_dir = paths.wavs_dir
    val_per_lang: int = args.val_per_lang
    max_val_total: int = args.max_val_total
    yomi_error: str = args.yomi_error

    preprocess(
        transcription_path=transcription_path,
        cleaned_path=cleaned_path,
        train_path=train_path,
        val_path=val_path,
        config_path=config_path,
        wavs_dir=wavs_dir,
        val_per_lang=val_per_lang,
        max_val_total=max_val_total,
        use_jp_extra=use_jp_extra,
        use_nanairo=use_nanairo,
        yomi_error=yomi_error,
    )
