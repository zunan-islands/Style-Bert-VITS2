"""
学習中の MAS を強制アライメントの範囲へ絞るために、発話ごとの音素の範囲を pydomino で求めて保存する。

範囲は音声ファイルの隣に `<音声>.domino_spans.pt` として置き、学習の初回に一度だけ作る。
pydomino と librosa はこの機能を使うときだけ読み込む。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.logging import logger
from training.utils import load_filepaths_and_text


# 句読点と前後の無音を表す SBV2 の音素 (pydomino では pau として扱う)
PUNCTUATION_PHONES = {"!", "?", "…", ",", ".", "'", "-", "_"}
# SBV2 の音素を pydomino の音素へ読み替える (促音は cl、長母音は短母音)
PHONE_TO_DOMINO = {
    "q": "cl",
    "ty": "t",
    "zy": "j",
    "a:": "a",
    "i:": "i",
    "u:": "u",
    "e:": "e",
    "o:": "o",
}
# pydomino が音声を受け取るサンプルレート
DOMINO_SAMPLE_RATE = 16000


class MASSpanWriter:
    """学習データの各発話について、音素ごとの [開始, 終了) フレームを強制アライメントで求めて保存する。"""

    def __init__(self, frames_per_second: float) -> None:
        """
        Args:
            frames_per_second (float): スペクトログラムの1秒あたりのフレーム数 (サンプルレート / ホップ長)
        """

        # pydomino はこの機能を使うときだけ必要なので、ここで読み込む
        from pydomino import Aligner

        # 音素列と音声から音素ごとの時刻を求めるアライナー (write_lists() と phone_spans() で使う)
        self.aligner = Aligner()
        # 秒をスペクトログラムのフレームへ換算する係数
        self.frames_per_second = frames_per_second

    def write_lists(self, list_paths: list[str], wavs_dir: Path) -> None:
        """
        ファイル一覧に並ぶ発話のうち、範囲のファイルがまだ無いものについて範囲を求めて保存する。
        求められなかった発話は保存せず、学習では範囲を絞らずに扱われる。

        Args:
            list_paths (list[str]): train.list などのファイル一覧
            wavs_dir (Path): ファイル一覧の音声パスの基準になるディレクトリ
        """

        # librosa は読み込みが重いので、範囲を作るときだけ読み込む
        import librosa

        written = 0
        failed = 0
        for list_path in list_paths:
            for row in load_filepaths_and_text(list_path):
                if len(row) < 5:
                    continue
                wav_path = wavs_dir / row[0]
                spans_path = Path(f"{wav_path}.domino_spans.pt")
                if spans_path.is_file():
                    continue
                audio, _ = librosa.load(wav_path, sr=DOMINO_SAMPLE_RATE, mono=True)
                spans = self.phone_spans(audio, row[4].split(" "))
                if spans is None:
                    failed += 1
                    continue
                torch.save(torch.from_numpy(spans.astype(np.float32)), spans_path)
                written += 1
        logger.info(
            f"Wrote forced-alignment spans for MAS: written={written} failed={failed}"
        )

    def phone_spans(
        self, audio: NDArray[np.float32], phones: list[str]
    ) -> NDArray[np.float64] | None:
        """
        音素ごとの [開始, 終了) フレームを求める。続けてまとめた休止や促音は、まとめた区間をそれぞれの範囲にする。

        Args:
            audio (NDArray[np.float32]): 16kHz のモノラル音声
            phones (list[str]): 発話の SBV2 の音素列 (前後の無音を含む)

        Returns:
            NDArray[np.float64] | None: [音素数, 2] のフレーム範囲。アライメントに失敗した場合は None
        """

        # 句読点と前後の無音は pau にし、続いた pau と促音は1つにまとめる (pydomino は同じ音素の連続を扱えない)
        units: list[tuple[str, list[int]]] = []
        for index, phone in enumerate(phones):
            domino = (
                "pau"
                if phone in PUNCTUATION_PHONES
                else PHONE_TO_DOMINO.get(phone, phone)
            )
            if len(units) > 0 and units[-1][0] == domino and domino in {"pau", "cl"}:
                units[-1][1].append(index)
                continue
            units.append((domino, [index]))
        # pydomino は前後が pau の音素列を求めるので、無ければ足す
        if units[0][0] != "pau":
            units.insert(0, ("pau", []))
        if units[-1][0] != "pau":
            units.append(("pau", []))
        try:
            aligned = self.aligner.align(
                audio.astype(np.float32), " ".join(unit[0] for unit in units), 3
            )
        except Exception as ex:
            logger.warning(f"Forced alignment failed: {ex}")
            return None
        if len(aligned) != len(units):
            return None
        spans = np.zeros((len(phones), 2), dtype=np.float64)
        for (start, end, _phone), (_domino, indexes) in zip(
            aligned, units, strict=True
        ):
            for index in indexes:
                spans[index] = (
                    start * self.frames_per_second,
                    end * self.frames_per_second,
                )
        return spans
