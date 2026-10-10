from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray
from pyopenjtalk import OpenJTalk

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp.nanairo_emoji import (
    contains_nanairo_emoji_symbols,
    is_nanairo_emoji_symbol,
    normalize_nanairo_emoji_text,
    split_text_by_nanairo_emoji_symbols,
    strip_nanairo_emoji_symbols,
)
from style_bert_vits2.nlp.symbols import (
    DURATION_EVENT_EMOJI_SYMBOLS,
    DURATION_PROSODY_MARKER_EMOJI_SYMBOLS,
    DURATION_SYMBOL_TYPE_BLANK,
    DURATION_SYMBOL_TYPE_COMMA,
    DURATION_SYMBOL_TYPE_CONTENT,
    DURATION_SYMBOL_TYPE_ELLIPSIS_DOT,
    DURATION_SYMBOL_TYPE_EVENT_EMOJI,
    DURATION_SYMBOL_TYPE_EXCLAMATION,
    DURATION_SYMBOL_TYPE_HYPHEN,
    DURATION_SYMBOL_TYPE_PERIOD,
    DURATION_SYMBOL_TYPE_PROSODY_MARKER_EMOJI,
    DURATION_SYMBOL_TYPE_QUESTION,
    DURATION_SYMBOL_TYPE_QUOTE_BOUNDARY,
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    NANAIRO_SYMBOLS,
    SYMBOLS,
)


# __init__.py は配下のモジュールをインポートした時点で実行される
# PyTorch のインポートは重いので、型チェック時以外はインポートしない
if TYPE_CHECKING:
    import torch


__symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
__nanairo_symbol_to_id = {s: i for i, s in enumerate(NANAIRO_SYMBOLS)}


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    language: Languages,
    device: str,
    *,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    sep_text: list[str] | None = None,
    use_nanairo: bool = False,
) -> torch.Tensor:
    """
    テキストから BERT の特徴量を抽出する (PyTorch 推論)

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語
        device (str): 推論に利用するデバイス
        assist_text (str | None, optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)
        sep_text (list[str] | None, optional): 単語単位の単語のリスト (デフォルト: None)
        use_nanairo (bool, optional): Nanairo 専用の絵文字モーラを保持するかどうか。Defaults to False.

    Returns:
        torch.Tensor: BERT の特徴量
    """

    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature

        return extract_bert_feature(
            text=text,
            word2ph=word2ph,
            device=device,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            sep_text=sep_text,  # 日本語のみ sep_text を指定する
            use_nanairo=use_nanairo,  # Nanairo 専用の絵文字モーラを BERT に渡すかどうか
        )
    elif language == Languages.EN:
        from style_bert_vits2.nlp.english.bert_feature import extract_bert_feature

        return extract_bert_feature(
            text=text,
            word2ph=word2ph,
            device=device,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
        )
    elif language == Languages.ZH:
        from style_bert_vits2.nlp.chinese.bert_feature import extract_bert_feature

        return extract_bert_feature(
            text=text,
            word2ph=word2ph,
            device=device,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
        )
    else:
        raise ValueError(f"Language {language} not supported")


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    language: Languages,
    onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    *,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
    sep_text: list[str] | None = None,
    use_nanairo: bool = False,
) -> NDArray[Any]:
    """
    テキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        assist_text (str | None, optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)
        sep_text (list[str] | None, optional): 単語単位の単語のリスト (デフォルト: None)
        use_nanairo (bool, optional): Nanairo 専用の絵文字モーラを保持するかどうか。Defaults to False.

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature_onnx

        return extract_bert_feature_onnx(
            text=text,
            word2ph=word2ph,
            onnx_providers=onnx_providers,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            sep_text=sep_text,  # 日本語のみ sep_text を指定する
            use_nanairo=use_nanairo,  # Nanairo 専用の絵文字モーラを BERT に渡すかどうか
        )
    elif language == Languages.EN:
        from style_bert_vits2.nlp.english.bert_feature import extract_bert_feature_onnx

        return extract_bert_feature_onnx(
            text=text,
            word2ph=word2ph,
            onnx_providers=onnx_providers,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
        )
    elif language == Languages.ZH:
        from style_bert_vits2.nlp.chinese.bert_feature import extract_bert_feature_onnx

        return extract_bert_feature_onnx(
            text=text,
            word2ph=word2ph,
            onnx_providers=onnx_providers,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
        )
    else:
        raise ValueError(f"Language {language} not supported")


def normalize_japanese_text(
    text: str,
    *,
    use_nanairo: bool = False,
) -> tuple[str, str]:
    """
    日本語のテキストを正規化し、正規化済みテキストと、pyopenjtalk の形態素解析に渡す解析用テキストを返す。
    正規化済みテキストは BERT の入力と word2ph の文字の位置に使うので音素用の句読点にし、
    解析用テキストは句読点を「、」「。」のまま、数と数の間の中黒を「・」のまま残す。

    学習の前処理は解析用テキストを train.list に保存し、BERT の特徴量の生成 (bert_gen.py) はそれを形態素解析し直す。
    g2p と同じ解析用テキストを解析し直すことで、word2ph と BERT の入力の単語の分かれ方が一致する。

    Args:
        text (str): 正規化する日本語のテキスト
        use_nanairo (bool, optional): Nanairo 専用の絵文字モーラを保持するかどうか。Defaults to False.

    Returns:
        tuple[str, str]: 正規化済みテキストと解析用テキスト
    """

    from style_bert_vits2.nlp.japanese.normalizer import normalize_text

    # Nanairo では、まず入力テキスト中の絵文字を Nanairo 定義済み絵文字に正規化し、
    # その後、絵文字部分のみテキスト正規化対象から除外してそれ以外の部分を通常通り正規化する
    if use_nanairo is True:
        text = normalize_nanairo_emoji_text(text)
    if use_nanairo is True and contains_nanairo_emoji_symbols(text) is True:
        normalized_segments: list[str] = []
        analysis_segments: list[str] = []
        for segment in split_text_by_nanairo_emoji_symbols(text):
            if is_nanairo_emoji_symbol(segment) is True:
                normalized_segments.append(segment)
                analysis_segments.append(segment)
            elif segment:
                result = normalize_text(segment, return_analysis_text=True)
                normalized_segments.append(result.text)
                analysis_segments.append(result.analysis_text)
        return "".join(normalized_segments), "".join(analysis_segments)

    result = normalize_text(text, return_analysis_text=True)
    return result.text, result.analysis_text


def _clean_text(
    text: str,
    language: Languages,
    *,
    use_jp_extra: bool = True,
    use_nanairo: bool = False,
    use_tsqyomi: bool = False,
    raise_yomi_error: bool = False,
    jtalk: OpenJTalk | None = None,
) -> tuple[
    str,
    list[str],
    list[int],
    list[int],
    list[str] | None,
    list[str] | None,
    list[str] | None,
]:
    """
    テキストをクリーニングし、音素に変換する
    この関数では実装の都合上 convert_unsupported_phones_for_current_model() を呼び出さないため、
    必ずこの _clean_text() の代わりに clean_text_with_given_phone_tone() を使うこと

    Args:
        text (str): クリーニングするテキスト
        language (Languages): テキストの言語
        use_jp_extra (bool, optional): テキストが日本語の場合に JP-Extra モデルを利用するかどうか。Defaults to True.
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ (デフォルト: False)
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.
        jtalk (OpenJTalk | None, optional): 未指定時は pyopenjtalk モジュール内部で保持されているインスタンスが自動的に利用される。

    Returns:
        tuple[str, list[str], list[int], list[int], list[str] | None, list[str] | None, list[str] | None]:
            - クリーニングされたテキスト
            - 音素
            - アクセント
            - 元のテキストの各文字に音素が何個割り当てられるかのリスト
            - 単語単位の単語のリスト
            - 単語単位の単語のカタカナ読みのリスト
            - 単語単位の単語のカタカナ読みに助詞を追加したリスト
    """

    # Changed to import inside if condition to avoid unnecessary import
    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.g2p import g2p

        # norm_text は BERT の入力と word2ph の文字の位置に使うので音素用の句読点のまま返し、
        # pyopenjtalk の形態素解析には句読点を「、」「。」のまま残した解析用テキストを渡す
        norm_text, analysis_text = normalize_japanese_text(
            text, use_nanairo=use_nanairo
        )
        phones, tones, word2ph, sep_text, sep_kata, sep_kata_with_joshi = g2p(
            norm_text,
            analysis_text=analysis_text,
            use_jp_extra=use_jp_extra,
            use_nanairo=use_nanairo,
            use_tsqyomi=use_tsqyomi,
            raise_yomi_error=raise_yomi_error,
            jtalk=jtalk,
        )
    elif language == Languages.EN:
        from style_bert_vits2.nlp.english.g2p import g2p
        from style_bert_vits2.nlp.english.normalizer import normalize_text

        norm_text = normalize_text(text)
        phones, tones, word2ph = g2p(norm_text)

        # 日本語以外では sep_text, sep_kata, sep_kata_with_joshi は None になる
        sep_text = None
        sep_kata = None
        sep_kata_with_joshi = None
    elif language == Languages.ZH:
        from style_bert_vits2.nlp.chinese.g2p import g2p
        from style_bert_vits2.nlp.chinese.normalizer import normalize_text

        norm_text = normalize_text(text)
        phones, tones, word2ph = g2p(norm_text)

        # 日本語以外では sep_text, sep_kata, sep_kata_with_joshi は None になる
        sep_text = None
        sep_kata = None
        sep_kata_with_joshi = None
    else:
        raise ValueError(f"Language {language} not supported")

    return norm_text, phones, tones, word2ph, sep_text, sep_kata, sep_kata_with_joshi


def clean_text_with_given_phone_tone(
    text: str,
    language: Languages,
    *,
    given_phone: list[str] | None = None,
    given_tone: list[int] | None = None,
    use_jp_extra: bool = True,
    use_nanairo: bool = False,
    use_tsqyomi: bool = False,
    raise_yomi_error: bool = False,
    jtalk: OpenJTalk | None = None,
) -> tuple[
    str,
    list[str],
    list[int],
    list[int],
    list[str] | None,
    list[str] | None,
    list[str] | None,
]:
    """
    テキストをクリーニングし、音素に変換する
    変換時、given_phone や given_tone が与えられた場合はそれを調整して使う
    この関数は内部で convert_unsupported_phones_for_current_model() を自動的に呼び出し、対応していない音素をフォールバックする

    Args:
        text (str): クリーニングするテキスト
        language (Languages): テキストの言語
        given_phone (list[str] | None, optional): 読み上げテキストの読みを表す音素列。指定する場合は given_tone も別途指定が必要. Defaults to None.
        given_tone (list[int] | None, optional): アクセントのトーンのリスト. Defaults to None.
        use_jp_extra (bool, optional): テキストが日本語の場合に JP-Extra モデルを利用するかどうか。Defaults to True.
        use_nanairo (bool, optional): Nanairo 専用の絵文字モーラを保持するかどうか。Defaults to False.
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ (デフォルト: False)
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.
        jtalk (OpenJTalk | None, optional): 未指定時は pyopenjtalk モジュール内部で保持されているインスタンスが自動的に利用される。

    Returns:
        tuple[str, list[str], list[int], list[int], list[str] | None, list[str] | None, list[str] | None]:
            - クリーニングされたテキスト
            - 音素
            - アクセント
            - 元のテキストの各文字に音素が何個割り当てられるかのリスト
            - 単語単位の単語のリスト
            - 単語単位の単語のカタカナ読みのリスト
            - 単語単位の単語のカタカナ読みに助詞を追加したリスト
    """

    # Nanairo 非対応モデルでは絵文字モーラを事前に除去して処理を統一する
    # text / given_phone / given_tone を同じ基準で落とし、従来モデルの挙動互換を維持する
    # 事前に正規化することで、VS16 付き絵文字（⏸️ 等）も確実に検出・除去する
    if language == Languages.JP and use_nanairo is False:
        text = normalize_nanairo_emoji_text(text)
        if contains_nanairo_emoji_symbols(text) is True:
            text = strip_nanairo_emoji_symbols(text)
        if given_phone is not None and given_tone is not None:
            # ユーザー指定の phone / tone 長を絵文字除去前に検証
            ## zip() による暗黙的な切り捨てにより、esd.list 内の音素列・アクセント列 (通常より2列多い6列の行) の整合性が
            ## 壊れている場合に早めに検知できるよう、先に長さ不整合を検出する
            if len(given_phone) != len(given_tone):
                raise InvalidPhoneError(
                    f"Length of given_phone ({len(given_phone)}) != length of given_tone ({len(given_tone)})"
                )
            filtered_pairs = [
                (phone_symbol, tone_value)
                for phone_symbol, tone_value in zip(given_phone, given_tone)
                if is_nanairo_emoji_symbol(phone_symbol) is False
            ]
            if filtered_pairs:
                given_phone = [phone_symbol for phone_symbol, _ in filtered_pairs]
                given_tone = [tone_value for _, tone_value in filtered_pairs]
            else:
                given_phone = None
                given_tone = None

    # 与えられたテキストをクリーニング
    norm_text, phone, tone, word2ph, sep_text, sep_kata, sep_kata_with_joshi = (
        _clean_text(
            text,
            language,
            use_jp_extra=use_jp_extra,
            use_nanairo=use_nanairo,
            use_tsqyomi=use_tsqyomi,
            raise_yomi_error=raise_yomi_error,
            jtalk=jtalk,
        )
    )

    # phone と tone の両方が与えられた場合はそれを使う
    if given_phone is not None and given_tone is not None:
        # 指定された phone と指定された tone 両方の長さが一致していなければならない
        if len(given_phone) != len(given_tone):
            raise InvalidPhoneError(
                f"Length of given_phone ({len(given_phone)}) != length of given_tone ({len(given_tone)})"
            )
        # 与えられた音素数と pyopenjtalk で生成した読みの音素数が一致しない
        if len(given_phone) != sum(word2ph):
            # 日本語の場合、len(given_phone) と sum(word2ph) が一致するように word2ph を適切に調整する
            # 他の言語は word2ph の調整方法が思いつかないのでエラー
            if language == Languages.JP:
                from style_bert_vits2.nlp.japanese.g2p import adjust_word2ph

                # use_jp_extra でない場合は given_phone 内の「N」を「n」に変換
                if not use_jp_extra:
                    given_phone = [p if p != "N" else "n" for p in given_phone]
                # _clean_text() から取得した word2ph を調整結果で上書き
                word2ph = adjust_word2ph(word2ph, phone, given_phone)
                # 上記処理により word2ph の合計が given_phone の長さと一致するはず
                # それでも一致しないとしたら、len(generated_phone) に比べて len(given_phone) があまりに少なすぎて、
                # 各文字ごとに最低 1 以上の音素を割り当てることが不可能だったことを意味する
                # 通常無理やりにでも辻褄を合わせるため発生しないはずだが、どうしても一致しない場合はエラーとする
                if len(given_phone) != sum(word2ph):
                    raise InvalidPhoneError(
                        f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                    )
            else:
                raise InvalidPhoneError(
                    f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                )
        phone = given_phone
        # 生成あるいは指定された phone と指定された tone 両方の長さが一致していなければならない
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    # tone だけが与えられた場合は _clean_text() で生成した phone と合わせて使う
    elif given_tone is not None:
        # 生成した phone と指定された tone 両方の長さが一致していなければならない
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    # 日本語のみ、g2p 処理では対応しているが現行モデルでは対応していない特定音素を変換 (フォールバック)
    # この処理は given_phone / given_tone が調整された後に実行する必要がある
    convert_unsupported_phones_for_current_model(phone, tone, word2ph, language)

    return norm_text, phone, tone, word2ph, sep_text, sep_kata, sep_kata_with_joshi


def convert_unsupported_phones_for_current_model(
    phone: list[str],
    tone: list[int],
    word2ph: list[int],
    language: Languages,
) -> None:
    """
    g2p 処理では対応しているが現行モデルでは対応していない特定音素を、対応する音素にフォールバックする
    変更は引数で与えられた phone / tone / word2ph に in-place で適用される
    必ず phone / tone を cleaned_text_to_sequence() に渡す前に、一度だけ実行する必要がある

    Args:
        phone (list[str]): 音素リスト
        tone (list[int]): アクセントリスト
        word2ph (list[int]): 各文字に割り当てられた音素数のリスト
        language (Languages): 言語
    """

    # ここでは必ず音素数・アクセント数・word2ph の長さが一致するはず（事前チェックとして念のため）
    # 通常起こり得ないが、万が一一致しない場合、誤った対応関係で学習される可能性があるためデータセットに含めるべきでない
    assert len(phone) == len(tone) == sum(word2ph)

    # 日本語のみ、g2p 処理では対応しているが現行モデルでは対応していない特定音素を変換 (フォールバック)
    if language == Languages.JP:
        # 音素変換マップ
        PHONE_CONVERSION_MAP = {
            "kw": ("k", "u", "w"),  # 「クヮ」→「クワ」
            "gw": ("g", "u", "w"),  # 「グヮ」→「グワ」
            "fy": ("hy",),  # 「フュ」→「ヒュ」
        }

        # 変換が必要な音素のインデックスを収集
        conversion_indices: list[tuple[int, str]] = []
        for i, p in enumerate(phone):
            if p in PHONE_CONVERSION_MAP:
                conversion_indices.append((i, p))

        # 音素変換が必要な場合のみ処理を実行
        if conversion_indices:
            # インデックスは後ろから処理することで、
            # 前の変換による位置ずれの影響を受けないようにする
            for orig_idx, orig_phone in reversed(conversion_indices):
                # 変換後の音素を取得
                converted_phones = PHONE_CONVERSION_MAP[orig_phone]

                # phone リストの更新
                ## スライスで置換すると要素数が変化する
                phone[orig_idx : orig_idx + 1] = list(converted_phones)

                # tone リストの更新
                ## 元の音素のトーンを、変換後の音素全てに適用
                orig_tone = tone[orig_idx]
                tone[orig_idx : orig_idx + 1] = [orig_tone] * len(converted_phones)

                # word2ph リストの更新
                ## 元の音素が属していた文字のインデックスを特定
                char_idx = 0
                phone_count = 0
                for i, count in enumerate(word2ph):
                    if phone_count + count > orig_idx:
                        char_idx = i
                        break
                    phone_count += count

                # 該当する文字の音素数を更新
                ## kw, gw の場合、1つの音素が3つの音素に変換されるので、2つ増える
                word2ph[char_idx] += len(converted_phones) - 1

        # ここでは必ず音素数・アクセント数・word2ph の長さが一致するはず
        assert len(phone) == len(tone) == sum(word2ph)


def cleaned_text_to_sequence(
    cleaned_phones: list[str],
    tones: list[int],
    language: Languages,
    *,
    use_nanairo: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """
    音素リスト・アクセントリスト・言語を、テキスト内の対応する ID に変換する。

    Args:
        cleaned_phones (list[str]): clean_text_with_given_phone_tone() でクリーニングされた音素のリスト
        tones (list[int]): 各音素のアクセント
        language (Languages): テキストの言語
        use_nanairo (bool, optional): Nanairo 専用の絵文字モーラを使用するかどうか。Defaults to False.

    Returns:
        tuple[list[int], list[int], list[int]]: 音素 ID・トーン ID・言語 ID のリスト
    """

    symbol_to_id = __nanairo_symbol_to_id if use_nanairo is True else __symbol_to_id
    phones = [symbol_to_id[symbol] for symbol in cleaned_phones]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for _ in phones]

    return phones, tones, lang_ids


def phone_symbols_to_duration_symbol_type_ids(
    phone_symbols: Sequence[str],
    *,
    add_blank: bool = False,
) -> list[int]:
    """
    音素記号列から Nanairo duration 用の記号種別 ID 列を生成する

    Args:
        phone_symbols (Sequence[str]): `clean_text_with_given_phone_tone()` 由来の音素記号列
        add_blank (bool): `commons.intersperse()` と同じ規則で blank 種別を挿入するかどうか

    Returns:
        list[int]: 音素記号列と同じ長さの duration 記号種別 ID 列
    """

    duration_symbol_type_ids: list[int] = []
    phone_count = len(phone_symbols)
    for phone_index, phone_symbol in enumerate(phone_symbols):
        if phone_symbol in {"_", "SP"}:
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_BLANK)
        elif phone_symbol in DURATION_EVENT_EMOJI_SYMBOLS:
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_EVENT_EMOJI)
        elif phone_symbol in DURATION_PROSODY_MARKER_EMOJI_SYMBOLS:
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_PROSODY_MARKER_EMOJI)
        elif phone_symbol == ".":
            previous_is_period = (
                phone_index > 0 and phone_symbols[phone_index - 1] == "."
            )
            next_is_period = (
                phone_index + 1 < phone_count and phone_symbols[phone_index + 1] == "."
            )
            if previous_is_period is True or next_is_period is True:
                duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_ELLIPSIS_DOT)
            else:
                duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_PERIOD)
        elif phone_symbol == ",":
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_COMMA)
        elif phone_symbol == "?":
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_QUESTION)
        elif phone_symbol == "!":
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_EXCLAMATION)
        elif phone_symbol == "'":
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_QUOTE_BOUNDARY)
        elif phone_symbol == "-":
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_HYPHEN)
        else:
            duration_symbol_type_ids.append(DURATION_SYMBOL_TYPE_CONTENT)

    if add_blank is False:
        return duration_symbol_type_ids

    # 音素 ID と同じ blank 挿入規則を使い、DP/SDP に渡す補助系列の長さを一致させる
    interspersed_duration_symbol_type_ids = [DURATION_SYMBOL_TYPE_BLANK] * (
        len(duration_symbol_type_ids) * 2 + 1
    )
    interspersed_duration_symbol_type_ids[1::2] = duration_symbol_type_ids
    return interspersed_duration_symbol_type_ids


class InvalidPhoneError(ValueError):
    pass


class InvalidToneError(ValueError):
    pass
