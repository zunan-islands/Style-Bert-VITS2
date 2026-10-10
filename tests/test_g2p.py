"""
日本語 g2p 処理の回帰テスト。
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from num2words import num2words
from pyopenjtalk import NJDFeature

from preprocess_text import process_line
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import InvalidPhoneError, clean_text_with_given_phone_tone
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
from style_bert_vits2.nlp.japanese.g2p_utils import (
    kata_tone2phone_tone,
    phone_tone2kata_tone,
)
from style_bert_vits2.nlp.nanairo_emoji import normalize_nanairo_emoji_text


NANAIRO_LAUGH_TEXT = "そう?🤭こうしてると,なんだか淳之介くんとデートしてるみたいだね?"
NANAIRO_PAUSE_TEXT = (
    "加藤の件はそこまでにしましょう.⏸小波さん,無理を承知でお願いしたいのですが."
)
NANAIRO_ANGER_TEXT = (
    "センパイってば,寧々先輩となんか2人だけのヒミツの話があったっぽいし😠"
)


def _assert_phone_tone_word2ph_consistency(
    phones: list[str],
    tones: list[int],
    word2ph: list[int],
) -> None:
    """
    g2p 結果の基本整合性を検証する。

    Args:
        phones (list[str]): 音素列
        tones (list[int]): アクセント列
        word2ph (list[int]): 文字ごとの音素数
    """

    assert len(phones) == len(tones) == sum(word2ph)
    assert phones[0] == "_"
    assert phones[-1] == "_"


def _extract_joined_sep_kata(text: str) -> tuple[str, str]:
    """
    入力テキストの正規化結果と、g2p が返したカタカナ読みを連結した文字列を取得する。

    Args:
        text (str): 読みを検証する入力テキスト

    Returns:
        tuple[str, str]: 正規化後テキストと、連結済みのカタカナ読み
    """

    norm_text, _, _, _, _, sep_kata, _ = clean_text_with_given_phone_tone(
        text=text,
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )
    assert sep_kata is not None
    return norm_text, "".join(sep_kata)


def _extract_sep_text_and_kata(text: str) -> tuple[str, list[str], list[str]]:
    """
    入力テキストの正規化結果と、g2p が返した単語表層・カタカナ読みを取得する。

    Args:
        text (str): 分かち書きと読みを検証する入力テキスト

    Returns:
        tuple[str, list[str], list[str]]: 正規化後テキスト、単語表層、カタカナ読み
    """

    norm_text, _, _, _, sep_text, sep_kata, _ = clean_text_with_given_phone_tone(
        text=text,
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )
    assert sep_text is not None
    assert sep_kata is not None
    return norm_text, sep_text, sep_kata


def _expected_minute_kata(minute: int) -> str:
    """
    分の助数詞表現における期待読みを返す。

    Args:
        minute (int): 分

    Returns:
        str: カタカナの期待読み
    """

    if minute == 0:
        return ""

    if minute in {10, 20, 30, 40, 50, 60, 70, 80, 90}:
        tens_only_map = {
            10: "ジュップン",
            20: "ニジュップン",
            30: "サンジュップン",
            40: "ヨンジュップン",
            50: "ゴジュップン",
            60: "ロクジュップン",
            70: "ナナジュップン",
            80: "ハチジュップン",
            90: "キュージュップン",
        }
        return tens_only_map[minute]

    ones_map = {
        1: "イップン",
        2: "ニフン",
        3: "サンプン",
        4: "ヨンプン",
        5: "ゴフン",
        6: "ロップン",
        7: "ナナフン",
        8: "ハップン",
        9: "キューフン",
    }
    tens_prefix_map = {
        0: "",
        1: "ジュー",
        2: "ニジュー",
        3: "サンジュー",
        4: "ヨンジュー",
        5: "ゴジュー",
        6: "ロクジュー",
        7: "ナナジュー",
        8: "ハチジュー",
        9: "キュージュー",
    }
    tens, ones = divmod(minute, 10)
    return f"{tens_prefix_map[tens]}{ones_map[ones]}"


def _expected_second_kata(second: int) -> str:
    """
    秒の助数詞表現における期待読みを返す。

    Args:
        second (int): 秒

    Returns:
        str: カタカナの期待読み
    """

    if second in {10, 20, 30, 40, 50, 60, 70, 80, 90}:
        tens_only_map = {
            10: "ジュービョー",
            20: "ニジュービョー",
            30: "サンジュービョー",
            40: "ヨンジュービョー",
            50: "ゴジュービョー",
            60: "ロクジュービョー",
            70: "ナナジュービョー",
            80: "ハチジュービョー",
            90: "キュージュービョー",
        }
        return tens_only_map[second]

    ones_map = {
        1: "イチビョー",
        2: "ニビョー",
        3: "サンビョー",
        4: "ヨンビョー",
        5: "ゴビョー",
        6: "ロクビョー",
        7: "ナナビョー",
        8: "ハチビョー",
        9: "キュービョー",
    }
    tens_prefix_map = {
        0: "",
        1: "ジュー",
        2: "ニジュー",
        3: "サンジュー",
        4: "ヨンジュー",
        5: "ゴジュー",
        6: "ロクジュー",
        7: "ナナジュー",
        8: "ハチジュー",
        9: "キュージュー",
    }
    tens, ones = divmod(second, 10)
    return f"{tens_prefix_map[tens]}{ones_map[ones]}"


def test_g2p_basic_sentence() -> None:
    """基本的な文でも phones / tones / word2ph の整合性が保たれる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="今日はいい天気ですね.",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_time_minutes_use_expected_counter_readings() -> None:
    """時刻の分が文末でも助数詞読みになり、慣用句読みに吸われない。"""

    # 0〜59 分の全パターンで、時刻の分が期待通りの読みになることを確認する
    for minute in range(60):
        norm_text, joined_sep_kata = _extract_joined_sep_kata(f"9時{minute}分")
        expected_norm_text = "九時" if minute == 0 else f"九時{minute}分"
        expected_joined_sep_kata = f"クジ{_expected_minute_kata(minute)}"
        assert norm_text == expected_norm_text, (
            f"Unexpected normalized text. minute: {minute}, "
            f"actual: {norm_text}, expected: {expected_norm_text}"
        )
        assert joined_sep_kata == expected_joined_sep_kata, (
            f"Unexpected minute reading. minute: {minute}, "
            f"norm_text: {norm_text}, actual: {joined_sep_kata}, "
            f"expected: {expected_joined_sep_kata}"
        )


def test_g2p_minute_counter_readings_match_up_to_99() -> None:
    """数字 + 分の助数詞読みを 1〜99 まで回帰検証する。"""

    # 助数詞そのものの読みは 99 まで全探索し、各十の位でも崩れないことを確認する
    for minute in range(1, 100):
        norm_text, joined_sep_kata = _extract_joined_sep_kata(f"{minute}分")
        expected_norm_text = f"{minute}分"
        expected_joined_sep_kata = _expected_minute_kata(minute)
        assert norm_text == expected_norm_text, (
            f"Unexpected normalized text. minute: {minute}, "
            f"actual: {norm_text}, expected: {expected_norm_text}"
        )
        assert joined_sep_kata == expected_joined_sep_kata, (
            f"Unexpected minute counter reading. minute: {minute}, "
            f"norm_text: {norm_text}, actual: {joined_sep_kata}, "
            f"expected: {expected_joined_sep_kata}"
        )


def test_g2p_time_seconds_keep_expected_counter_readings_after_12_fun() -> None:
    """問題が出やすい 12 分の後ろに秒が続いても、時刻として正しく読める。"""

    # 秒を全探索し、「十二分」が慣用句読みへ戻らないことと秒の助数詞読みを同時に確認する
    for second in range(1, 60):
        norm_text, joined_sep_kata = _extract_joined_sep_kata(f"9時12分{second}秒")
        expected_norm_text = f"九時12分{num2words(second, lang='ja')}秒"
        expected_joined_sep_kata = f"クジジューニフン{_expected_second_kata(second)}"
        assert norm_text == expected_norm_text, (
            f"Unexpected normalized text. second: {second}, "
            f"actual: {norm_text}, expected: {expected_norm_text}"
        )
        assert joined_sep_kata == expected_joined_sep_kata, (
            f"Unexpected second reading. second: {second}, "
            f"norm_text: {norm_text}, actual: {joined_sep_kata}, "
            f"expected: {expected_joined_sep_kata}"
        )


def test_g2p_second_counter_readings_match_up_to_99() -> None:
    """数字 + 秒の助数詞読みを 1〜99 まで回帰検証する。"""

    # 秒の助数詞読みも 99 まで全探索し、期待どおりの読みを維持することを確認する
    for second in range(1, 100):
        norm_text, joined_sep_kata = _extract_joined_sep_kata(f"{second}秒")
        expected_norm_text = f"{second}秒"
        expected_joined_sep_kata = _expected_second_kata(second)
        assert norm_text == expected_norm_text, (
            f"Unexpected normalized text. second: {second}, "
            f"actual: {norm_text}, expected: {expected_norm_text}"
        )
        assert joined_sep_kata == expected_joined_sep_kata, (
            f"Unexpected second counter reading. second: {second}, "
            f"norm_text: {norm_text}, actual: {joined_sep_kata}, "
            f"expected: {expected_joined_sep_kata}"
        )


@pytest.mark.parametrize(
    ("text", "expected_norm_text", "expected_joined_sep_kata"),
    [
        (
            "2025年9月1日",
            "2025年9月1日",
            "ニセンニジューゴネンクガツツイタチ",
        ),
        (
            "2025年9月10日9時30分",
            "2025年9月10日九時30分",
            "ニセンニジューゴネンクガツトーカクジサンジュップン",
        ),
        (
            "2025年9月12日9時12分",
            "2025年9月12日九時12分",
            "ニセンニジューゴネンクガツジューニニチクジジューニフン",
        ),
        (
            "2025年9月20日9時45分",
            "2025年9月20日九時45分",
            "ニセンニジューゴネンクガツハツカクジヨンジューゴフン",
        ),
        (
            "2025年9月24日",
            "2025年9月24日",
            "ニセンニジューゴネンクガツニジューヨッカ",
        ),
    ],
)
def test_g2p_date_and_datetime_keep_expected_readings(
    text: str,
    expected_norm_text: str,
    expected_joined_sep_kata: str,
) -> None:
    """年月日と年月日 + 時刻の代表ケースが期待どおりに読まれる。"""

    norm_text, joined_sep_kata = _extract_joined_sep_kata(text)
    assert norm_text == expected_norm_text
    assert joined_sep_kata == expected_joined_sep_kata


@pytest.mark.parametrize(
    ("text", "expected_norm_text", "expected_sep_text", "expected_sep_kata"),
    [
        (
            "@",
            "@",
            ["＠"],
            ["アット"],
        ),
        (
            "&",
            "&",
            ["＆"],
            ["アンド"],
        ),
        (
            "A@B",
            "A@B",
            ["Ａ", "＠", "Ｂ"],
            ["エイ", "アット", "ビー"],
        ),
        (
            "A&B",
            "A&B",
            ["Ａ", "＆", "Ｂ"],
            ["エイ", "アンド", "ビー"],
        ),
        (
            "1@2",
            "1@2",
            ["一", "＠", "二"],
            ["イチ", "アット", "ニ"],
        ),
        (
            "1&2",
            "1&2",
            ["一", "＆", "二"],
            ["イチ", "アンド", "ニ"],
        ),
        (
            "ABC&ABC",
            "エービーシー&エービーシー",
            ["エービーシー", "＆", "エービーシー"],
            ["エービーシー", "アンド", "エービーシー"],
        ),
        (
            "OpenAI&ChatGPT",
            "オープンエーアイ&チャットジーピーティー",
            ["オープン", "エーアイ", "＆", "チャット", "ジー", "ピー", "ティー"],
            ["オープン", "エーアイ", "アンド", "チャット", "ジー", "ピー", "ティー"],
        ),
        (
            "OpenAI & ChatGPT",
            "オープンエーアイ&チャットジーピーティー",
            ["オープン", "エーアイ", "＆", "チャット", "ジー", "ピー", "ティー"],
            ["オープン", "エーアイ", "アンド", "チャット", "ジー", "ピー", "ティー"],
        ),
        (
            "test@example.com",
            "テスト,アットマーク,イグザンプルドットコム",
            [
                "テスト",
                ",",
                "アット",
                "マーク",
                ",",
                "イグザンプル",
                "ドット",
                "コム",
            ],
            [
                "テスト",
                ",",
                "アット",
                "マーク",
                ",",
                "イグザンプル",
                "ドット",
                "コム",
            ],
        ),
    ],
)
def test_g2p_keeps_openjtalk_readable_symbols_as_symbol_tokens(
    text: str,
    expected_norm_text: str,
    expected_sep_text: list[str],
    expected_sep_kata: list[str],
) -> None:
    """OpenJTalk が読める記号をカタカナへ潰さず、記号ノードとして読ませる。"""

    norm_text, sep_text, sep_kata = _extract_sep_text_and_kata(text)
    assert norm_text == expected_norm_text
    assert sep_text == expected_sep_text
    assert sep_kata == expected_sep_kata


def test_g2p_dates_keep_same_prefix_reading_with_sentence_suffix() -> None:
    """2025 年の全日付で、年月日の読みが文末でも崩れないことを確認する。"""

    current_date = date(2025, 1, 1)
    end_date = date(2025, 12, 31)
    one_day = timedelta(days=1)
    while current_date <= end_date:
        base_text = f"{current_date.year}年{current_date.month}月{current_date.day}日"
        compare_text = f"{base_text}です"
        _, base_joined_sep_kata = _extract_joined_sep_kata(base_text)
        _, compare_joined_sep_kata = _extract_joined_sep_kata(compare_text)
        assert compare_joined_sep_kata.startswith(base_joined_sep_kata), (
            f"Unexpected date reading prefix. text: {base_text}, "
            f"base: {base_joined_sep_kata}, compare: {compare_joined_sep_kata}"
        )
        current_date += one_day


def test_g2p_datetimes_keep_same_prefix_reading_with_sentence_suffix() -> None:
    """2025 年の全日付で、年月日 + 9時12分 の読みが文末でも崩れないことを確認する。"""

    current_date = date(2025, 1, 1)
    end_date = date(2025, 12, 31)
    one_day = timedelta(days=1)
    while current_date <= end_date:
        base_text = (
            f"{current_date.year}年{current_date.month}月{current_date.day}日9時12分"
        )
        compare_text = f"{base_text}です"
        _, base_joined_sep_kata = _extract_joined_sep_kata(base_text)
        _, compare_joined_sep_kata = _extract_joined_sep_kata(compare_text)
        assert compare_joined_sep_kata.startswith(base_joined_sep_kata), (
            f"Unexpected datetime reading prefix. text: {base_text}, "
            f"base: {base_joined_sep_kata}, compare: {compare_joined_sep_kata}"
        )
        current_date += one_day


def test_g2p_idiomatic_juunibun_keeps_lexical_reading() -> None:
    """時刻以外の「十二分」は慣用表現としてジュウニブンを維持する。"""

    norm_text, joined_sep_kata = _extract_joined_sep_kata("もう十二分に満足した")
    expected_joined_sep_kata = "モージューニブンニマンゾクシタ"
    assert norm_text == "もう十二分に満足した"
    assert joined_sep_kata == expected_joined_sep_kata


def test_clean_text_with_given_phone_tone_nanairo_keeps_emoji_mora() -> None:
    """Nanairo モードでは絵文字モーラを 1 記号として保持する。"""

    norm_text, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_LAUGH_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )

    assert "🤭" in norm_text
    assert "🤭" in phones
    assert tones[phones.index("🤭")] == 0
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)
    assert len(word2ph) == len(norm_text) + 2


def test_clean_text_with_given_phone_tone_standard_strips_emoji_mora() -> None:
    """従来モデルでは絵文字モーラを text / phone / tone から除去する。"""

    norm_text, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_LAUGH_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=False,
        raise_yomi_error=False,
    )

    assert "🤭" not in norm_text
    assert "🤭" not in phones
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)
    assert len(word2ph) == len(norm_text) + 2


def test_clean_text_with_given_phone_tone_given_phone_round_trip() -> None:
    """6 列入力相当の given_phone / given_tone でも絵文字モーラを保持できる。"""

    norm_text, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_PAUSE_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )

    round_tripped = clean_text_with_given_phone_tone(
        text=norm_text,
        language=Languages.JP,
        given_phone=phones,
        given_tone=tones,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    standard_result = clean_text_with_given_phone_tone(
        text=norm_text,
        language=Languages.JP,
        given_phone=phones,
        given_tone=tones,
        use_jp_extra=True,
        use_nanairo=False,
        raise_yomi_error=False,
    )

    assert round_tripped[1] == phones
    assert round_tripped[2] == tones
    assert "⏸" in round_tripped[0]
    assert "⏸" in round_tripped[1]
    assert round_tripped[2][round_tripped[1].index("⏸")] == 0
    assert "⏸" not in standard_result[0]
    assert "⏸" not in standard_result[1]
    _assert_phone_tone_word2ph_consistency(
        round_tripped[1],
        round_tripped[2],
        round_tripped[3],
    )
    _assert_phone_tone_word2ph_consistency(
        standard_result[1],
        standard_result[2],
        standard_result[3],
    )


def test_clean_text_with_given_phone_tone_non_nanairo_validates_given_phone_tone_length() -> (
    None
):
    """Nanairo 以外でも given_phone と given_tone の長さ不整合は InvalidPhoneError とする。"""

    with pytest.raises(InvalidPhoneError):
        clean_text_with_given_phone_tone(
            text="あ💨",
            language=Languages.JP,
            given_phone=["_", "a", "💨", "_"],
            given_tone=[0, 0, 0, 0, 0],
            use_jp_extra=True,
            use_nanairo=False,
            raise_yomi_error=False,
        )


def test_clean_text_with_given_phone_tone_non_nanairo_reconciles_emoji_only_given_phone() -> (
    None
):
    """
    Nanairo 以外で given_phone が Nanairo 絵文字のみの場合でも、g2p 結果との整合のために
    word2ph を調整し、最終的な実音素列は生成側に合わせる。
    """

    norm_text, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="あ",
        language=Languages.JP,
        given_phone=["💨"],
        given_tone=[0],
        use_jp_extra=True,
        use_nanairo=False,
        raise_yomi_error=False,
    )

    assert norm_text == "あ"
    assert phones == ["_", "a", "_"]
    assert tones == [0, 0, 0]
    assert len(phones) == len(tones) == sum(word2ph)


def test_process_line_nanairo_preserves_emoji_and_standard_ignores_it(
    tmp_path: Path,
) -> None:
    """preprocess_text.process_line() でも use_nanairo=True であれば絵文字モーラを保持する。"""

    norm_text, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_ANGER_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    # wavs_dir は process_line のパス正規化に使われるだけなので、任意のディレクトリでよい
    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    line = (
        f"{wavs_dir / 'sample.ogg'}|spk_0000|JP|"
        f"{norm_text}|{' '.join(phones)}|{' '.join(str(tone) for tone in tones)}"
    )

    nanairo_processed = process_line(
        line=line,
        wavs_dir=wavs_dir,
        use_jp_extra=True,
        use_nanairo=True,
        yomi_error="raise",
    ).strip()
    standard_processed = process_line(
        line=line,
        wavs_dir=wavs_dir,
        use_jp_extra=True,
        use_nanairo=False,
        yomi_error="raise",
    ).strip()

    nanairo_fields = nanairo_processed.split("|")
    standard_fields = standard_processed.split("|")

    assert len(nanairo_fields) == 7
    assert len(standard_fields) == 7
    assert "😠" in nanairo_fields[3]
    assert "😠" in nanairo_fields[4].split()
    assert "😠" not in standard_fields[3]
    assert "😠" not in standard_fields[4].split()

    nanairo_phones = nanairo_fields[4].split()
    nanairo_tones = [int(tone) for tone in nanairo_fields[5].split()]
    nanairo_word2ph = [int(phone_count) for phone_count in nanairo_fields[6].split()]
    emoji_index = nanairo_phones.index("😠")

    assert nanairo_tones[emoji_index] == 0
    _assert_phone_tone_word2ph_consistency(
        nanairo_phones,
        nanairo_tones,
        nanairo_word2ph,
    )

    standard_phones = standard_fields[4].split()
    standard_tones = [int(tone) for tone in standard_fields[5].split()]
    standard_word2ph = [int(phone_count) for phone_count in standard_fields[6].split()]

    _assert_phone_tone_word2ph_consistency(
        standard_phones,
        standard_tones,
        standard_word2ph,
    )


def test_g2p_punctuation_marks() -> None:
    """句読点を含む文でも punctuation が保持される。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="マジすか...本当にダメなんですか?そんなに...?",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    assert "." in phones
    assert "?" in phones
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_exclamation_and_question() -> None:
    """感嘆符と疑問符が混在しても整合性が保たれる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="何やってるんですか!?早く来てよ!",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_passes_japanese_punctuation_to_pyopenjtalk() -> None:
    """
    正規化済みテキストの句点は音素用の「.」になるが、pyopenjtalk には「。」のまま渡すことを確認する。
    「番号は309.次は」のように半角の「.」で渡すと、pyopenjtalk は「309.」を小数点の付いた数として「サンビャクキュー」と読み、
    続く「次」を接頭辞の「ジ」と読む。「。」で渡せば文の切れ目として扱われ、「次」は「ツギ」と読まれる。
    BERT の入力と word2ph に使う正規化済みテキストは、従来どおり「.」のまま返す。
    """

    norm_text, phones, tones, word2ph, _, sep_kata, _ = (
        clean_text_with_given_phone_tone(
            text="番号は309。次は405。",
            language=Languages.JP,
            use_jp_extra=True,
            raise_yomi_error=False,
        )
    )

    assert norm_text == "番号は309.次は405."
    assert sep_kata is not None
    assert "ツギ" in sep_kata
    assert len(word2ph) == len(norm_text) + 2
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


@pytest.mark.parametrize(
    "text",
    [
        "番号は1234。次は5678。",
        "番号は309。次は405。",
        "番号は309、一二三。",
        "1・2年生です。",
    ],
)
def test_process_line_saves_text_that_reproduces_word2ph_for_bert(
    text: str, tmp_path: Path
) -> None:
    """
    preprocess_text.process_line() が train.list に保存するテキストを、BERT の特徴量の生成と同じく text_to_sep_kata() で解析し直したとき、
    前処理の g2p と同じ単語に分かれ、保存した word2ph の長さと一致することを確認する。
    前処理の g2p は句読点を「、」「。」のまま残した解析用テキストを形態素解析するので、「.」に置き換えた正規化済みテキストを保存すると、
    bert_gen.py が「1234.」を「千二百三十四」と解析し直し、前処理の「一二三四」と文字数が食い違って止まる。
    """

    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        process_line(
            line=f"{wavs_dir / 'sample.ogg'}|spk_0000|JP|{text}",
            wavs_dir=wavs_dir,
            use_jp_extra=True,
            use_nanairo=False,
            yomi_error="raise",
        )
        .strip()
        .split("|")
    )
    word2ph = [int(phone_count) for phone_count in fields[6].split()]
    _, _, _, _, sep_text, _, _ = clean_text_with_given_phone_tone(
        text=text,
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=True,
    )

    # bert_gen.py の extract_bert_feature() は、保存したテキストを text_to_sep_kata() で解析し直して BERT の入力を作る
    reanalyzed_sep_text = text_to_sep_kata(fields[3], raise_yomi_error=False)[0]
    assert reanalyzed_sep_text == sep_text
    assert len(word2ph) == len("".join(reanalyzed_sep_text)) + 2


def test_g2p_number_list_middle_dot_has_no_pause_phone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    コアが「1・2年生」の数を並べる中黒に付ける「数の区切り」の印があれば、g2p がその中黒を休止の音素「,」にしないことを確認する。
    コアは数を並べる中黒を休止なしに読むので、音素列にも休止を入れず、正規化済みテキストの「,」1文字には音素を割り当てない。
    固定している版のコアはこの印を付けず、数の間の中黒を小数点として読むので、
    数以外の語の間の中黒 (解析用テキストでは「、」) の語に印を付けて、g2p の扱いだけを確かめる。
    """

    original_run_frontend = pyopenjtalk_worker.run_frontend

    def run_frontend_with_number_list_mark(
        text: str, **kwargs: Any
    ) -> list[NJDFeature]:
        njd_features = original_run_frontend(text, **kwargs)
        for njd_feature in njd_features:
            if njd_feature["string"] == "、":
                njd_feature["pos_group3"] = "数の区切り"
        return njd_features

    monkeypatch.setattr(
        pyopenjtalk_worker, "run_frontend", run_frontend_with_number_list_mark
    )

    norm_text, phones, tones, word2ph, sep_text, sep_kata, _ = (
        clean_text_with_given_phone_tone(
            text="東京・大阪",
            language=Languages.JP,
            use_jp_extra=True,
            raise_yomi_error=False,
        )
    )

    assert norm_text == "東京,大阪"
    assert sep_text is not None
    assert sep_kata is not None
    assert sep_text[sep_kata.index("")] == ","
    assert "," not in phones
    assert len(word2ph) == len(norm_text) + 2
    assert word2ph[norm_text.index(",") + 1] == 0
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_long_text() -> None:
    """長文でも phones / tones / word2ph の整合性が崩れない。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text=(
            "いい機会じゃないか.天のもたらすみそぎの聖水だと思えよ."
            "澱と積もった不浄な我欲を残さず洗いながすがいい."
        ),
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_katakana_text() -> None:
    """カタカナ混じりの台詞でも整合性が保たれる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="センパイってば,ヒミツの話があったっぽいし.",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_english_mixed() -> None:
    """英語混じりの短文でも InvalidPhoneError なく処理できる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="OKです,ありがとう.",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_empty_and_minimal() -> None:
    """最小級の短文でも整合性が保たれる。"""

    for text in ("あ.", "うん."):
        _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
            text=text,
            language=Languages.JP,
            use_jp_extra=True,
            raise_yomi_error=False,
        )

        _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_ellipsis_and_dash() -> None:
    """三点リーダー相当や間を置く文でも整合性が保たれる。"""

    for text in (
        "私みたいに...全力で幸せな人生を歩んでね.",
        "まったくもって.大したものです.",
    ):
        _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
            text=text,
            language=Languages.JP,
            use_jp_extra=True,
            raise_yomi_error=False,
        )

        _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


# ============================================================
# phone_tone2kata_tone / kata_tone2phone_tone の Nanairo 絵文字パススルーテスト
# ============================================================


def test_phone_tone2kata_tone_passes_through_nanairo_emoji() -> None:
    """phone_tone2kata_tone が Nanairo 絵文字モーラを句読点と同様にそのままパススルーする。"""

    # 実運用相当: 「そう?🤭こうしてると」の g2p 出力を模擬
    _, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text="そう?🤭こうしてると.",
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    phone_tone = list(zip(phones, tones))
    kata_tone = phone_tone2kata_tone(phone_tone)

    # 🤭 がカタカナ列の中にそのまま残っている
    kata_symbols = [k for k, _ in kata_tone]
    assert "🤭" in kata_symbols, f"Emoji 🤭 was dropped from kata_tone: {kata_tone}"

    # 往復変換で絵文字が保持される
    round_tripped = kata_tone2phone_tone(kata_tone)
    round_tripped_phones = [p for p, _ in round_tripped]
    assert "🤭" in round_tripped_phones, (
        f"Emoji 🤭 was dropped after round-trip: {round_tripped}"
    )


def test_phone_tone2kata_tone_handles_trailing_emoji() -> None:
    """文末の方式B 絵文字モーラも正しくパススルーされる。"""

    # 実運用相当: 「ありがとう😊」の g2p 出力を模擬
    _, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text="ありがとう.😊",
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    phone_tone = list(zip(phones, tones))
    kata_tone = phone_tone2kata_tone(phone_tone)

    kata_symbols = [k for k, _ in kata_tone]
    assert "😊" in kata_symbols, f"Trailing emoji 😊 was dropped: {kata_tone}"

    round_tripped = kata_tone2phone_tone(kata_tone)
    round_tripped_phones = [p for p, _ in round_tripped]
    assert "😊" in round_tripped_phones


def test_kata_tone2phone_tone_preserves_emoji_position() -> None:
    """kata_tone2phone_tone が絵文字の位置と tone を正しく保持する。"""

    # 手動で構築: カタカナ「ア」+ 絵文字 😠 + カタカナ「イ」
    kata_tone: list[tuple[str, int]] = [("ア", 1), ("😠", 0), ("イ", 0)]
    result = kata_tone2phone_tone(kata_tone)

    phones = [p for p, _ in result]
    tones = [t for _, t in result]

    # 先頭 _ + a + 😠 + i + _ の 5 トークン
    assert phones == ["_", "a", "😠", "i", "_"]
    assert tones[phones.index("😠")] == 0


# ============================================================
# normalize_nanairo_emoji_text のテスト
# ============================================================


def test_normalize_nanairo_emoji_text_removes_vs16_only_for_nanairo_emojis() -> None:
    """Nanairo 定義済み絵文字に付く VS16 のみ除去し、それ以外の記号の VS16 は保持する。"""

    # Nanairo 定義済み: ⏸️ (U+23F8 + U+FE0F) → ⏸ (U+23F8)
    assert normalize_nanairo_emoji_text("テスト⏸\ufe0fです") == "テスト⏸です"
    # Nanairo 定義済み: 🌬️ (U+1F32C + U+FE0F) → 🌬 (U+1F32C)
    assert normalize_nanairo_emoji_text("🌬\ufe0f") == "🌬"
    # Nanairo 非定義: ♾️ (U+267E + U+FE0F) → VS16 は保持される（normalize_text が ♾️ を認識するのを壊さない）
    assert normalize_nanairo_emoji_text("♾\ufe0f") == "♾\ufe0f"
    # Nanairo 非定義: 👍 + 肌色修飾子 → 保持される
    assert normalize_nanairo_emoji_text("👍\U0001f3fd") == "👍\U0001f3fd"


def test_normalize_nanairo_emoji_text_removes_skin_tone_modifiers() -> None:
    """肌色修飾子 (U+1F3FB-U+1F3FF) を除去する。"""

    # 🙏🏽 (U+1F64F + U+1F3FD) → 🙏 (U+1F64F)
    assert normalize_nanairo_emoji_text("🙏\U0001f3fd") == "🙏"
    # 👌🏻 (U+1F44C + U+1F3FB) → 👌 (U+1F44C)
    assert normalize_nanairo_emoji_text("テスト👌\U0001f3fbです") == "テスト👌です"


def test_normalize_nanairo_emoji_text_resolves_zwj_sequence() -> None:
    """ZWJ sequence を定義済み絵文字に分解する。"""

    # 😮‍💨 (U+1F62E + U+200D + U+1F4A8) → 💨 (U+1F4A8)
    assert normalize_nanairo_emoji_text("テスト😮\u200d💨です") == "テスト💨です"


def test_normalize_nanairo_emoji_text_applies_semantic_mapping() -> None:
    """意味的に近い絵文字を定義済み絵文字にマッピングする。"""

    # 😡 → 😠 (怒り系)
    assert normalize_nanairo_emoji_text("なんで😡") == "なんで😠"
    # 😂 → 😆 (笑い系)
    assert normalize_nanairo_emoji_text("ウケる😂") == "ウケる😆"
    # 😢 → 😭 (泣き系)
    assert normalize_nanairo_emoji_text("悲しい😢") == "悲しい😭"


def test_normalize_nanairo_emoji_text_preserves_defined_emojis() -> None:
    """Nanairo 定義済み絵文字はそのまま保持する。"""

    assert normalize_nanairo_emoji_text("テスト🤭です😠よ") == "テスト🤭です😠よ"


def test_normalize_nanairo_emoji_text_combined() -> None:
    """VS16 + 意味的マッピングが同時に適用される。"""

    # ❤️ (U+2764 + U+FE0F) → VS16除去 → ❤ → 意味的マッピング → 🫶
    assert normalize_nanairo_emoji_text("大好き❤\ufe0f") == "大好き🫶"
