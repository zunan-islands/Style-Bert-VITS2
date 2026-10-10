"""
normalize_text() のテスト。

テスト関数の記述順は normalizer.py の処理パイプライン順に従う。統合テストは一番最後に配置する。
新しいテスト関数を追加する際は、対応する処理ステップの位置に挿入すること。
"""

import pyopenjtalk
import pytest

from style_bert_vits2.nlp.japanese import normalizer as japanese_normalizer
from style_bert_vits2.nlp.japanese.normalizer import (
    __CORE_DICTIONARY_ALPHANUMERIC_WORD_PATTERN,  # pyright: ignore[reportPrivateUsage]
    __IRODORI_SYMBOL_REPLACE_MAP,  # pyright: ignore[reportPrivateUsage]
    NormalizationResult,
    normalize_text,
)


def test_normalize_text_basic():
    """
    句読点の半角化、特殊な空白文字や制御文字の処理、連続する重複記号の畳み込みなど、基本的な正規化処理が行われることを確認する。
    """

    # 基本的な句読点の正規化
    assert normalize_text("こんにちは。さようなら。") == "こんにちは.さようなら."
    assert normalize_text("おはよう、こんばんは、") == "おはよう,こんばんは,"
    assert normalize_text("すごい！やばい！") == "すごい!やばい!"
    assert normalize_text("なに？どうして？") == "なに?どうして?"
    # 特殊な空白文字
    assert normalize_text("text\u200btext") == "テキストテキスト"  # ゼロ幅スペース
    assert (
        normalize_text("text\u3000text") == "テキスト,テキスト"
    )  # 英単語同士の「text　text」の全角空白を読点へ変換
    assert normalize_text("text\ttext") == "テキストテキスト"  # タブ
    # 制御文字
    assert normalize_text("text\ntext") == "テキスト.テキスト"  # 改行
    assert normalize_text("text\rtext") == "テキストテキスト"  # キャリッジリターン
    # 重複する記号
    assert normalize_text("！！！？？？。。。") == "!!!???..."
    assert normalize_text("。。。、、、") == "...,,,"


def test_normalize_text_return_details_tracks_spoken_replacements():
    """
    return_details=True を指定した場合に、パーセンテージ・URL・記号・単位・日付など発話内容を置換する入力のみが元区間と正規化後区間付きで details に記録されることを確認する。
    """

    text = "90% https://example.com/a?b=1 %s 100m 2024/01/01"
    result = normalize_text(text, return_details=True)

    assert isinstance(result, NormalizationResult)
    assert result.text == (
        "90パーセントエイチティーティーピーエス,イグザンプルドットコム,"
        "スラッシュ,a,クエスチョン,bイコール1パーセントs100メートル2024年1月1日"
    )
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == [
        ("percentage", "90%", "90パーセント"),
        (
            "url",
            "https://example.com/a?b=1",
            "エイチティーティーピーエス,イグザンプルドットコム,スラッシュ,a,クエスチョン,bイコール1",
        ),
        ("symbol", "%", "パーセント"),
        ("unit", "100m", "100メートル"),
        ("date", "2024/01/01", "2024年1月1日"),
    ]
    for detail in result.details:
        assert text[detail.original_start : detail.original_end] == detail.original_text
        assert (
            result.text[detail.normalized_start : detail.normalized_end]
            == detail.normalized_text
        )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "fragment", "start", "end"),
    [
        ("text　1kg", "1キログラム", 4, 10),
        ("text　100%", "100パーセント", 4, 12),
        ("テキスト1キログラム、text　1kg", "1キログラム", 15, 21),
    ],
)
def test_normalize_text_english_space_detail_positions(
    text: str, fragment: str, start: int, end: int, for_irodori: bool
) -> None:
    """
    英単語と数量の間の全角空白が読点へ変換されて文字位置がずれた場合でも、後続の数量やパーセンテージの正規化後区間が正確に記録されることを確認する。
    """

    result = normalize_text(text, for_irodori=for_irodori, return_details=True)
    assert len(result.details) == 1
    detail = result.details[0]
    assert (detail.normalized_text, detail.normalized_start, detail.normalized_end) == (
        fragment,
        start,
        end,
    )
    assert result.text[start:end] == fragment
    assert text[detail.original_start : detail.original_end] == detail.original_text


def test_normalize_text_return_details_keeps_decorative_percent_symbols():
    """
    装飾用途で連続するパーセント記号において、発話へ変換された各記号が個別の区間として details に記録されることを確認する。
    """

    result = normalize_text("装飾%%%", return_details=True)

    assert result.text == "装飾パーセントパーセントパーセント"
    assert [
        (detail.original_start, detail.original_end) for detail in result.details
    ] == [
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    assert all(detail.category == "symbol" for detail in result.details)


def test_normalize_text_return_details_keeps_percent_encoding_inside_url():
    """
    URL 内のパーセントエンコーディングが独立したパーセント記号として誤認されず、URL 全体の一つの発話区間として記録されることを確認する。
    """

    text = "https://x.test/a%20b?x=50%25"
    result = normalize_text(text, for_irodori=True, return_details=True)

    assert len(result.details) == 1
    detail = result.details[0]
    assert detail.category == "url"
    assert detail.original_text == text
    assert "パーセント20" in detail.normalized_text
    assert "パーセント25" in detail.normalized_text
    assert result.text[detail.normalized_start : detail.normalized_end] == result.text


def test_normalize_text_return_details_keeps_pathless_url_query_as_one_segment():
    """
    パスを持たないクエリ付き URL についても、パーセントエンコーディングを含めて全体が一つの発話区間として details に記録されることを確認する。
    """

    text = "https://example.com?q=a%20b"
    result = normalize_text(text, for_irodori=True, return_details=True)

    assert len(result.details) == 1
    detail = result.details[0]
    assert detail.category == "url"
    assert detail.original_text == text
    assert "クエスチョン" in detail.normalized_text
    assert "パーセント20" in detail.normalized_text
    assert result.text[detail.normalized_start : detail.normalized_end] == result.text


def test_normalize_text_return_details_tracks_fullwidth_url_and_email():
    """
    全角で記述された URL やメールアドレスについても、半角化後の照合により発話区間として details に正確に記録されることを確認する。
    """

    url_text = "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ"
    email_text = "ｔｅｓｔ＠ｅｘａｍｐｌｅ．ｃｏｍ"
    url_result = normalize_text(url_text, for_irodori=True, return_details=True)
    email_result = normalize_text(email_text, for_irodori=True, return_details=True)

    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in url_result.details
    ] == [
        (
            "url",
            url_text,
            "エイチティーティーピーエス、イグザンプルドットコム",
        )
    ]
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in email_result.details
    ] == [
        (
            "email",
            email_text,
            "テスト、アットマーク、イグザンプルドットコム",
        )
    ]


def test_normalize_text_return_details_tracks_contextual_kakeru_cross_mark():
    """
    両側を漢字に挟まれた「×」が、全文正規化の文脈判定と同じく「かける」として置換され、その区間が details に記録されることを確認する。
    """

    text = "山×里"
    result = normalize_text(text, for_irodori=True, return_details=True)

    assert result.text == "山かける里"
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == [("symbol", "×", "かける")]
    hiragana_result = normalize_text("あ×い", for_irodori=True, return_details=True)
    assert hiragana_result.text == "あバツい"
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in hiragana_result.details
    ] == [("symbol", "×", "バツ")]


def test_normalize_text_return_details_tracks_number_expressions():
    """
    数式、小数、位取りの漢数字などが数値表現としてそれぞれ一つの区間として details に記録されることを確認する。
    """

    standard_result = normalize_text("1+1=2 と 二〇二四", return_details=True)
    irodori_result = normalize_text(
        "値は1.5です", for_irodori=True, return_details=True
    )

    assert standard_result.text == "1プラス1イコール2と2024"
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in standard_result.details
    ] == [
        ("number", "1+1=2", "1プラス1イコール2"),
        ("number", "二〇二四", "2024"),
    ]
    assert irodori_result.text == "値は一点五です"
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in irodori_result.details
    ] == [
        ("number", "1.5", "一点五"),
    ]
    for result, text in (
        (standard_result, "1+1=2 と 二〇二四"),
        (irodori_result, "値は1.5です"),
    ):
        for detail in result.details:
            assert (
                text[detail.original_start : detail.original_end]
                == detail.original_text
            )
            assert (
                result.text[detail.normalized_start : detail.normalized_end]
                == detail.normalized_text
            )


@pytest.mark.parametrize(
    ("text", "expected_details"),
    [
        # 前後の語で日付か分数かが決まる「1/2」も、全文と同じ分数の読みで区間を記録する
        ("小さじ1/2", [("number", "1/2", "二ぶんの一")]),
        ("玉ねぎ1/3個", [("number", "1/3", "三ぶんの一")]),
        # 前後の項で読み方が決まる等号と乗算記号も、全文と同じ読みで区間を記録する
        ("x=3", [("symbol", "=", "イコール")]),
        ("面積＝縦×横", [("symbol", "＝", "イコール"), ("symbol", "×", "かける")]),
    ],
)
def test_normalize_text_return_details_tracks_context_dependent_replacements(
    text: str, expected_details: list[tuple[str, str, str]]
) -> None:
    """
    「小さじ1/2」の分数や「面積＝縦×横」の等号のように、前後の文字列で変換が決まる区間も、
    区間だけを再変換した結果ではなく全文の変換結果に合わせて details に記録されることを確認する。
    """

    result = normalize_text(text, return_details=True)

    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == expected_details
    for detail in result.details:
        assert text[detail.original_start : detail.original_end] == detail.original_text
        assert (
            result.text[detail.normalized_start : detail.normalized_end]
            == detail.normalized_text
        )


def test_normalize_text_return_details_keeps_str_compatibility_for_irodori():
    """
    return_details=False の通常呼び出しや Irodori-TTS 向け呼び出しにおいて、戻り値が互換性を保った str 型のまま返ることを確認する。
    """

    assert normalize_text("90%") == "90パーセント"
    assert normalize_text("90%", for_irodori=True) == "90パーセント"


@pytest.mark.parametrize(
    ("text", "category", "original_text", "normalized_text"),
    [
        ("†", "symbol", "†", "ダガー"),
        ("2.5%", "percentage", "2.5%", "二ー点五パーセント"),
        ("25℃", "unit", "25℃", "25度"),
        ("3‰", "symbol", "‰", "パーミル"),
        ("a@b.com", "email", "a@b.com", "a、アットマーク、bドットコム"),
        ("R6.1.1", "date", "R6.1.1", "令和6年1月1日"),
        ("12+34=46", "number", "12+34=46", "12プラス34イコール46"),
        ("5〜10", "number", "5〜10", "5から10"),
        ("2^4", "number", "2^4", "2の4乗"),
        ("09:05", "number", "09:05", "九時5分"),
        (
            "〒100-0001",
            "number",
            "〒100-0001",
            "郵便番号100-0001",
        ),
        # 電話番号は数字のままコアに渡すので、中黒の区切りをハイフンにそろえたものが置換区間になる
        (
            "03・1234・5678",
            "number",
            "03・1234・5678",
            "03-1234-5678",
        ),
        ("東京都1-2-3", "number", "都1-2-3", "都1の2の3"),
        ("1920×1080", "number", "1920×1080", "1920かける1080"),
        ("○か×か", "symbol", "○", "マル"),
        ("⓫", "number", "⓫", "11"),
        ("♪", "symbol", "♪", ""),
        ("😀", "symbol", "😀", ""),
    ],
)
def test_normalize_text_return_details_tracks_all_spoken_replacement_kinds(
    text: str,
    category: str,
    original_text: str,
    normalized_text: str,
) -> None:
    """
    正規化パターンによって置換・追加・削除されたすべての発話区間が漏れなく details に記録されることを確認する。
    """

    result = normalize_text(text, for_irodori=True, return_details=True)

    matching_details = [
        detail
        for detail in result.details
        if detail.original_text == original_text
        and detail.normalized_text == normalized_text
    ]
    assert len(matching_details) == 1
    detail = matching_details[0]
    assert detail.category == category
    assert text[detail.original_start : detail.original_end] == original_text
    assert (
        result.text[detail.normalized_start : detail.normalized_end] == normalized_text
    )


def test_normalize_text_return_details_prefers_percentage_over_decimal_and_symbol():
    """
    「12.5%」のように小数とパーセント記号が組み合わさった場合に、個別の小数や記号ではなく一つの百分率区間として details に記録されることを確認する。
    """

    result = normalize_text("2.5%", for_irodori=True, return_details=True)

    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == [("percentage", "2.5%", "二ー点五パーセント")]


@pytest.mark.parametrize("text", ["日本語", "パーセント", "五半期", "5試合"])
def test_normalize_text_return_details_omits_unmodified_or_non_target_ranges(
    text: str,
) -> None:
    """
    発話内容が変化しない通常のテキストや、数値・記号に見えても正規化置換の対象外となる区間は details に記録されないことを確認する。
    """

    result = normalize_text(text, for_irodori=True, return_details=True)

    assert result.details == ()


@pytest.mark.parametrize(
    ("text", "original_text", "normalized_text"),
    [
        ("全体の½が賛成", "½", "二ぶんの一"),
        ("値は-0.5です", "-0.5", "マイナス零点五"),
        ("それは1ctです", "1ct", "1カラット"),
        ("Ver.2.0を公開", "Ver.2.0", "バージョン二ー点零"),
        ("v1.2.3", "v1.2.3", "ブイ一点二点三"),
        ("参加者は1、000名", "1、000名", "1000名"),
    ],
)
def test_normalize_text_return_details_records_reading_rules(
    text: str, original_text: str, normalized_text: str
) -> None:
    """
    分数の文字や負の数の符号、単位記号のように読みを書き換えた区間が、details に元の文字列と書き換えた後の文字列の組として記録されることを確認する。
    記録が抜けると、details から原文を復元するときに書き換えた後の読みが原文に残る。
    """

    result = normalize_text(text, for_irodori=True, return_details=True)

    matching_details = [
        detail
        for detail in result.details
        if detail.original_text == original_text
        and detail.normalized_text == normalized_text
    ]
    assert len(matching_details) == 1
    detail = matching_details[0]
    assert text[detail.original_start : detail.original_end] == original_text
    assert (
        result.text[detail.normalized_start : detail.normalized_end] == normalized_text
    )


@pytest.mark.parametrize(
    ("text", "expected", "expected_analysis_text"),
    [
        # 部屋番号の漢数字の直後の読点と句点を「、」「。」のまま残し、pyopenjtalk に「一〇一」を桁区切りの「百一」と読ませない
        (
            "石田ハイツ101・102です。",
            "石田ハイツ'一〇一,'一〇二です.",
            "石田ハイツ'一〇一、'一〇二です。",
        ),
        # 住所や電話番号の後に正規化が入れる区切りも、解析用テキストでは「、」にする
        (
            "東京都港区赤坂1-2-3-405 電話03-1234-5678",
            "東京都港区赤坂1の2の3の四〇五,電話03-1234-5678",
            "東京都港区赤坂1の2の3の四〇五、電話03-1234-5678",
        ),
        (
            "こんにちは、世界。元気ですか？",
            "こんにちは,世界.元気ですか?",
            "こんにちは、世界。元気ですか?",
        ),
        # 小数点は句読点ではないので、解析用テキストでも「.」のまま残す
        ("値は3.14です。", "値は3.14です.", "値は3.14です。"),
        # 文の終わりの全角の「．」は、前段で半角の「.」になっても解析用テキストでは「。」にし、「309.次」を小数と読ませない
        ("番号は309．次は405．", "番号は309.次は405.", "番号は309。次は405。"),
        ("十五．二人の銀座", "十五.二人の銀座", "十五。二人の銀座"),
        # 全角の「．」でも、数字や漢数字が続く小数点や、英字の略語の「．」は文の終わりにしない
        ("１．５倍", "1.5倍", "1.5倍"),
        ("三．五％", "三点五パーセント", "三点五パーセント"),
        ("ｅｔｃ．です", "エットセトラ.です", "エットセトラ.です"),
        ("Ｑ１．あいのり", "Q1あいのり", "Q1あいのり"),
        ("６．１．役割", "6.1.役割", "6.1.役割"),
        ("なるほど．．．", "なるほど...", "なるほど..."),
        # 数と数の間の中黒は、数を並べる区切りか縦書きの小数点かをコアが決めるので、解析用テキストでも「・」のまま残す
        ("1・2・3と数える。", "1,2,3と数える.", "1・2・3と数える。"),
        ("1・2年生です。", "1,2年生です.", "1・2年生です。"),
        ("3・4月", "3,4月", "3・4月"),
        ("二・三割", "二,三割", "二・三割"),
        ("〇・五％", "〇,五パーセント", "〇・五パーセント"),
        ("一・〇倍", "一,〇倍", "一・〇倍"),
        # 数以外の語の間の中黒や、伏せ字の「〇〇」の間の中黒は、従来どおり「、」にする
        ("東京・大阪", "東京,大阪", "東京、大阪"),
        ("〇〇・〇〇さん", "マルマル,マルマルさん", "マルマル、マルマルさん"),
        ("統一・一本化", "統一,一本化", "統一、一本化"),
        # 電話番号の中黒の区切りは、正規化が数字をカナにして「、」で区切る
        (
            "03・1234・5678",
            "03-1234-5678",
            "03-1234-5678",
        ),
    ],
)
def test_normalize_text_return_analysis_text_keeps_japanese_punctuation(
    text: str, expected: str, expected_analysis_text: str
) -> None:
    """
    return_analysis_text=True を指定すると、正規化済みテキストは従来と1文字も変えずに返し、
    句読点だけを音素用の「,」「.」に置き換える前の「、」「。」に戻した解析用テキストを返すことを確認する。
    pyopenjtalk は漢数字や数字の直後の半角「,」「.」を桁区切りや小数点として読み、「一〇一,」を「ヒャクイチ」、
    「309.次」を「サンビャクキュー」「ジ」と読むので、形態素解析には解析用テキストを渡す。
    「1・2年生」の数の間の中黒は、コアが数を並べる区切りとして休止なしに読み、「〇・五％」のような縦書きの小数点は小数として読むので、「・」のまま渡す。
    Irodori-TTS 向けも、正規化済みテキストの「、」のうち数の間の中黒だった区切りだけを「・」のまま残した解析用テキストを返す。
    """

    result = normalize_text(text, return_analysis_text=True)

    assert result.text == expected
    assert result.text == normalize_text(text)
    assert result.analysis_text == expected_analysis_text
    # 置換区間は return_details=True のときだけ集める
    assert result.details == ()

    irodori_result = normalize_text(text, for_irodori=True, return_analysis_text=True)
    assert irodori_result.text == normalize_text(text, for_irodori=True)
    # Irodori-TTS 向けの解析用テキストは、数の間の中黒を「、」にしていないことだけが正規化済みテキストと異なる
    assert irodori_result.analysis_text.count("・") == expected_analysis_text.count(
        "・"
    )
    assert irodori_result.analysis_text.replace("・", "、") == irodori_result.text


def _build_irodori_symbol_replace_map_cases() -> list[tuple[str, str]]:
    """
    __IRODORI_SYMBOL_REPLACE_MAP の各エントリをひらがな文脈で検証するためのテストケース一覧を生成する。
    """

    cases: list[tuple[str, str]] = []
    for key, value in __IRODORI_SYMBOL_REPLACE_MAP.items():
        if key == "\n":
            continue
        cases.append((f"あ{key}い", f"あ{value}い"))
    return cases


def _build_dash_variant_between_japanese_cases() -> list[tuple[str, str]]:
    """
    日本語文字に挟まれたダッシュ各種が半角ハイフンとして保持されるテストケース一覧を生成する。
    """

    dash_chars = [
        "-",
        "\u02d7",
        "\u2010",
        "\u2012",
        "\u2013",
        "\u2014",
        "\u2015",
        "\u2212",
        "\u2500",
        "\u2501",
        "\u2e3a",
        "\u2e3b",
    ]
    cases: list[tuple[str, str]] = []
    for dash_char in dash_chars:
        cases.append((f"あ{dash_char}い", "あ-い"))
        cases.append((f"東京{dash_char}大阪", "東京-大阪"))
    return cases


def _build_irodori_fullwidth_punctuation_variant_cases() -> list[tuple[str, str]]:
    """
    NFKC 正規化で半角化される全角句読点や各種括弧の入力を、自然な日本語表記へ復元するテストケース一覧を生成する。
    """

    return [
        # 句読点
        ("こんにちは。さようなら。", "こんにちは。さようなら。"),
        ("おはよう、こんばんは、", "おはよう、こんばんは、"),
        ("すごい!やばい!", "すごい！やばい！"),
        ("なに?どうして?", "なに？どうして？"),
        ("テスト，です", "テスト、です"),
        ("終了．", "終了。"),
        ("項目：名称", "項目、名称"),
        ("A；B", "A、B"),
        # 括弧
        ("(テスト)", "（テスト）"),
        ("（全角）", "（全角）"),
        ("[引用]", "「引用」"),
        ("【見出し】", "「見出し」"),
        ("《書名》", "「書名」"),
        # 鉤括弧・引用符
        ("「すごい！」と聞いた？", "「すごい！」と聞いた？"),
        ("\u201ctest\u201d", "「テスト」"),
        ("\u2018test\u2019", "「テスト」"),
        # 三点リーダー
        ("…", "…"),
        ("···", "…"),
        ("・・・", "…"),
        # 重複記号
        ("！！！？？？。。。", "！！！？？？。。。"),
        ("。。。、、、", "。。。、、、"),
        # Irodori-TTS で小数を英語読みさせないよう、整数部と小数部を漢数字へ変換する
        ("1.3", "一点三"),
        ("2.3", "二ー点三"),
        ("5.6", "五ー点六"),
        ("2.5次元の誘惑", "二ー点五次元の誘惑"),
        ("２．５次元", "二ー点五次元"),
        ("小数点は3.14です", "小数点は三点一四です"),
        ("10.25", "十点二五"),
        ("GPT-5.6 が", "ジーピーティー五ー点六が"),
        # 数字に挟まれていない「.」は従来通り句点へ変換される
        ("これで終わり.", "これで終わり。"),
        (".hack", "。ハック"),
    ]


def _build_irodori_differs_from_sbv2_cases() -> list[tuple[str, str, str]]:
    """
    Irodori-TTS と SBV2 で句読点の正規化結果が異なるテストケース一覧 (text, irodori_expected, sbv2_expected) を生成する。
    """

    return [
        (
            "こんにちは。さようなら。",
            "こんにちは。さようなら。",
            "こんにちは.さようなら.",
        ),
        ("おはよう、こんばんは、", "おはよう、こんばんは、", "おはよう,こんばんは,"),
        ("すごい！やばい！", "すごい！やばい！", "すごい!やばい!"),
        ("「テスト」", "「テスト」", "'テスト'"),
        ("(確認)", "（確認）", "'確認'"),
        ("なるほど…。", "なるほど…。", "なるほど...."),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_irodori_symbol_replace_map_cases(),
)
def test_normalize_text_for_irodori_symbol_replace_map(text: str, expected: str):
    """
    Irodori-TTS 向けの記号置換マップに登録された各エントリが、自然な日本語表記へ適切に置換されることを確認する。
    """

    assert normalize_text(text, for_irodori=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_irodori_fullwidth_punctuation_variant_cases(),
)
def test_normalize_text_for_irodori_fullwidth_punctuation_variants(
    text: str, expected: str
):
    """
    Irodori-TTS 向けにおいて、全角句読点が保持され、NFKC で半角化された記号が自然な日本語表記へ復元されることを確認する。
    """

    assert normalize_text(text, for_irodori=True) == expected


@pytest.mark.parametrize(
    ("text", "irodori_expected", "sbv2_expected"),
    _build_irodori_differs_from_sbv2_cases(),
)
def test_normalize_text_for_irodori_differs_from_sbv2_on_punctuation(
    text: str, irodori_expected: str, sbv2_expected: str
):
    """
    Irodori-TTS では自然な日本語表記へ、SBV2 では symbols.PUNCTUATIONS の半角記号へ正規化され、句読点の出力が各用途に合わせて正しく分かれることを確認する。
    """

    assert normalize_text(text, for_irodori=True) == irodori_expected
    assert normalize_text(text, for_irodori=False) == sbv2_expected


def test_normalize_text_for_irodori_whitespace_and_newlines():
    """
    Irodori-TTS 向けにおいて、英単語間の全角空白が読点へ変換され、段落間の改行が句点へ変換されることを確認する。
    """

    assert normalize_text("text\u3000text", for_irodori=True) == "テキスト、テキスト"
    assert normalize_text("text\ntext", for_irodori=True) == "テキスト。テキスト"
    assert normalize_text("上段\n下段", for_irodori=True) == "上段。下段"


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("重要なお知らせ　明日は休業です", "重要なお知らせ,明日は休業です"),
        ("山田太郎　代表取締役", "山田太郎,代表取締役"),
        ("東京大学　教授　山田太郎", "東京大学,教授,山田太郎"),
        ("長さ五十ｃｍ　幅は十ｃｍ", "長さ五十cm,幅は十cm"),
        ("表計算　［ＥＸＣＥＬ］", "表計算,'エクセル'"),
        ("日本語 English 日本語", "日本語イングリッシュ日本語"),
        ("問　03-1234-5678", "トイ,03-1234-5678"),
        ("質問　03-1234-5678", "質問,03-1234-5678"),
        ("問　03", "問,03"),
        ("上段\n下段", "上段.下段"),
    ],
)
def test_normalize_text_fullwidth_spaces_between_japanese_phrases(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    日本語語句に挟まれた全角空白が読点へ、改行が句点へ変換され、英単語前後の半角空白の除去や電話番号前の「問」の読み替えが適切に行われることを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、").replace(".", "。")
        if text == "表計算　［ＥＸＣＥＬ］":
            expected = "表計算、「エクセル」"
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ｍａｃ　ＯＳ　Ｘ", "マックオーエスX"),
        ("Ｍａｃ　OS　Ｘ", "マックオーエスX"),
        ("Mac　ＯＳ　X", "マックオーエスX"),
        ("Mac　OS　X", "マック,オーエス,X"),
        ("Mac OS X", "マックオーエスX"),
        ("ｔｅｘｔ　ｔｅｘｔ", "テキストテキスト"),
        ("text　text", "テキスト,テキスト"),
        ("ｈｙｄｒｏｘｙｌ　ｒａｄｉｃａｌ", "ハイドゥロキシールラディカル"),
        ("五十ｃｍ　", "五十cm,"),
        ("五百八十ｇ　", "五百八十g,"),
        ("［ｄｉｒｅｃｔ　、］", "'ダイレクト,'"),
        ("太郎　ＶＳ　次郎", "太郎,バーサス,次郎"),
        ("ＮＢＡ　８日", "エヌビーエー,8日"),
        ("ｕｎｉｔ　３", "ユニット3"),
        ("section　4", "セクション4"),
        ("Ｌｅｓｓｏｎ　８", "レッスン8"),
        ("ＴＯＰＩＣＳ　５", "トピックス5"),
        ("ｆａｌｌｏｕｔ　３", "フォールアウト3"),
        ("unit 3", "ユニットスリー"),
        (
            "ＸＰ　Ｐｒｏｆｅｓｓｉｏｎａｌ　Ｓｅｒｖｉｃｅ　Ｐａｃｋ　１",
            "エックスピープロフェッショナルサービスパック1",
        ),
        ("ＧｅＦｏｒｃｅ　ＦＸ", "ジーフォースエフエックス"),
    ],
)
def test_normalize_text_spaces_inside_english_and_after_units(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    英単語内や全角・半角英字の間の空白、および単位の後ろの全角空白が、連続した読みまたは読点へ適切に変換されることを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    if for_irodori is True and text == "［ｄｉｒｅｃｔ　、］":
        expected = "「ダイレクト、」"
    elif for_irodori is True and text == "ＮＢＡ　８日":
        expected = "エヌビーエー、8日"
    elif for_irodori is True and text == "太郎　ＶＳ　次郎":
        expected = "太郎、バーサス、次郎"
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "irodori_expected"),
    [
        (
            "七十ｃｍ　０．６ｃｍ幅平ゴム",
            "七十cm0.6センチメートル幅平ゴム",
            "七十cm零点六センチメートル幅平ゴム",
        ),
        (
            "60cm 0.7cmコード",
            "60センチメートル0.7センチメートルコード",
            "60センチメートル零点七センチメートルコード",
        ),
        ("2kg　0.7kg", "2キログラム0.7キログラム", "2キログラム零点七キログラム"),
        (
            "5mm 0.6mm",
            "5ミリメートル0.6ミリメートル",
            "5ミリメートル零点六ミリメートル",
        ),
    ],
)
def test_normalize_text_spaced_quantities(
    text: str, expected: str, irodori_expected: str, for_irodori: bool
) -> None:
    """
    数量や単位の間に置かれた空白が除去され、小数が適切に保持された上で正規化されることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == (
        irodori_expected if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "irodori_expected"),
    [
        ("1/2 個人面談を実施", "1月2日個人面談を実施", "1月2日個人面談を実施"),
        ("5kg−1.5kg", "5キログラム1.5キログラム", "5キログラム一点五キログラム"),
        (
            "○九〇一二三四五六七八",
            "090-1234-5678",
            "090-1234-5678",
        ),
        (
            "○さんに連絡してください",
            "マルさんに連絡してください",
            "マルさんに連絡してください",
        ),
    ],
)
def test_normalize_text_quantity_and_circle_contexts(
    text: str, expected: str, irodori_expected: str, for_irodori: bool
) -> None:
    """
    文脈に応じた日付・数量表現の変換や、電話番号および人名伏字における丸記号の読み分けが正しく行われることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == (
        irodori_expected if for_irodori is True else expected
    )


def test_normalize_text_for_irodori_retains_pause_apostrophe():
    """
    Irodori-TTS 向けにおいて、数字の桁区切りポーズとして使われる単独のアポストロフィが鉤括弧へ変換されず保持されることを確認する。
    """

    assert (
        normalize_text("グリーンコート赤坂409号室", for_irodori=True)
        == "グリーンコート赤坂'409号室"
    )
    assert normalize_text("5090 32G", for_irodori=True) == "5090'32G"


def test_normalize_text_for_irodori_dash_variants_to_halfwidth_hyphen():
    """
    Irodori-TTS 向けにおいて、各種ダッシュ記号が半角ハイフンへ統一されて保持されることを確認する。
    """

    assert normalize_text("あ—い", for_irodori=True) == "あ-い"
    assert normalize_text("あ―い", for_irodori=True) == "あ-い"
    assert normalize_text("あ─い", for_irodori=True) == "あ-い"
    assert normalize_text("東京–大阪", for_irodori=True) == "東京-大阪"


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_dash_variant_between_japanese_cases(),
)
def test_normalize_text_dash_variants_preserved_between_japanese_sbv2(
    text: str, expected: str
):
    """
    SBV2 向けにおいて、日本語文字に挟まれた各種ダッシュ記号が symbols.PUNCTUATIONS の半角ハイフンとして保持されることを確認する。
    """

    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_dash_variant_between_japanese_cases(),
)
def test_normalize_text_dash_variants_preserved_between_japanese_irodori(
    text: str, expected: str
):
    """
    Irodori-TTS 向けにおいて、日本語文字に挟まれた各種ダッシュ記号が半角ハイフンとして保持されることを確認する。
    """

    assert normalize_text(text, for_irodori=True) == expected


def test_normalize_text_english_hyphenated_words_still_merge():
    """
    英単語同士がハイフンで連結されている場合に、分離されずに一語の英単語として連結処理されることを確認する。
    """

    assert normalize_text("good-pen") == "グッドペン"
    assert normalize_text("OFDMEXA-modular") == "OFDMEXAモジュラー"
    assert normalize_text("good-pen", for_irodori=True) == "グッドペン"


def test_normalize_text_for_irodori_natural_prose_integration():
    """
    実際の読み上げ文に近い複合的な入力テキストが、Irodori-TTS 向けの自然な日本語表記へ正規化されることを確認する。
    """

    assert (
        normalize_text(
            "彼は「本当に!?\u3000そうなの?」と聞いた。",
            for_irodori=True,
        )
        == "彼は「本当に！？、そうなの？」と聞いた。"
    )
    assert (
        normalize_text(
            "【重要】今日の予定(仮)を確認，お願いします…",
            for_irodori=True,
        )
        == "「重要」今日の予定（仮）を確認、お願いします…"
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("そんな草wwwww", "そんな草ワラワラ"),
        ("マジかｗ", "マジかワラ"),
        ("マジかw。", "マジかワラ."),
        ("それはないｗｗ", "それはないワラワラ"),
        ("ｗｗｗ", "ワラワラ"),
        ("w", "ワラ"),
        ("草ｗｗそれな", "草ワラワラそれな"),
        ("接戦ですなｗ、見ている方も", "接戦ですなワラ,見ている方も"),
        ("眠たくｗ　寒い", "眠たくワラ,寒い"),
        ("明日も行けたら嬉しいですｗｗ〜", "明日も行けたら嬉しいですワラワラー"),
        ("こんな点数ｗだけど満足", "こんな点数ワラだけど満足"),
        ("笑ったww, nice", "笑ったワラワラ,ナイス"),
        ("笑ったww.\nHello", "笑ったワラワラ..ハロー"),
        ("ｗｗｗ\nこんにちは", "ワラワラ.こんにちは"),
        ("ﾏｼﾞｗ", "マジワラ"),
        ("笑ったww、Xで見たよ。", "笑ったワラワラ,Xで見たよ."),
    ],
)
def test_normalize_text_laughing_w(text: str, expected: str, for_irodori: bool) -> None:
    """
    日本語の文字や和文の記号の直後に続く小文字の「w」「ｗ」の並びと、括弧や行の中で単独で置かれた並びを、1個なら「ワラ」、2個以上なら「ワラワラ」と読むことを確認する。
    全角の「ｗ」は NFKC で半角になり、半角の「www」は英単語のカタカナ変換で「ウィウ」のように読まれてしまうので、どちらの経路でも先に読みへ置き換える必要がある。
    「点数ｗだけど」のように IME で打った全角の「ｗ」は、直後に日本語が続いても笑いとして読む。
    「ﾏｼﾞｗ」のように半角カタカナに続く「ｗ」も笑いとして読む。
    「笑ったww, nice」「笑ったww、Xで見たよ。」や、次の行が英文で始まる「笑ったww.」は、後ろの英字を変数の列とみなさずに笑いとして読むことを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(".", "。").replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        ("（ｗ）", "'ワラ'", "（ワラ）"),
        (
            "返信は［ｗ］だけだった。",
            "返信は'ワラ'だけだった.",
            "返信は「ワラ」だけだった。",
        ),
        ("｛ｗ｝", "ワラ", "ワラ"),
    ],
)
def test_normalize_text_laughing_w_in_brackets(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    丸括弧・角括弧・波括弧の中に単独で置かれた「ｗ」を、笑いとして「ワラ」と読むことを確認する。
    括弧そのものの変換は経路ごとに異なるため、経路ごとの出力差を含めて意図通りに正規化されることを確認する。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "www.example.com",
        "http://www.example.com/",
        "WWW",
        "Ｗｏｗ",
        "ｗｋｔｋ",
        "3w",
        "幅wの長方形",
        "Twitter",
        "w/o",
        "pwの値を求める",
        "―ｗフラグを付けて実行する",
        "関数をｗ，ｘ，ｙ，ｚの式で表す",
        "x, wの値",
        "w@example.com",
        "https://example.com/?q=w&lang=ja",
        "重み w=4.25 を使う。",
        "幅 w の長方形",
        "-w を指定する。",
        "w、x、y",
        "x,  w",
        "変数w²を使う",
        "値はw₁",
        "重みw≠0の場合だけ計算する。",
        "重みw≤0",
        "重みw≥0",
        "重みw≦0",
        "重みw≧0",
        "重みw≈0",
        "幅wﾒｰﾄﾙの長方形を描く。",
    ],
)
def test_normalize_text_laughing_w_excludes_words_and_variables(
    text: str, for_irodori: bool
) -> None:
    """
    英単語・URL・メールアドレスの中の「w」、大文字の「W」、数字に付いた単位、オプションの「-w」は、笑いとして「ワラ」と読まないことを確認する。
    直後に日本語が続くもの (「幅wの長方形」「幅wﾒｰﾄﾙ」)、空白の後に置いたもの (「重み w=4.25」)、比較や等号の演算子が続くもの (「w≠0」「w≦0」)、上付きや下付きの数字が続くもの (「w²」「w₁」)、1文字の英字と読点で並ぶもの (「w、x、y」) などの変数「w」も笑いに変換されないことを確認する。
    """

    assert "ワラ" not in normalize_text(text, for_irodori=for_irodori)


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("w@example.com", "w,アットマーク,イグザンプルドットコム"),
        (
            "https://example.com/?q=w&lang=ja",
            "エイチティーティーピーエス,イグザンプルドットコム,スラッシュ,クエスチョン,qイコールw,アンド,ラングイコールジャ",
        ),
    ],
)
def test_normalize_text_laughing_w_keeps_url_and_email_reading(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    URL やメールアドレスの中の「w」を笑いとして書き換えず、アットマークやドメインを含むアドレス全体の読みが従来どおり保たれることを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "original_text", "normalized_text"),
    [
        ("そんな草wwwww", "wwwww", "ワラワラ"),
        ("こんな点数ｗだけど満足", "ｗ", "ワラ"),
    ],
)
def test_normalize_text_return_details_tracks_laughing_w(
    text: str, original_text: str, normalized_text: str, for_irodori: bool
) -> None:
    """
    笑いの「w」を「ワラ」「ワラワラ」へ置き換えた区間が、発話内容を変えた記号の区間として details に記録されることを確認する。
    音声と照合する処理において原表記へ正しく復元できるよう、置換位置が正確に記録されていることを確認する。
    """

    result = normalize_text(text, for_irodori=for_irodori, return_details=True)
    assert [
        (detail.original_text, detail.normalized_text, detail.category)
        for detail in result.details
    ] == [(original_text, normalized_text, "symbol")]
    detail = result.details[0]
    assert text[detail.original_start : detail.original_end] == original_text
    assert (
        result.text[detail.normalized_start : detail.normalized_end] == normalized_text
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ⅱ", "二"),
        ("Ⅲ", "三"),
        ("Ⅵ", "六"),
        ("Ⅷ", "八"),
        ("Ⅻ", "十二"),
        ("ⅲ", "三"),
        ("ⅰＣ", "iC"),
        ("ＳＨ九百一ⅰＣ", "シュ九百一iC"),
        ("Core ⅰ7-12700K", "コアi7-12700K"),
        ("Core ⅰ７-12700K", "コアi7-12700K"),
        ("Ｃｏｒｅ　ⅰ７－１２７００Ｋ", "コア,i7-12700K"),
        ("Core ｉ７-12700K", "コアi7-12700K"),
        ("ⅰ7", "アイセブン"),
        ("ⅱ5", "イーアイファイブ"),
        ("ⅲ5", "アイファイブ"),
        ("ⅻ12", "シー12"),
        ("ⅰ７", "アイセブン"),
        ("ⅡG", "二G"),
        ("Ⅲ5", "三5"),
        ("Ⅱ7", "二7"),
        ("Ⅼ", "五十"),
        ("Ⅰ〜Ⅻ", "一から十二"),
        ("ⅩⅤⅠⅠ", "十七"),
        ("ⅩⅠⅠⅠ", "十三"),
        ("ⅩⅬ", "四十"),
        ("ⅩⅪ章", "二十一章"),
        ("ⅩⅫ章", "二十二章"),
        ("ⅩⅪ", "二十一"),
        ("ⅹⅺ章", "二十一章"),
        ("ⅹⅻ章", "二十二章"),
        ("図ⅩⅫ‐四十二", "図二十二の四十二"),
        ("図Ⅱ‐四十二", "図二の四十二"),
        ("図Ⅵ‐二十四", "図六の二十四"),
        ("Ⅲ‐三", "三の三"),
        ("Ⅰ〜Ⅲ", "一から三"),
        ("Ⅱ　計画の推進方策", "二,計画の推進方策"),
        ("II", "II"),
        ("VIII", "VIII"),
    ],
)
def test_normalize_text_roman_numerals(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    ローマ数字記号が漢数字へ適切に変換され、型番や通常の英字列が誤変換されず保持されることを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(".", "。").replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # カタカナや英字の名前の直後の11までのローマ数字は、英語で読む
        ("ドラゴンクエストⅧ", "ドラゴンクエストエイト"),
        ("ドラゴンクエストⅧの発売", "ドラゴンクエストエイトの発売"),
        ("ロッキーⅣ", "ロッキーフォー"),
        # 波ダッシュで範囲を示すローマ数字は、後段で「から」と読む数の範囲になるよう数詞で読む
        ("ロッキーⅣ〜Ⅵを観た", "ロッキー四から六を観た"),
        ("FFⅦ", "エフエフセブン"),
        # 12からは日本語の音韻で読みにくいので、名前の後でも数詞で読む
        ("ファイナルファンタジーⅫ", "ファイナルファンタジー十二"),
        # 「世」「型」などの漢字が続くものや、漢字の後のものは、名前と番号と確かめられないので数詞で読む
        ("ヘンリーⅧ世", "ヘンリー八世"),
        ("Ⅲ型コラーゲン", "三型コラーゲン"),
        ("第Ⅱ章", "第二章"),
        ("三國志Ⅺ", "三国志十一"),
    ],
)
def test_normalize_text_roman_numerals_after_names(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「ドラゴンクエストⅧ」「ロッキーⅣ」のように名前の直後に置いた11までのローマ数字が、「ドラゴンクエスト八」と数詞で読まれず「エイト」と英語で読まれることを確認する。
    ローマ数字を使う時点で英語で読む意図があることが多く、12からは「トゥエルブ」が日本語の音韻で読みにくいので数詞にする。
    """

    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        # 数を英語で読むか日本語で読むかが付く英字ごとに決まる語は、読みとアクセントをコアの辞書の行に任せ、英字のまま渡す
        ## 次元の「D」と世代の「G」は英語 (「スリーディ＼ー」「ファイブジ＼ー」)
        "3Dプリンター",
        "2Dの絵",
        "5Gの通信",
        "4G",
        # 解像度や間取りの「K」「LDK」などは日本語で、1のときだけ英語の「ワン」(「ヨンケ＼ー」「ニエルディーケ＼ー」「ワンケ＼ー」)
        "4Kテレビ",
        "8K",
        "2K",
        "2LDKの部屋",
        "1LDK",
        "1K",
        "3SLDK",
        "2DK",
        # 紙の大きさの「A」「B」は日本語で平板 (「エーヨン」「ビーゴ」)
        "A4の紙",
        "B5",
        "A10",
        "COVID-19",
    ],
)
def test_normalize_text_alphanumeric_words_left_to_core_dictionary(
    text: str, for_irodori: bool
) -> None:
    """
    「3D」「5G」「4K」「2LDK」「A4」「COVID-19」のように数と英字を組み合わせた語が、英単語のカタカナ読みの表で「スリーディー」「よんケー」「にーエルディーケー」と書き換えられず、そのままコアに渡ることを確認する。
    これらの語の読みとアクセント (「スリーディ＼ー」「ヨンケ＼ー」「エーヨン」の平板など) は、コアの辞書の行が持つ。
    カタカナに書き換えると、コアがカタカナの語として「スリ＼ーディー」のようにアクセントを付け直してしまう。
    """

    assert normalize_text(text, for_irodori=for_irodori) == text


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 規則の表にない英字と数字の組み合わせは、従来どおり英単語のカタカナ読みの表で読む
        ("3DS", "スリーディーエス"),
        # サッカーの「J1」「J2」と格闘技の「K-1」は、英字の後の数を英語で読む名前として表に入れている
        ("J1リーグ", "ジェイワンリーグ"),
        ("去年J2落ち", "去年ジェイツー落ち"),
        ("K-1グランプリ", "ケーワングランプリ"),
        ("H2O", "エイチツーオー"),
        # 「BS4K」は「BS」と「4K」に分けて読み、「4K」はコアの辞書の行に任せる
        ("BS4K", "ビーエス4K"),
    ],
)
def test_normalize_text_alphanumeric_words_outside_rule_table(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    数と英字の組み合わせの規則の表にない「3DS」「H2O」は、従来どおり英単語のカタカナ読みの表で読むことを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize(
    ("word", "is_paper_size"),
    [
        # ビタミンの「B1」「B2」「B6」や地下の階の「ビルB1」の用法の方が多いので、紙の大きさの規則の表に入れない
        ("B1", False),
        ("B2", False),
        ("B6", False),
        # それ以外の紙の大きさは、規則の表に入れてコアの辞書の行に任せる
        ("B0", True),
        ("B4", True),
        ("B5", True),
        ("A4", True),
    ],
)
def test_normalize_text_paper_size_rule_excludes_vitamin_and_basement_b(
    word: str, is_paper_size: bool
) -> None:
    """
    「ビタミンB1」「ビルB1」の「B1」が紙の大きさとして「ビーイチ」と平板で読まれないよう、コアの辞書の行に任せる語に入れないことを確認する。
    「B1」「B2」「B6」はビタミンや地下の階の用法の方が多いので、従来どおり英単語のカタカナ読みの経路で扱う。
    """

    assert (
        __CORE_DICTIONARY_ALPHANUMERIC_WORD_PATTERN.fullmatch(word) is not None
    ) is is_paper_size


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "fragment", "reading"),
    [
        ("Ⅱ章", "Ⅱ", "二"),
        ("図Ⅱ‐四十二", "Ⅱ‐四十二", "二の四十二"),
        ("ⅩⅬ章", "ⅩⅬ", "四十"),
        ("ⅩⅪ章", "ⅩⅪ", "二十一"),
        ("ⅩⅫ章", "ⅩⅫ", "二十二"),
    ],
)
def test_normalize_text_roman_numeral_details(
    text: str, fragment: str, reading: str, for_irodori: bool
) -> None:
    """
    ローマ数字や後続の区切りを含む表現が、元の入力区間と対応付けられて details に記録されることを確認する。
    """

    result = normalize_text(text, for_irodori=for_irodori, return_details=True)
    assert [
        (d.original_text, d.normalized_text, d.category) for d in result.details
    ] == [(fragment, reading, "number")]
    for detail in result.details:
        assert text[detail.original_start : detail.original_end] == fragment
        assert result.text[detail.normalized_start : detail.normalized_end] == reading


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "fragment", "expected", "irodori_expected"),
    [
        ("図Ⅱ‐1.1", "Ⅱ‐1.1", "二の1.1", "二の一点一"),
        ("図Ⅱ‐0.5", "Ⅱ‐0.5", "二の0.5", "二の零点五"),
        ("図Ⅱ‐１．１", "Ⅱ‐１．１", "二の1.1", "二の一点一"),
        ("図ⅰ‐1.1", "ⅰ‐1.1", "一の1.1", "一の一点一"),
        ("（図Ⅱ‐1.1）", "Ⅱ‐1.1", "二の1.1", "二の一点一"),
        ("図Ⅱ‐1〜3", "Ⅱ‐1〜3", "二の1から3", "二の1から3"),
        ("図Ⅱ‐１～３", "Ⅱ‐１～３", "二の1から3", "二の1から3"),
        ("図Ⅱ‐1~3", "Ⅱ‐1~3", "二の1から3", "二の1から3"),
        ("図Ⅱ‐1/128", "Ⅱ‐1/128", "二の百二十八ぶんの一", "二の百二十八ぶんの一"),
        (
            "図Ⅱ‐１／１２８",
            "Ⅱ‐１／１２８",
            "二の百二十八ぶんの一",
            "二の百二十八ぶんの一",
        ),
        ("図Ⅱ‐1/2", "Ⅱ‐1/2", "二の1月2日", "二の1月2日"),
    ],
)
def test_normalize_text_roman_numeral_following_number_details(
    text: str, fragment: str, expected: str, irodori_expected: str, for_irodori: bool
) -> None:
    """
    ローマ数字とそれに続く数値や範囲・分数表現が、まとめて一つの発話区間として details に記録されることを確認する。
    """

    result = normalize_text(text, for_irodori=for_irodori, return_details=True)
    reading = irodori_expected if for_irodori is True else expected
    assert [
        (d.original_text, d.normalized_text, d.category) for d in result.details
    ] == [(fragment, reading, "number")]
    detail = result.details[0]
    assert detail.original_start == text.index(fragment)
    assert detail.original_end == text.index(fragment) + len(fragment)
    assert result.text[detail.normalized_start : detail.normalized_end] == reading
    assert detail.normalized_start == result.text.index(reading)


def test_normalize_text_zero_variant_characters():
    """
    漢数字の「〇」の代用として使われる各種丸系文字がゼロとして認識され、電話番号や郵便番号などの数値パターンとして正しく正規化されることを確認する。
    """

    # --- ○ (U+25CB WHITE CIRCLE) をゼロとして使用 ---
    assert normalize_text("〒三○四ー○○○二") == "郵便番号304-0002"
    assert normalize_text("○三ー一二三四ー五六七八") == "03-1234-5678"

    # --- ◯ (U+25EF LARGE CIRCLE) をゼロとして使用 ---
    assert normalize_text("◯九◯ー一一一一ー二二二二") == "090-1111-2222"

    # --- ⭕ (U+2B55 HEAVY LARGE CIRCLE) をゼロとして使用 ---
    assert normalize_text("〒三⭕四ー⭕⭕⭕二") == "郵便番号304-0002"

    # --- ⚪ (U+26AA MEDIUM WHITE CIRCLE) をゼロとして使用 ---
    assert normalize_text("⚪三ー一二三四ー五六七八") == "03-1234-5678"

    # --- バリエーションセレクタ付きのゼロ ---
    # ⭕️ (U+2B55 + U+FE0F emoji style)
    assert (
        normalize_text("〒三⭕\ufe0f四ー⭕\ufe0f⭕\ufe0f⭕\ufe0f二")
        == "郵便番号304-0002"
    )
    # ⚪︎ (U+26AA + U+FE0E text style)
    assert normalize_text("⚪\ufe0e三ー一二三四ー五六七八") == "03-1234-5678"

    # --- 混合パターン: 異なるゼロ表記揺れ文字が混在 ---
    assert normalize_text("〒一○○ー○○○一") == "郵便番号100-0001"


def test_normalize_text_kanji_numeral_phone_numbers():
    """
    漢数字で記述された電話番号や、ハイフンの代わりに長音記号「ー」が使われた表記が、算用数字の電話番号と同じ読みへ正規化されることを確認する。
    """

    # --- 固定電話（漢数字 + 「ー」区切り） ---
    # 2桁市外局番（東京 03）
    assert normalize_text("〇三ー一二三四ー五六七八") == "03-1234-5678"
    # 3桁市外局番（横浜 045）
    assert normalize_text("〇四五ー一二三ー四五六七") == "045-123-4567"
    # 4桁市外局番
    assert normalize_text("〇四四ー一二ー三四五六") == "044-12-3456"

    # --- 携帯電話（漢数字 + 「ー」区切り） ---
    assert normalize_text("〇九〇ー一一一一ー二二二二") == "090-1111-2222"
    assert normalize_text("〇八〇ー四二〇五ー七四九一") == "080-4205-7491"

    # --- フリーダイヤル・ナビダイヤル（漢数字 + 「ー」区切り） ---
    assert normalize_text("〇一二〇ー九八二ー九五四") == "0120-982-954"
    assert normalize_text("〇五七〇ー〇一二ー三四五") == "0570-012-345"

    # --- IP 電話（漢数字 + 「ー」区切り） ---
    assert normalize_text("〇五〇ー一二三四ー五六七八") == "050-1234-5678"

    # --- ハイフンなし携帯（漢数字連続） ---
    assert normalize_text("〇九〇一一一一二二二二") == "090-1111-2222"
    # フリーダイヤル（ハイフンなし漢数字）
    assert normalize_text("〇一二〇九八二九五四") == "0120-982-954"

    # --- 文中の漢数字電話番号 ---
    assert (
        normalize_text("お電話は〇三ー一二三四ー五六七八までお願いします。")
        == "お電話は03-1234-5678までお願いします."
    )
    assert (
        normalize_text("携帯は〇九〇ー一一一一ー二二二二です。")
        == "携帯は090-1111-2222です."
    )

    # --- 半角ハイフンと漢数字の組み合わせ ---
    assert normalize_text("〇三-一二三四-五六七八") == "03-1234-5678"


def test_normalize_text_kanji_numeral_postal_codes():
    """
    漢数字で記述された郵便番号が、算用数字の郵便番号と同じ読みへ正規化されることを確認する。
    """

    # --- 〒 付き郵便番号（漢数字 + 「ー」区切り） ---
    assert normalize_text("〒三〇四ー〇〇〇二") == "郵便番号304-0002"
    assert normalize_text("〒一〇〇ー〇〇〇一") == "郵便番号100-0001"
    assert normalize_text("〒八〇二ー〇八三八") == "郵便番号802-0838"

    # --- 〒 なし郵便番号（漢数字 + 「ー」区切り） ---
    assert normalize_text("三〇四ー〇〇〇二") == "304-0002"

    # --- 半角ハイフンと漢数字の組み合わせ ---
    assert normalize_text("〒三〇四-〇〇〇二") == "郵便番号304-0002"

    # --- 文中の漢数字郵便番号 ---
    assert (
        normalize_text("〒三〇四ー〇〇〇二 茨城県下妻市今泉")
        == "郵便番号304-0002,茨城県下妻市今泉"
    )


def test_normalize_text_kanji_numeral_addresses():
    """
    漢数字で記述された住所番地が、算用数字の住所番地と同じ読みへ正規化されることを確認する。
    """

    # --- 地番形式（2要素） ---
    assert normalize_text("茨城県下妻市今泉六一三ー六") == "茨城県下妻市今泉613の6"

    # --- 住居表示形式（3要素） ---
    assert normalize_text("東京都港区六本木一ー二ー三") == "東京都港区六本木1の2の3"
    assert (
        normalize_text("大阪府大阪市北区梅田三ー一ー三")
        == "大阪府大阪市北区梅田3の1の3"
    )

    # --- 住居表示 + 部屋番号（4要素） ---
    assert normalize_text("赤坂一ー二ー三ー六〇九") == "赤坂1の2の3の六〇九"

    # --- 文中の漢数字住所 ---
    assert (
        normalize_text("住所は東京都港区六本木一ー二ー三です。")
        == "住所は東京都港区六本木1の2の3です."
    )


def test_normalize_text_fullwidth_digit_phone_postal_address():
    """
    全角数字で記述された電話番号・郵便番号・住所が、半角数字版と同じ読みへ正規化されることを確認する。
    """

    # --- 電話番号（全角数字 + 全角ハイフンマイナス） ---
    # 全角ハイフンマイナス（U+FF0D → jaconv.z2h() で半角ハイフンに変換される）
    assert normalize_text("０３\uff0d１２３４\uff0d５６７８") == "03-1234-5678"
    # 全角数字 + カタカナ長音記号「ー」
    assert normalize_text("０３ー１２３４ー５６７８") == "03-1234-5678"
    # 全角携帯
    assert normalize_text("０９０ー１１１１ー２２２２") == "090-1111-2222"
    # 全角フリーダイヤル（ハイフンなし連続数字）
    assert normalize_text("０１２０９８２９５４") == "0120-982-954"

    # --- 郵便番号（全角数字） ---
    assert normalize_text("〒３０４ー０００２") == "郵便番号304-0002"
    assert normalize_text("３０４ー０００２") == "304-0002"

    # --- 住所（全角数字） ---
    assert normalize_text("東京都港区六本木１ー２ー３") == "東京都港区六本木1の2の3"
    assert normalize_text("赤坂１ー２ー３ー６０９") == "赤坂1の2の3の六〇九"

    # --- 文中の全角数字 ---
    assert (
        normalize_text("お電話は０３ー１２３４ー５６７８までお願いします。")
        == "お電話は03-1234-5678までお願いします."
    )


def test_normalize_text_hyphen_variants_phone_postal():
    """
    各種ハイフンやダッシュ記号で区切られた電話番号や郵便番号が、セパレータとして正しく認識されて正規化されることを確認する。
    """

    # 期待される結果（全て同一）
    phone_expected = "03-1234-5678"
    postal_expected = "郵便番号304-0002"

    # --- 各種ハイフン・ダッシュ文字での電話番号 ---
    # U+002D HYPHEN-MINUS（標準）
    assert normalize_text("03-1234-5678") == phone_expected
    # U+2010 HYPHEN
    assert normalize_text("03\u20101234\u20105678") == phone_expected
    # U+2012 FIGURE DASH
    assert normalize_text("03\u20121234\u20125678") == phone_expected
    # U+2013 EN DASH
    assert normalize_text("03\u20131234\u20135678") == phone_expected
    # U+2212 MINUS SIGN
    assert normalize_text("03\u22121234\u22125678") == phone_expected
    # U+02D7 MODIFIER LETTER MINUS SIGN
    assert normalize_text("03\u02d71234\u02d75678") == phone_expected
    # U+FF0D FULLWIDTH HYPHEN-MINUS（jaconv.z2h() で半角に変換される）
    assert normalize_text("03\uff0d1234\uff0d5678") == phone_expected
    # カタカナ長音記号「ー」（U+30FC）を数字間で使用
    assert normalize_text("03ー1234ー5678") == phone_expected

    # --- 各種ハイフンでの郵便番号 ---
    # U+2010 HYPHEN
    assert normalize_text("〒304\u20100002") == postal_expected
    # U+2212 MINUS SIGN
    assert normalize_text("〒304\u22120002") == postal_expected
    # カタカナ長音記号「ー」
    assert normalize_text("〒304ー0002") == postal_expected

    # --- 漢数字 + ハイフン亜種の組み合わせ ---
    # EN DASH + 漢数字
    assert normalize_text("〇三\u2013一二三四\u2013五六七八") == phone_expected
    # MINUS SIGN + 漢数字
    assert normalize_text("〇三\u2212一二三四\u2212五六七八") == phone_expected


def test_normalize_text_kanji_numeral_non_conversion():
    """
    熟語や漢語の一部として用いられる単独の漢数字が、電話番号や番地として誤変換されずに保持されることを確認する。
    """

    # --- 漢語・熟語中の漢数字は変換されない ---
    # 「一般」の「一」は単独なので変換対象外
    assert normalize_text("一般的な話です。") == "一般的な話です."
    # 「三月」の「三」も単独
    assert normalize_text("三月に会いましょう。") == "三月に会いましょう."
    # 「二酸化炭素」の「二」も単独
    assert normalize_text("二酸化炭素が増えた。") == "二酸化炭素が増えた."
    # 「四季」の「四」も単独
    assert normalize_text("四季折々の風景") == "四季折々の風景"
    # 「七転八倒」の漢数字は単独（隣接しているが間に漢字がある）
    assert normalize_text("七転八倒する。") == "七転八倒する."

    # --- 漢数字と半角数字の混合入力 ---
    # 漢数字と半角数字がハイフンで混在するケース
    assert normalize_text("03-一二三四-五六七八") == "03-1234-5678"
    assert normalize_text("〇三-1234-五六七八") == "03-1234-5678"

    # --- 全〇 + ハイフン + 非全〇の伝播テスト ---
    # Step 1 で全〇パートは保護されるが、Step 2 の伝播でハイフン隣接の半角数字から
    # 連鎖的に変換される
    assert normalize_text("〒〇〇〇-〇〇〇一") == "郵便番号000-0001"

    # --- 〇〇 プレースホルダーは 00 に変換されず「マルマル」として読まれる ---
    # 〇〇 は数値コンテキスト外のためプレースホルダーとして「マルマル」に変換される
    assert "00" not in normalize_text("〇〇マンション205号室")
    assert "マルマル" in normalize_text("〇〇マンション205号室")

    # --- U+02D7 MODIFIER LETTER MINUS SIGN でのハイフン ---
    assert normalize_text("03\u02d71234\u02d75678") == "03-1234-5678"

    # --- 半角数字 + 漢数字の隣接は変換されない ---
    # ハイフン区切りのない数字+漢数字の隣接は、電話番号・住所の文脈ではないため変換しない
    assert normalize_text("第3四半期") == "第3四半期"
    assert normalize_text("第1四半期") == "第1四半期"
    assert normalize_text("第2三共") == "第2三共"
    assert normalize_text("3四球") == "3四球"

    # --- 10文字未満の連続漢数字は変換されない ---
    # 人名・固有名詞の可能性があるため変換しない
    assert normalize_text("一二三さん") == "一二三さん"
    assert normalize_text("一二三四五郎") == "一二三四五郎"


def test_normalize_text_kanji_numeral_sequence_with_zero() -> None:
    """
    「二千〇二十年」「一〇八」のように漢数字のゼロを含む位取り表記が、電話番号などに誤判定されず数値として保持されることを確認する。
    """

    # `〇` と非ゼロ漢数字が連続する表記は、伏せ字の丸ではなく各桁を表す数字列
    assert normalize_text("成功率七〇％", for_irodori=True) == "成功率70パーセント"
    assert normalize_text("二〇二六年", for_irodori=True) == "2026年"

    # 非数値文脈の丸印と全て丸の伏せ字は従来どおり読みを残す
    assert normalize_text("○子宮から", for_irodori=True) == "マル子宮から"
    assert normalize_text("〇〇パン", for_irodori=True) == "マルマルパン"


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 漢数字に挟まれた全角の「．」は小数点として「点」に書き換える
        ("五十九．二％", "五十九点二パーセント"),
        ("体温は三十六．五度", "体温は三十六点五度"),
        ("十．五円", "十点五円"),
        ("九十一．四メートル", "九十一点四メートル"),
        ("九百九十一．七三五五", "九百九十一点七三五五"),
        # 小数部の「〇」は伏せ字の「マル」ではなく、数の「零」として書く
        ("三十九．〇％", "三十九点零パーセント"),
        ("〇．七五％", "零点七五パーセント"),
        # 点が2つ以上続く日付や節番号、小数部に「十」を含む月日は小数として扱わない
        ("平成十三．四．六", "平成十三.四.六"),
        ("三．二十一", "三.二十一"),
        # 項目番号の後の「．」は小数点ではない
        ("十二．曲名", "十二.曲名"),
        # 元号に続く「平成十三．四月」「令和六．五」は和暦の年月なので小数点にしない
        ("平成十三．四月", "平成十三.四月"),
        ("令和六．五", "令和六.五"),
    ],
)
def test_normalize_text_kanji_decimal_point(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「五十九．二％」のように漢数字に挟まれた全角の「．」が句点として扱われて「テン」が消えないよう、
    「五十九点二」と小数点を書き換えることを確認する。
    「平成十三．四．六」のような日付や「十二．曲名」のような項目番号の「．」は、小数点に変換されないことを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(".", "。")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 曲目の番号の後の「二人の銀座」は、小数の「点二」ではなく「フタリ」と読む
        ("十五．二人の銀座", "十五.二人の銀座", "十五。二人の銀座"),
        # 元号と年の間に空白があっても、「平成　十三．四月」は和暦の年月なので小数点にしない
        ("平成　十三．四月", "平成,十三.四月", "平成、十三。四月"),
        # 漢数字の小数は、従来どおり「点」で読む
        ("五十九．二", "五十九点二", "五十九点二"),
    ],
)
def test_normalize_text_kanji_decimal_point_excludes_item_numbers_and_spaced_era(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    曲目の番号の「十五．二人の銀座」が「十五点二人」と小数に読まれず、空白を挟んだ元号の「平成　十三．四月」も「十三点四月」にならないことを確認する。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 版番号と節番号は、通常の経路でも「点」でつなぎ、コアに「ヨンテンゼロテンロク」と読ませる
        ("バージョン6.2.1", "バージョン6点2点1", "バージョン六点二点一"),
        ("４．０．６", "4点0点6", "四点ゼロ点六"),
        (
            "バージョン4.0.6が公開",
            "バージョン4点0点6が公開",
            "バージョン四点ゼロ点六が公開",
        ),
        ("6.2.1　創造性の奨励", "6点2点1,創造性の奨励", "六点二点一、創造性の奨励"),
        # IP アドレスや先頭が0の項を含む識別子、小数は「点」でつながない
        ("192.168.0.1", "192.168.0.1", "192。168。0。1"),
        ("1.02.003", "1.02.003", "1。02。003"),
        ("3.14", "3.14", "三点一四"),
        # 数の前の「Ver.」「ver」は、英語の「バー」ではなく「バージョン」と読む
        (
            "Ver.2.0を公開した",
            "バージョン2.0を公開した",
            "バージョン二ー点零を公開した",
        ),
        ("ver3", "バージョン3", "バージョン3"),
        # 「v」の後の版番号は、英単語の変換で点が消えないように「ブイ」と書き分けてから「点」でつなぐ
        ("v1.2.3", "ブイ1点2点3", "ブイ一点二点三"),
        ("v2.0", "ブイ2.0", "ブイ二ー点零"),
        # 「v」「Ver」が英単語の一部のときは書き換えない
        ("Verdi", "バーディ", "バーディ"),
    ],
)
def test_normalize_text_dotted_version_numbers(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「4.0.6」のような版番号や「6.2.1」のような節番号が、通常の経路でも Irodori 向けの経路でも「点」でつながれ、同じ読みになることを確認する。
    「.」のままだとコアは「ヨンテンゼロ、ロク」と2つ目の点で区切る。
    版番号の0は「零」と書くと「レイ」と読まれるので、Irodori 向けの経路では「ゼロ」と書く。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_circle_to_maru() -> None:
    """
    電話番号や住所などの数値コンテキスト外にある丸系文字（〇、○、◯、⭕、⚪ 等）が、伏字やプレースホルダーの「マル」として読まれることを確認する。
    """

    # --- 基本的なふせ字・伏せ字 ---
    # 〇 (U+3007 IDEOGRAPHIC NUMBER ZERO)
    assert normalize_text("〇〇電鉄") == "マルマル電鉄"
    assert normalize_text("〇〇ビール") == "マルマルビール"

    # --- 丸系 Unicode 文字のバリエーション ---
    # ○ (U+25CB WHITE CIRCLE)
    assert normalize_text("○○ビール") == "マルマルビール"
    assert normalize_text("ぶっ○せ") == "ぶっマルせ"

    # ◯ (U+25EF LARGE CIRCLE)
    assert normalize_text("◯◯会社") == "マルマル会社"

    # ⭕ (U+2B55 HEAVY LARGE CIRCLE)
    assert normalize_text("⭕⭕テスト") == "マルマルテスト"

    # ⚪ (U+26AA MEDIUM WHITE CIRCLE)
    assert normalize_text("⚪⚪マーク") == "マルマルマーク"

    # --- バリエーションセレクタ付き ---
    # U+FE0F (VARIATION SELECTOR-16, 絵文字スタイル)
    assert normalize_text("○\ufe0f○\ufe0fショップ") == "マルマルショップ"
    assert normalize_text("⭕\ufe0f⭕\ufe0f印") == "マルマル印"

    # U+FE0E (VARIATION SELECTOR-15, テキストスタイル)
    assert normalize_text("○\ufe0e○\ufe0eファクトリー") == "マルマルファクトリー"

    # --- 単独のマル ---
    assert normalize_text("これは〇です") == "これはマルです"
    # × は文脈依存で読み分けられる（ひらがな隣接 → バツ）
    assert normalize_text("○か×か") == "マルかバツか"

    # --- 数値コンテキスト内の〇は「マル」にならず数値として読まれることを確認 ---
    # 電話番号内の〇は 0 として変換される
    assert normalize_text("〇三ー一二三四ー五六七八") == "03-1234-5678"
    # 郵便番号内の〇も 0 として変換される
    assert normalize_text("〒三〇四ー〇〇〇二") == "郵便番号304-0002"
    # 住所の地番内の〇も 0 として変換される
    assert normalize_text("赤坂一ー二ー三ー六〇九") == "赤坂1の2の3の六〇九"


def test_normalize_text_url_email():
    """
    URL やメールアドレスが、スキーム・ドメイン・記号を含めて適切なカタカナ読みへ正規化されることを確認する。
    """

    # URL
    assert (
        normalize_text("https://example.com")
        == "エイチティーティーピーエス,イグザンプルドットコム"
    )
    assert (
        normalize_text("http://test.jp")
        == "エイチティーティーピー,テストドットジェイピー"
    )
    # 全角 URL
    assert (
        normalize_text("ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ")
        == "エイチティーティーピーエス,イグザンプルドットコム"
    )
    # メールアドレス
    assert (
        normalize_text("test@example.com")
        == "テスト,アットマーク,イグザンプルドットコム"
    )
    assert (
        normalize_text("info@test.co.jp")
        == "インフォ,アットマーク,テストドットシーオードットジェイピー"
    )
    # パスが無くてもクエリとパーセント符号化を URL 読みに含める
    pathless_query_url = normalize_text("https://example.com?q=a%20b")
    assert "クエスチョン" in pathless_query_url
    assert "パーセント20" in pathless_query_url
    # パーセント符号化の途中で URL が切れず、後続のクエリも URL 読みになる
    url_with_percent_encoding = normalize_text("https://x.test/a%20b?x=50%25")
    assert "クエスチョン" in url_with_percent_encoding
    assert "パーセント20" in url_with_percent_encoding
    assert "パーセント25" in url_with_percent_encoding
    # `%20` に続く数字は、`%` を号室判定より後で読むため、数字列のまま残る
    url_with_percent_then_digit = normalize_text("https://example.com/foo#bar?x=1%202")
    assert "パーセント202" in url_with_percent_then_digit
    assert "ニーマルニ" not in url_with_percent_then_digit


def test_normalize_text_divider_blocks():
    """
    装飾や境界線として用いられる連続記号ブロックが、句点へ畳み込まれて適切に区切られることを確認する。
    """

    assert normalize_text("###########") == "."
    assert normalize_text("-------------") == "."
    assert normalize_text("_____   _____") == "."
    assert normalize_text(":::::") == "."
    assert normalize_text("*****") == "."
    assert normalize_text("#$#$#$#") == "."
    assert normalize_text("そうなんだよな ##### でもなぁ") == "そうなんだよな.でもなぁ"
    # 複数行
    assert (
        normalize_text("""--------------------------
######### これはコメントです #########
=================
""")
        == ".これはコメントです."
    )
    # このような感情表現としての連続する記号は変換されない
    assert (
        normalize_text(
            "やった〜〜〜〜！！！！テストでようやく満点取れたよ・・・・・・………。。。。。。あなたはどう?????!!!"
        )
        == "やったーーーー!!!!テストでようやく満点取れたよ.....................あなたはどう?????!!!"
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize("wave_dash", ["〜", "～"])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 文末・句読点・空白の前にある数量の波ダッシュは「から」と読む
        ("スーツは27万円〜。", "スーツは27万円から.", "スーツは27万円から。"),
        ("1,000円〜", "1000円から", "1000円から"),
        ("10時〜", "10時から", "10時から"),
        ("100円〜、相談可", "100円から,相談可", "100円から、相談可"),
        ("100円〜！", "100円から!", "100円から！"),
        ("100円〜？", "100円から?", "100円から？"),
        ("「100円〜」", "'100円から'", "「100円から」"),
        ("100円〜 ", "100円から", "100円から"),
        ("100円〜\t相談可", "100円から相談可", "100円から相談可"),
        ("100円〜\n相談可", "100円から相談可", "100円から相談可"),
        ("100円 〜", "100円から", "100円から"),
        # 日付・時刻・単位を展開した結果にも同じ規則を適用する
        ("10:30〜", "十時30分から", "十時30分から"),
        ("2026/10/7〜", "2026年10月7日から", "2026年10月7日から"),
        ("0時〜", "零時から", "零時から"),
        ("10時半〜", "10時半から", "10時半から"),
        ("1kg〜", "1キログラムから", "1キログラムから"),
        ("2.5kg〜", "2.5キログラムから", "二ー点五キログラムから"),
        ("百円〜", "百円から", "百円から"),
        ("２７万円〜", "27万円から", "27万円から"),
        ("¥1,000〜", "1000円から", "1000円から"),
        ("50%〜", "50パーセントから", "50パーセントから"),
        ("六月一日〜", "六月一日から", "六月一日から"),
        ("十時三十分〜", "十時三十分から", "十時三十分から"),
        ("一週間〜", "一週間から", "一週間から"),
        ("5〜", "5から", "5から"),
        ("百〜", "百から", "百から"),
        ("10時〜 の予約", "10時からの予約", "10時からの予約"),
        ("10時〜、", "10時から,", "10時から、"),
        ("(10時〜)", "'10時から'", "（10時から）"),
        ("【3人〜】", "'3人から'", "「3人から」"),
        ("10時〜　受付開始", "10時から受付開始", "10時から受付開始"),
    ],
)
def test_normalize_text_open_ended_number_ranges(
    text: str, expected: str, expected_irodori: str, wave_dash: str, for_irodori: bool
) -> None:
    """
    数量・金額・時刻・日付に続き、終点が省略された波ダッシュが「から」に書き換えられることを確認する。
    小数や句読点については、それぞれの正規化処理での表記が保たれることを確認する。
    """

    assert normalize_text(text.replace("〜", wave_dash), for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize("wave_dash", ["〜", "～"])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        ("27万〜", "27万から", "27万から"),
        ("1.5万〜", "1.5万から", "一点五万から"),
        ("二点五〜", "二点五から", "二点五から"),
        ("二点五万〜", "二点五万から", "二点五万から"),
        ("２７万〜", "27万から", "27万から"),
        ("27億〜", "27億から", "27億から"),
        ("1万5千〜", "1万5千から", "1万5千から"),
        ("三十六点零〜", "三十六点零から", "三十六点零から"),
        ("2.5万円〜", "2.5万円から", "二ー点五万円から"),
        ("二点五〜の値", "二点五ーの値", "二点五ーの値"),
        ("27万〜の予算", "27万ーの予算", "27万ーの予算"),
        ("世界一〜", "世界一ー", "世界一ー"),
        ("Ver.2〜", "バージョン2から", "バージョン2から"),
        ("iPhone15〜", "アイフォン15から", "アイフォン15から"),
        ("第3〜", "第3から", "第3から"),
    ],
)
def test_normalize_text_open_ended_mixed_and_decimal_numbers(
    text: str, expected: str, expected_irodori: str, wave_dash: str, for_irodori: bool
) -> None:
    """
    単位のない混在表記や漢数字の小数が数量の範囲として「から」に変換され、助詞の前では長音記号が保持されることを確認する。
    """

    assert normalize_text(text.replace("〜", wave_dash), for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize("wave_dash", ["〜", "～"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5m/s〜", "5メートル毎秒から"),
        ("60km/h〜", "60キロメートル毎時から"),
        ("5メートル毎秒〜", "5メートル毎秒から"),
        ("60キロメートル毎時〜", "60キロメートル毎時から"),
        ("3時間半〜", "3時間半から"),
        ("1か月半〜", "1か月半から"),
        ("1ヶ月半〜", "1ヶ月半から"),
        ("1週間半〜", "1週間半から"),
        ("三時間半〜", "三時間半から"),
        ("一か月半〜", "一か月半から"),
        ("3h30m〜", "3時間30分から"),
        ("3時間半〜の予定", "3時間半ーの予定"),
        ("1か月半〜を想定", "1か月半ーを想定"),
        ("5m/s〜で動く", "5メートル毎秒ーで動く"),
        ("毎秒〜", "毎秒ー"),
        ("半〜", "半ー"),
    ],
)
def test_normalize_text_open_ended_rates_and_half_durations(
    text: str, expected: str, wave_dash: str, for_irodori: bool
) -> None:
    """
    「毎秒」「毎時」などのレート単位や「半」を含む期間表現の後ろの波ダッシュが、数量の範囲として「から」へ変換されることを確認する。
    """

    assert (
        normalize_text(text.replace("〜", wave_dash), for_irodori=for_irodori)
        == expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize("wave_dash", ["〜", "～"])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        ("“100円〜”", "'100円から'", "「100円から」"),
        ("‘100円〜’", "'100円から'", "「100円から」"),
        ("'100円〜'", "'100円から'", "「100円から」"),
        ('"100円〜"', "'100円から'", "「100円から「"),
        ("“100円〜”、相談可", "'100円から',相談可", "「100円から」、相談可"),
        ("‘10時〜’", "'10時から'", "「10時から」"),
        ("“27万〜”", "'27万から'", "「27万から」"),
        ("“ね〜”", "'ねー'", "「ねー」"),
        ("‘月1度〜の方針’", "'月1度ーの方針'", "「月1度ーの方針」"),
    ],
)
def test_normalize_text_open_ended_ranges_before_closing_quotes(
    text: str, expected: str, expected_irodori: str, wave_dash: str, for_irodori: bool
) -> None:
    """
    閉じ引用符の直前にある数量の波ダッシュが「から」と読まれ、語尾や助詞の前の長音はそのまま保持されることを確認する。
    """

    assert normalize_text(text.replace("〜", wave_dash), for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize("wave_dash", ["〜", "～"])
@pytest.mark.parametrize(
    ("text", "original_fragment", "expected_fragment", "expected_irodori_fragment"),
    [
        ("予算100円〜。", "100円〜", "100円から", "100円から"),
        ("27万円〜", "27万円〜", "27万円から", "27万円から"),
        ("1,000円〜", "1,000円〜", "1000円から", "1000円から"),
        ("受付10:30〜", "10:30〜", "十時30分から", "十時30分から"),
        ("2026/10/7〜", "2026/10/7〜", "2026年10月7日から", "2026年10月7日から"),
        ("¥1,000〜", "¥1,000〜", "1000円から", "1000円から"),
        ("2.5kg〜", "2.5kg〜", "2.5キログラムから", "二ー点五キログラムから"),
        ("50%〜", "50%〜", "50パーセントから", "50パーセントから"),
        ("5m/s〜", "5m/s〜", "5メートル毎秒から", "5メートル毎秒から"),
        ("60km/h〜", "60km/h〜", "60キロメートル毎時から", "60キロメートル毎時から"),
        ("3時間半〜", "3時間半〜", "3時間半から", "3時間半から"),
        ("1か月半〜", "1か月半〜", "1か月半から", "1か月半から"),
        ("27万〜", "27万〜", "27万から", "27万から"),
        ("1.5万〜", "1.5万〜", "1.5万から", "一点五万から"),
        ("二点五〜", "二点五〜", "二点五から", "二点五から"),
        ("“100円〜”", "100円〜", "100円から", "100円から"),
        ("‘100円〜’", "100円〜", "100円から", "100円から"),
        ("100円 〜", "100円 〜", "100円から", "100円から"),
        ("２７万〜", "２７万〜", "27万から", "27万から"),
        ("3h30m〜", "3h30m〜", "3時間30分から", "3時間30分から"),
        # 「Ver.」は「バージョン」に書き換えるので、版番号と範囲の「〜」まで含めて1つの区間になる
        ("Ver.2〜", "Ver.2〜", "バージョン2から", "バージョン2から"),
        ("iPhone15〜", "15〜", "15から", "15から"),
        ("第3〜", "3〜", "3から", "3から"),
    ],
)
def test_normalize_text_open_ended_range_details(
    text: str,
    original_fragment: str,
    expected_fragment: str,
    expected_irodori_fragment: str,
    wave_dash: str,
    for_irodori: bool,
) -> None:
    """
    数量から終点省略の波ダッシュまでが一つの数値区間として details に記録され、単位や日付の展開結果にも正しく対応付けられることを確認する。
    """

    text = text.replace("〜", wave_dash)
    result = normalize_text(text, for_irodori=for_irodori, return_details=True)
    assert result.text == normalize_text(text, for_irodori=for_irodori)
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == [
        (
            "number",
            original_fragment.replace("〜", wave_dash),
            expected_irodori_fragment if for_irodori is True else expected_fragment,
        )
    ]
    detail = result.details[0]
    assert text[detail.original_start : detail.original_end] == detail.original_text
    assert (
        result.text[detail.normalized_start : detail.normalized_end]
        == detail.normalized_text
    )


@pytest.mark.parametrize("for_irodori", [False, True])
def test_normalize_text_open_ended_range_details_preserve_context(
    for_irodori: bool,
) -> None:
    """
    文中に複数の範囲表現が含まれる場合にそれぞれが独立して記録され、助詞の前や数量ではない語尾の長音は範囲の区間から除外されることを確認する。
    """

    text = "予算100円〜、受付10:30〜。月1度～の方針、よろしく〜"
    result = normalize_text(text, for_irodori=for_irodori, return_details=True)
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == [("number", "100円〜", "100円から"), ("number", "10:30〜", "十時30分から")]
    for detail in result.details:
        assert text[detail.original_start : detail.original_end] == detail.original_text
        assert (
            result.text[detail.normalized_start : detail.normalized_end]
            == detail.normalized_text
        )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize("count", [30, 90])
def test_normalize_text_range_details_normalizes_linear_amount_of_text(
    monkeypatch: pytest.MonkeyPatch, count: int, for_irodori: bool
) -> None:
    """
    番号付きの語が多数繰り返された場合でも、区間情報の再変換量が入力長に対して線形に抑えられ、全区間の位置が正しく検証されることを確認する。
    """

    normalized_character_count = 0

    def normalize_fragment(text: str, for_irodori: bool = False) -> str:
        """
        正規化した文字数をカウントしながらテキストを正規化する。
        """

        nonlocal normalized_character_count
        normalized_character_count += len(text)
        return normalize_text(text, for_irodori=for_irodori)

    monkeypatch.setattr(japanese_normalizer, "normalize_text", normalize_fragment)
    text = "iPhone15〜、" * count
    result = normalize_text(text, for_irodori=for_irodori, return_details=True)
    expected = "アイフォン15から、" if for_irodori is True else "アイフォン15から,"
    assert result.text == expected * count
    assert len(result.details) == count
    for index, detail in enumerate(result.details):
        assert (detail.original_start, detail.original_end) == (
            index * 10 + 6,
            index * 10 + 9,
        )
        assert (detail.normalized_start, detail.normalized_end) == (
            index * 10 + 5,
            index * 10 + 9,
        )
        assert (detail.category, detail.original_text, detail.normalized_text) == (
            "number",
            "15〜",
            "15から",
        )
    assert normalized_character_count <= len(text) * 8


def test_normalize_text_ranges():
    """
    数量・時刻・日付・金額などの両端が指定された範囲表現において、波ダッシュ各種が「から」へ適切に正規化されることを確認する。
    """

    # 数値範囲
    assert normalize_text("1〜10") == "1から10"
    assert normalize_text("1~10") == "1から10"
    assert normalize_text("1～10") == "1から10"
    # 文字を含む範囲
    assert normalize_text("AからZ") == "AからZ"
    assert normalize_text("1から100まで") == "1から100まで"
    # 単位付きの範囲
    assert normalize_text("100m〜200m") == "100メートルから200メートル"
    assert normalize_text("1kg〜2kg") == "1キログラムから2キログラム"
    # 片側が頻度語でも、数字と単位から始まる波ダッシュは範囲として読む
    assert normalize_text("月1度～毎週") == "月1度から毎週"
    # 数字を伴わない語尾の波ダッシュは従来どおり長音として残す
    assert normalize_text("やった〜最高") == "やったー最高"
    # 見出しを囲む波ダッシュの末尾は、後続助詞との範囲を表さない
    assert normalize_text("～7か年戦略～の策定") == "ー7か年戦略ーの策定"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 数字+日本語単位から一般語へ続く範囲は「から」に展開する
        ("月1度～毎週", "月1度から毎週"),
        ("月1度〜毎週", "月1度から毎週"),
        ("月1度~毎週", "月1度から毎週"),
        ("年1回～月2回", "年1回から月2回"),
        ("週3日〜毎日", "週3日から毎日"),
        ("日1回～週1回", "日1回から週1回"),
        ("3か月～半年", "3か月から半年"),
        ("月1度～毎週の", "月1度から毎週の"),
        ("月1度～毎週です", "月1度から毎週です"),
        # 数値同士の範囲は従来の数値範囲変換 (`月1～3回` 内の `1～3`)
        ("月1～3回", "月1から3回"),
    ],
)
def test_normalize_text_mixed_number_ranges(text: str, expected: str) -> None:
    """
    数値と日本語単位から一般語へと続く波ダッシュの範囲表現が、長音記号にならず「から」へ正規化されることを確認する。
    """

    assert normalize_text(text) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 漢数字どうしの範囲は、算用数字と同じく「から」に書き換える
        ("六〜八メートル", "六から八メートル"),
        ("二十九〜三十一", "二十九から三十一"),
        ("四十五〜五十くらい", "四十五から五十くらい"),
        ("十〜十二月", "十から十二月"),
        ("一万五千〜二万円", "一万五千から二万円"),
        # 算用数字と漢数字が混ざった範囲も「から」に書き換える
        ("６〜十二月", "6から十二月"),
        # 左側に助数詞や単位が付いた範囲も「から」に書き換える
        ("一週間〜十日", "一週間から十日"),
        ("百円〜四百円", "百円から四百円"),
        ("1年～2年", "1年から2年"),
        ("100円～200円", "100円から200円"),
        # 漢数字の小数どうしの範囲も「から」に書き換える
        ("三十六．〇〜三十六．三度", "三十六点零から三十六点三度"),
        # 1ずつ違う1桁の数どうしも、時刻や日付の範囲と区別できないので「から」で読む
        ("六〜七メートル", "六から七メートル"),
        ("二〜三人", "二から三人"),
        ("２〜３分煮る", "2から3分煮る"),
        ("1〜2年", "1から2年"),
        ("５〜六人", "5から六人"),
        ("十五〜六年前", "十五から六年前"),
        ("１５〜６年前", "15から6年前"),
        ("受付時間は2〜3時", "受付時間は2から3時"),
        ("2〜3日まで休業", "2から3日まで休業"),
        # 符号や小数を含む数も、両端の数全体を範囲の端として「から」でつなぐ
        ("温度は−2〜3℃", "温度はマイナス2から3度"),
        ("二点三〜四点五度", "二点三から四点五度"),
        ("２〜４分", "2から4分"),
        ("0〜1の範囲", "0から1の範囲"),
        ("1kg〜2kg", "1キログラムから2キログラム"),
    ],
)
def test_normalize_text_kanji_number_ranges(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「六〜八メートル」「二十九〜三十一」のように漢数字を含む数の範囲の「〜」が長音の「ー」に変わり、
    「ロクーハチ」や「ニジューキュウー」のように前の母音を伸ばして読まれないよう、「から」に書き換えることを確認する。
    「受付時間は2〜3時」のような時刻の範囲を2つの時刻の列挙と取り違えないよう、1ずつ違う数どうしの範囲も「から」でつなぐことを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 英字1文字どうしの範囲は、「〜」を長音にせず「から」でつなぐ
        ("Ａ〜Ｄをピーシングし", "AからDをピーシングし"),
        ("ａ〜ｋをピーシング", "aからkをピーシング"),
        ("A～Z", "AからZ"),
        # 記号の付いた英字で終わる範囲も「から」でつなぐ
        ("Ａ〜Ｂ′をピーシング", "AからBプライムをピーシング"),
        # 順番が逆の英字や、「〜」で3つ以上つないだ英字は範囲ではなく手順の並びなので書き換えない
        ("Ｐ〜Ｄ〜Ｃ〜Ａサイクル", "PーDーCーAサイクル"),
        ("Ｄ〜Ａ", "DーA"),
        # 2文字以上の英字やローマ数字の並びは、1文字の英字の範囲として扱わない
        ("民法Ｉ〜ＩＩＩ", "民法IーIII"),
        # 空白を挟んで3つ以上つないだ並びも、範囲ではないので書き換えない
        ("A 〜 B 〜 C", "AーBーC"),
        # 英字の前後に数字が接していれば1文字の英字ではないので、範囲として扱わない
        ("A〜D4", "AーD4"),
        ("3A〜D", "3アンペアーD"),
        # 空白を挟んでも、独立した英字1文字どうしの範囲なら「から」でつなぐ
        ("Ａ 〜 Ｄをピーシング", "AからDをピーシング"),
    ],
)
def test_normalize_text_latin_letter_ranges(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「Ａ〜Ｄ」「ａ〜ｋ」のように英字1文字どうしを「〜」でつないだ範囲が、
    「〜」の長音化で「エーーディー」のように前の音を伸ばして読まれず、「エーからディー」と読まれることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 語尾・感嘆の波ダッシュは長音として残す
        ("よろしく〜", "よろしくー"),
        ("ね〜", "ねー"),
        ("よろしく～", "よろしくー"),
        ("ね～", "ねー"),
        ("やった〜最高", "やったー最高"),
        ("すごい〜!", "すごいー!"),
        ("おはよう〜", "おはようー"),
        ("〜おはよう", "ーおはよう"),
        # 見出しを囲む波ダッシュは長音のまま残す
        ("～7か年戦略～の策定", "ー7か年戦略ーの策定"),
        ("～重要～なお知らせ", "ー重要ーなお知らせ"),
        # 数字や漢数字が単語の一部でも、数量を表さない末尾の波ダッシュは長音として残す
        ("3回目〜", "3回目ー"),
        ("7か年戦略〜の策定", "7か年戦略ーの策定"),
        ("世界一〜", "世界一ー"),
        ("一緒〜", "一緒ー"),
        # 後続助詞が付く場合は範囲として読まない
        ("月1度～の方針", "月1度ーの方針"),
        ("月1度～を", "月1度ーを"),
        ("3人〜の予約", "3人ーの予約"),
        ("5月1日〜で受付", "5月1日ーで受付"),
        ("10時〜は営業", "10時ーは営業"),
        ("10時〜も営業", "10時ーも営業"),
        ("10時〜に変更", "10時ーに変更"),
        ("10時〜が対象", "10時ーが対象"),
        # 両側とも漢字やカタカナの語のときは、語の範囲や区切りなので「から」にも長音にもせず休止にする
        ("毎週～毎日", "毎週,毎日"),
        ("月～曜日", "月,曜日"),
        ("東京～大阪", "東京,大阪"),
    ],
)
def test_normalize_text_wave_dash_not_converted_to_range(
    text: str,
    expected: str,
    for_irodori: bool,
) -> None:
    """
    語尾の伸ばしや感情表現として使われる波ダッシュが、誤って範囲の「から」へ変換されず長音記号として保持されることを確認する。
    """

    if for_irodori is True:
        expected = expected.replace("!", "！").replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 「間」が続く語の範囲は、1つの句として「トーキョーオーサカカン」とつなげて読む
        ("東京〜大阪間", "東京大阪間", "東京大阪間"),
        ("東京～大阪間", "東京大阪間", "東京大阪間"),
        ("青山一丁目〜渋谷間", "青山一丁目渋谷間", "青山一丁目渋谷間"),
        # それ以外の語の範囲や経路、副題の区切りは、長音にせず休止にする
        ("パリ〜ロンドン", "パリ,ロンドン", "パリ、ロンドン"),
        (
            "パリ〜ロンドン〜ベルリン〜ペテルブルグ",
            "パリ,ロンドン,ベルリン,ペテルブルグ",
            "パリ、ロンドン、ベルリン、ペテルブルグ",
        ),
        ("セルリアン〜黒", "セルリアン,黒", "セルリアン、黒"),
        (
            "天使の歌声〜小児病棟の奇跡〜",
            "天使の歌声,小児病棟の奇跡ー",
            "天使の歌声、小児病棟の奇跡ー",
        ),
        # 1文字の語や小書きのかなで終わる語、「ン」「ッ」が続く語中の伸ばしは、従来どおり長音にする
        ("ダ〜メ", "ダーメ", "ダーメ"),
        ("ア〜ア", "アーア", "アーア"),
        ("キャ〜素敵", "キャー素敵", "キャー素敵"),
        ("ルネサ〜ンス", "ルネサーンス", "ルネサーンス"),
        ("ボ〜ッと", "ボーッと", "ボーッと"),
        ("すご〜い", "すごーい", "すごーい"),
        # 波ダッシュを長音にすると知っているカタカナの語になる語中の伸ばしは、2文字以上のカタカナの後でも長音にする
        (
            "スーパ〜マーケットで買う",
            "スーパーマーケットで買う",
            "スーパーマーケットで買う",
        ),
        ("スパ〜クリングワイン", "スパークリングワイン", "スパークリングワイン"),
        # 数の範囲は、従来どおり「から」で読む
        ("四十五〜五十", "四十五から五十", "四十五から五十"),
        ("100円〜400円の間で", "100円から400円の間で", "100円から400円の間で"),
    ],
)
def test_normalize_text_word_range_wave_dash(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「パリ〜ロンドン」「東京〜大阪間」のように漢字やカタカナの語どうしをつなぐ波ダッシュが、長音の「ー」にされて「パリーロンドン」と読まれないことを確認する。
    「間」が続く範囲は台帳の「トーキョーオーサカカン」のとおり1つの句につなげ、それ以外の語の範囲や副題の区切りは休止にする。
    「ダ〜メ」「ルネサ〜ンス」のような語中の伸ばしは、従来どおり長音にする。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 曜日・時刻・季節・旬・期間などの時を表す語どうしの範囲は「から」で読む
        ("月曜〜金曜", "月曜から金曜", "月曜から金曜"),
        ("（月曜〜金曜までの？）", "'月曜から金曜までの?'", "（月曜から金曜までの？）"),
        ("木〜日曜特売", "木曜から日曜特売", "木曜から日曜特売"),
        ("十一月中〜下旬", "十一月中から下旬", "十一月中から下旬"),
        (
            "土曜日は正午〜午後５時",
            "土曜日は正午から午後5時",
            "土曜日は正午から午後5時",
        ),
        ("前十一時半〜後３時", "前十一時半から後3時", "前十一時半から後3時"),
        ("秋〜冬のスキンケア", "秋から冬のスキンケア", "秋から冬のスキンケア"),
        (
            "十一月末〜年内発売予定",
            "十一月末から年内発売予定",
            "十一月末から年内発売予定",
        ),
        ("数ヵ月〜数年", "数ヵ月から数年", "数ヵ月から数年"),
        # 後ろの期間が「間」で終わっても、時を表す語どうしの範囲なので、区間の「間」としてつなげずに「から」で読む
        ("数日〜数週間かかる", "数日から数週間かかる", "数日から数週間かかる"),
        ("数ヵ月〜数年間", "数ヵ月から数年間", "数ヵ月から数年間"),
        (
            "数ヵ月〜数年間保存する",
            "数ヵ月から数年間保存する",
            "数ヵ月から数年間保存する",
        ),
        # 曜日1文字どうしの範囲は「月から金」だとコアが「ツキ」と読むので、「曜」を補う
        ("月〜金　営業", "月曜から金曜,営業", "月曜から金曜、営業"),
        # 時を表す語でないものや、後ろが時を表す語でないものは、従来どおり休止にする
        ("十二時〜＝橘ノ円", "十二時,イコール橘ノ円", "十二時、イコール橘ノ円"),
        ("パリ〜ロンドン", "パリ,ロンドン", "パリ、ロンドン"),
        ("毎週～毎日", "毎週,毎日", "毎週、毎日"),
        # 「間」が続く範囲は、従来どおり1つの句につなげる
        ("東京〜大阪間", "東京大阪間", "東京大阪間"),
    ],
)
def test_normalize_text_time_word_range_wave_dash(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「月曜〜金曜」「正午〜午後５時」「秋〜冬」のように時を表す語どうしをつなぐ波ダッシュが、休止ではなく「から」と読まれることを確認する。
    「月〜金」のような曜日1文字の範囲は、「月から金」だとコアが「月」を「ツキ」と読むので、「月曜から金曜」と「曜」を補う。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_mathematical():
    """
    四則演算子や等号・不等号などの各種数学記号が、数式文脈において適切な日本語の読みへ正規化されることを確認する。
    """

    # 数学記号
    assert normalize_text("∞") == "無限大"
    assert normalize_text("π") == "パイ"
    assert normalize_text("√4") == "ルート4"
    assert normalize_text("∛8") == "立方根8"
    assert normalize_text("∜16") == "四乗根16"
    assert normalize_text("∑") == "シグマ"
    assert normalize_text("∫") == "インテグラル"
    assert normalize_text("∬") == "二重積分"
    assert normalize_text("∭") == "三重積分"
    assert normalize_text("∮") == "周回積分"
    assert normalize_text("∯") == "面積分"
    assert normalize_text("∰") == "体積分"
    assert normalize_text("∂") == "パーシャル"
    assert normalize_text("∇") == "ナブラ"
    assert normalize_text("∝") == "比例"
    # Unicode の減算記号は英字変数の間でも消さずに読む
    assert normalize_text("c−a") == "cマイナスa"
    # 英単語と型番の内部では U+2212 がハイフンとして使われるため、減算読みを挿入しない
    assert (
        normalize_text("non−inertialcavitation") == "ノンイナーシャルキャビテーション"
    )
    assert normalize_text("ATX−S10") == "エーティーエックスS10"
    # キャレット表記の指数
    assert normalize_text("^4") == "の4乗"
    assert normalize_text("2^4") == "2の4乗"
    assert normalize_text("2 ^ 4") == "2の4乗"
    assert normalize_text("x^4") == "xの4乗"
    assert normalize_text("10^-3") == "10のマイナス3乗"
    assert normalize_text("10^+3") == "10のプラス3乗"
    assert normalize_text("10^−3") == "10のマイナス3乗"
    assert normalize_text("10^－3") == "10のマイナス3乗"
    assert normalize_text("10^ー3") == "10のマイナス3乗"
    # 数字指数ではない `^` は装飾・脚注の可能性があるため、指数表記として展開しない
    assert normalize_text("x^n") == "xn"
    assert normalize_text("x^") == "x"
    assert normalize_text("^n") == "n"
    assert normalize_text("注釈^4") == "注釈4"
    assert normalize_text("10^^3") == "10'3"
    assert normalize_text("10^^3", for_irodori=True) == "10'3"
    # 集合記号
    assert normalize_text("∈") == "属する"
    assert normalize_text("∉") == "属さない"
    assert normalize_text("∋") == "含む"
    assert normalize_text("∌") == "含まない"
    assert normalize_text("∪") == "和集合"
    assert normalize_text("∩") == "共通部分"
    assert normalize_text("⊂") == "部分集合"
    assert normalize_text("⊃") == "上位集合"
    assert normalize_text("⊄") == "部分集合でない"
    assert normalize_text("⊅") == "上位集合でない"
    assert normalize_text("⊆") == "部分集合または等しい"
    assert normalize_text("⊇") == "上位集合または等しい"
    assert normalize_text("∅") == "空集合"
    assert normalize_text("∖") == "差集合"
    assert normalize_text("∆") == "対称差"
    # 幾何記号
    assert normalize_text("∥") == "平行"
    assert normalize_text("⊥") == "垂直"
    assert normalize_text("∠") == "角"
    assert normalize_text("∟") == "直角"
    assert normalize_text("∡") == "測定角"
    assert normalize_text("∢") == "球面角"


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 数や変数に挟まれた等号は、数式として「イコール」と読む
        ("ｘ＝３", "xイコール3"),
        ("x = 3", "xイコール3"),
        ("Ｎ＝十九", "Nイコール十九"),
        ("３×三十＝九十時間", "3かける三十イコール九十時間"),
        # 演算子を含む語の式も、数式として「イコール」と読む
        ("面積＝縦×横", "面積イコール縦かける横"),
        ("利回り＝利益÷株価", "利回りイコール利益わる株価"),
        # 発信地と通信社、写真説明、人名と肩書きをつなぐ等号は読まず、読点で区切る
        ("ロンドン＝共同", "ロンドン,共同"),
        ("記念撮影する選手たち＝写真", "記念撮影する選手たち,写真"),
        ("佐藤一郎さん＝東京都＝が受賞した", "佐藤一郎さん,東京都,が受賞した"),
        # 外国人名の区切りの等号も、中黒と同じく読点で区切る
        ("ジャン＝ポール", "ジャン,ポール"),
        # 項目名と値、通貨の換算のような等号は、片側が数でも数式として読まない
        ("観衆＝三万人", "観衆,三万人"),
        ("１ドル＝百二十円台", "1ドル,百二十円台"),
        # 語と語を同一視するだけの等号は、演算子がなければ数式として読まない
        ("強気＝買いという判断", "強気,買いという判断"),
        # 行頭の「＊」や区切りのダッシュは演算子ではないので、項目名と値の等号は読まない
        ("＊別名＝七変化", "別名,七変化"),
        # 根号も数式の項として「イコール」と読む
        ("x=√2", "xイコールルート2"),
        # 減算や語どうしの割り算も、記号を読みに変える前の式で判定して「イコール」と読む
        ("差額＝x−y", "差額イコールxマイナスy"),
        ("速度＝距離／時間", "速度イコール距離/時間"),
        # 英数字や記号が絡んで数式と認識できない等号は、読点にせず「イコール」と読む
        ("カトマンズ＝ＡＰ共同", "カトマンズイコールエーピー共同"),
        ("１回−巨人＝３回まで", "1回-巨人イコール3回まで"),
        # 「＝＝」のように続く等号も、両側が日本語の語なら読点で区切る
        ("ジャワ原人＝＝が", "ジャワ原人,,が"),
        # 日付の「/」は語の割り算ではないので、公演の日付と会場をつなぐ等号は読点で区切る
        ("十二／二十八＝北九州芸術劇場", "十二/二十八,北九州芸術劇場"),
    ],
)
def test_normalize_text_equals_sign(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「ｘ＝３」「面積＝縦×横」のように数や変数、演算子の式に挟まれた等号は「イコール」と読み、
    「ロンドン＝共同」「選手たち＝写真」のように新聞で語をつなぐ等号は読まずに読点で区切ることを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 括弧と関数呼び出しも数式の項として「イコール」と読む
        ("f(x)=x", "f'x'イコールx", "f（x）イコールx"),
        ("x=(3)", "xイコール'3'", "xイコール（3）"),
        ("x = (a + b)", "xイコール'aプラスb'", "xイコール（aプラスb）"),
    ],
)
def test_normalize_text_equals_sign_with_brackets(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「f(x)=x」「x=(3)」のように括弧でくくった項や関数呼び出しに挟まれた等号も、
    語をつなぐ等号として読点にせず、数式の「イコール」と読むことを確認する。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_dates():
    """
    西暦・和暦・スラッシュ区切り・ドット区切りなどの各種日付表現が、適切な日本語の日付読みへ正規化されることを確認する。
    """

    # 様々な日付形式
    assert normalize_text("2024/01/01") == "2024年1月1日"
    assert normalize_text("2024-01-01") == "2024年1月1日"
    assert normalize_text("2024年01月01日") == "2024年1月1日"  # 0埋めを除去
    assert normalize_text("01/01") == "1月1日"
    assert normalize_text("1/1") == "1月1日"
    assert normalize_text("２０２４／０１／０１") == "2024年1月1日"  # 全角英数記号
    assert normalize_text("２０２４/０１/０１") == "2024年1月1日"  # 全角英数
    # 曜日付きの日付
    assert normalize_text("2024/01/01(月)") == "2024年1月1日月曜日"
    assert normalize_text("2024-01-01（火）") == "2024年1月1日火曜日"
    assert normalize_text("2024/01/01（月曜）") == "2024年1月1日'月曜'"
    assert normalize_text("2024/01/01（月曜日）") == "2024年1月1日'月曜日'"
    # 年月のみ
    assert normalize_text("2024/01") == "2024年1月"
    assert normalize_text("1930/9") == "1930年9月"
    assert normalize_text("1880/10") == "1880年10月"
    assert normalize_text("2081/12") == "2081年12月"
    assert normalize_text("2181/12") == "2181年12月"
    # 2桁年の自動補完 (50以上は1900年代、49以下は2000年代)
    assert normalize_text("98/01/01") == "1998年1月1日"
    assert normalize_text("24/01/01") == "2024年1月1日"
    # 和暦
    assert normalize_text("令和6年1月1日") == "令和6年1月1日"
    assert normalize_text("平成30年12月31日") == "平成30年12月31日"
    # 日付範囲
    assert normalize_text("1/1〜1/3") == "1月1日から1月3日"
    # 年月
    assert normalize_text("2024年1月") == "2024年1月"
    # 区切り文字のバリエーション
    assert normalize_text("2024.01.01") == "2024年1月1日"
    assert normalize_text("20240101") == "2024年1月1日"
    assert normalize_text("19640820") == "1964年8月20日"
    # 省略表記の和暦
    assert normalize_text("R6.1.1") == "令和6年1月1日"
    assert normalize_text("R6.01.01") == "令和6年1月1日"
    assert normalize_text("H31.4.30") == "平成31年4月30日"
    assert normalize_text("H31.04.30") == "平成31年4月30日"
    assert normalize_text("S64.1.7") == "昭和64年1月7日"
    assert normalize_text("S64.01.07") == "昭和64年1月7日"
    assert normalize_text("S47.12.31") == "昭和47年12月31日"
    # 零時チェック
    assert normalize_text("午前00時") == "午前零時"
    assert normalize_text("午後00時") == "午後零時"
    assert normalize_text("午前00時00分") == "午前零時"
    assert normalize_text("午後00時00分") == "午後零時"
    assert normalize_text("午前00時00分00秒") == "午前零時零分零秒"
    assert normalize_text("午後00時00分00秒") == "午後零時零分零秒"
    assert normalize_text("今日は0時に就寝します") == "今日は零時に就寝します"
    assert normalize_text("今日は00時に就寝します") == "今日は零時に就寝します"
    assert (
        normalize_text("今日は0時間勉強した") == "今日は0時間勉強した"
    )  # 変換されない
    assert normalize_text("1000時間勉強した") == "1000時間勉強した"  # 変換されない
    # 異常な日付
    assert (
        normalize_text("2024/13/01") == "十三ぶんの二千二十四/01"
    )  # 13月は異常値なので分数判定される
    assert (
        normalize_text("2024/01/32") == "2024年1月/32"
    )  # 32日は異常値なので年と月だけ変換される
    assert normalize_text("2024/02/30") == "2024年2月/30"  # 存在しない日付
    assert normalize_text("2024/00/00") == "2024/00/00"  # ゼロの月日

    # 追加のテストケース
    assert normalize_text("2024年5月8日 （月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日　（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年5月8日　　（月）") == "2024年5月8日月曜日"
    assert normalize_text("2024年05月08日　　（月）") == "2024年5月8日月曜日"
    assert normalize_text("08日　　（月）") == "8日月曜日"
    assert normalize_text("05/31　　（月）") == "5月31日月曜日"
    assert normalize_text("05/30　　（月）") == "5月30日月曜日"
    assert normalize_text("05/20　　（月）") == "5月20日月曜日"
    assert normalize_text("02/21　　（月）") == "2月21日月曜日"
    assert normalize_text("12/21　　（月）") == "12月21日月曜日"
    assert normalize_text("24/12/21　　（月）") == "2024年12月21日月曜日"
    assert normalize_text("24/02/21　　（月）") == "2024年2月21日月曜日"
    assert normalize_text("24/02/1　　（月）") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01　　（月）") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01(月)") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01 (月)") == "2024年2月1日月曜日"
    assert normalize_text("24/02/01 (火)") == "2024年2月1日火曜日"
    assert normalize_text("24/02/01 (水)") == "2024年2月1日水曜日"
    assert normalize_text("24/02/29 (木)") == "2024年2月29日木曜日"
    assert normalize_text("24/02/29 (金)") == "2024年2月29日金曜日"
    assert normalize_text("24/02/29 (土)") == "2024年2月29日土曜日"
    assert normalize_text("24/02/29 （日）") == "2024年2月29日日曜日"
    assert normalize_text("24/02/29") == "2024年2月29日"
    assert normalize_text("98/02/21（水）") == "1998年2月21日水曜日"
    assert normalize_text("98/02/21") == "1998年2月21日"
    assert normalize_text("98/02（水）") == "二ぶんの九十八水曜日"
    assert normalize_text("（水）") == "'水'"
    assert normalize_text("そうだ（水）") == "そうだ'水'"
    assert normalize_text("そうだ（水）に（）行こう") == "そうだ'水'に''行こう"
    assert normalize_text("そうだ（水）に行こう") == "そうだ'水'に行こう"
    assert normalize_text("01/01") == "1月1日"
    assert normalize_text("01月03") == "1月03"
    assert normalize_text("01月03日") == "1月3日"
    assert normalize_text("1/3") == "1月3日"
    assert normalize_text("01月03日") == "1月3日"
    assert normalize_text("95/01/03") == "1995年1月3日"
    assert normalize_text("01/03") == "1月3日"
    assert normalize_text("今年01/03にですね") == "今年1月3日にですね"
    assert normalize_text("今年12/3にですね") == "今年12月3日にですね"
    assert normalize_text("今年1/3にですね") == "今年1月3日にですね"
    assert normalize_text("今年9/13にですね") == "今年9月13日にですね"
    assert normalize_text("今年08-13にですね") == "今年08-13にですね"
    assert normalize_text("今年24-08-13にですね") == "今年2024年8月13日にですね"
    assert normalize_text("今年08/13にですね") == "今年8月13日にですね"
    assert normalize_text("今年24/12/03にですね") == "今年2024年12月3日にですね"
    assert normalize_text("今年12/03にですね") == "今年12月3日にですね"
    assert normalize_text("今年08/13にですね") == "今年8月13日にですね"
    assert normalize_text("20年には") == "20年には"
    assert normalize_text("05年には") == "05年には"
    assert normalize_text("85年には") == "85年には"
    assert normalize_text("05年01月には") == "05年1月には"
    assert normalize_text("09-01-03 24:34") == "2009年1月3日二十四時34分"
    assert normalize_text("87-01-03 24:34") == "1987年1月3日二十四時34分"
    assert normalize_text("明治45年07月30日") == "明治45年7月30日"
    assert normalize_text("大正15年12月25日") == "大正15年12月25日"
    assert normalize_text("昭和64年01月07日") == "昭和64年1月7日"
    assert normalize_text("平成31年04月30日") == "平成31年4月30日"
    assert normalize_text("令和05年12月31日") == "令和05年12月31日"
    assert normalize_text("西暦2024年1月1日") == "西暦2024年1月1日"
    assert normalize_text("AD2024") == "エーディー2024"
    assert normalize_text("BC356") == "ビーシー356"


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 元号に続く「15.4.1」は和暦の年月日なので、2桁の年を西暦の「2015年」に広げない
        ("大正15.4.1", "大正15年4月1日", "大正15年4月1日"),
        ("昭和64.1.7に", "昭和64年1月7日に", "昭和64年1月7日に"),
        ("平成元.1.8", "平成元年1月8日", "平成元年1月8日"),
        ("令和6.5.1", "令和6年5月1日", "令和6年5月1日"),
        ("大正１５．４．１", "大正15年4月1日", "大正15年4月1日"),
        # 元号のない「98.04.11」は、従来どおり2桁の年を西暦に広げる
        ("98.04.11", "1998年4月11日", "1998年4月11日"),
    ],
)
def test_normalize_text_era_dotted_dates(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「大正15.4.1」のように元号の後に点で区切った年月日が、年を西暦の「2015年」と取り違えずに「大正15年4月1日」と読まれることを確認する。
    「令和6.5.1」のように年が1桁でも、版番号の「6点5点1」ではなく和暦の日付にする。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 小数点の後の「01秒」は月日や時刻のゼロ埋めではないので、0を落とさない
        ("0.01秒差で負けた", "0.01秒差で負けた", "零点零一秒差で負けた"),
        ("9.05秒", "9.05秒", "九点零五秒"),
        # 月日や時刻のゼロ埋めは、従来どおり落とす
        ("05月01日", "5月1日", "5月1日"),
    ],
)
def test_normalize_text_zero_padding_keeps_decimal_digits(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「0.01秒」の小数点の後の「01」が、月日や時刻のゼロ埋めと取り違えられて「0.1秒」にならないことを確認する。
    0を落とすと、コアが「レーテンイチビョー」と10倍の値で読む。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 語や記号の後で数字の直前にある「-」「−」「－」は、負の数の符号なので「マイナス」と読む
        ("値は-0.5です", "値はマイナス0.5です", "値はマイナス零点五です"),
        (
            "前年比-3.2%",
            "前年比マイナス3.2パーセント",
            "前年比マイナス三点二パーセント",
        ),
        ("－１", "マイナス1", "マイナス1"),
        ("差は−0.3ポイント", "差はマイナス0.3ポイント", "差はマイナス零点三ポイント"),
        ("x = -3", "xイコールマイナス3", "xイコールマイナス3"),
        ("(-5)", "'マイナス5'", "（マイナス5）"),
        ("-5〜5", "マイナス5から5", "マイナス5から5"),
        # 数字や英字の後の「-」は範囲・引き算・型番の区切りなので、符号として読まない
        ("5-3=2", "5マイナス3イコール2", "5マイナス3イコール2"),
        ("1-2", "1-2", "1-2"),
        ("A-1", "A1", "A1"),
        # 箇条書きのように「-」の後に空白があるものは、符号ではない
        ("- 1つ目", "-1つ目", "-1つ目"),
        # 「序-2-1図」のように番号が「-」で続くものは、図表の番号の区切りなので符号にしない
        ("序-2-1図", "序-2-1図", "序-2-1図"),
    ],
)
def test_normalize_text_negative_numbers(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「値は-0.5」「前年比-3.2%」「－１」の負の数の符号が、休止として読み落とされずに「マイナス」と読まれることを確認する。
    「5-3」の引き算や「1-2」の範囲、「A-1」の型番のように数字や英字の後の「-」は、符号として読まない。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_time():
    """
    コロン区切りの時刻表記や午前・午後を伴う時刻表現が、適切な日本語の時刻読みへ正規化されることを確認する。
    """

    # 基本的な時刻表現
    assert normalize_text("9時3分") == "九時3分"
    assert normalize_text("9時4分") == "九時4分"
    assert normalize_text("9時6分") == "九時6分"
    assert normalize_text("9時12分") == "九時12分"
    assert normalize_text("9時22分") == "九時22分"
    assert normalize_text("9時30分") == "九時30分"
    assert normalize_text("14時00分") == "十四時"
    assert normalize_text("7時45分30秒") == "七時45分三十秒"
    # コロン区切りの時刻
    assert normalize_text("09:12") == "九時12分"
    assert normalize_text("09:30") == "九時30分"
    assert normalize_text("14:00") == "十四時"
    assert normalize_text("07:45:30") == "七時45分三十秒"
    # アスペクト比（時刻として解釈されない数値の組み合わせ）
    assert normalize_text("16:9") == "十六タイ九"
    # 午前・午後
    assert normalize_text("午前9時30分") == "午前九時30分"
    assert normalize_text("午後3時45分") == "午後三時45分"
    # 特殊な時刻
    assert normalize_text("0時0分") == "零時"
    assert normalize_text("24時00分") == "二十四時"
    assert normalize_text("25:00") == "二十五時"  # 25時までは許容
    assert normalize_text("30:00") == "三十タイ零"  # 30時はアスペクト比として解釈される
    # 秒以下の単位
    assert normalize_text("10時20分30.5秒") == "十時20分30.5秒"
    # 異常な時刻
    assert normalize_text("24:60") == "二十四時六十"  # 存在しない分
    assert normalize_text("00:00:60") == "零時零分六十"  # 存在しない秒
    # 27時台までは許容
    assert normalize_text("27:59:00") == "二十七時59分零秒"
    assert (
        normalize_text("28:00:00") == "二十八タイ零タイ零"
    )  # 28時はアスペクト比として解釈される

    # 追加のテストケース
    assert normalize_text("03:34に") == "三時34分に"
    assert normalize_text("03:3:564に") == "三タイ三タイ五百六十四に"
    assert normalize_text("03:34:54に") == "三時34分五十四秒に"
    assert normalize_text("03:03:03に") == "三時3分三秒に"
    assert normalize_text("03:03:62に") == "三時3分六十二に"
    assert normalize_text("03:03:60に") == "三時3分六十に"
    assert normalize_text("03:03:59に") == "三時3分五十九秒に"
    assert normalize_text("03:03:01に") == "三時3分一秒に"
    assert normalize_text("03:03:5に") == "三時3分五秒に"
    assert normalize_text("04:03") == "四時3分"
    assert normalize_text("4:3") == "四タイ三"
    assert normalize_text("16:3") == "十六タイ三"
    assert normalize_text("04:3") == "四タイ三"
    assert normalize_text("4:30") == "四時30分"
    assert normalize_text("04:30") == "四時30分"
    assert normalize_text("04時30分") == "四時30分"
    assert normalize_text("2024年05月08日 03時06分08秒") == "2024年5月8日三時6分八秒"
    assert normalize_text("2024年05月08日 00時00分00秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024:05:08 00:00:00") == "二千二十四タイ五タイ八零時零分零秒"
    assert normalize_text("2024/05/08 00:03:00") == "2024年5月8日零時3分零秒"
    assert normalize_text("2024/05/08 00:03") == "2024年5月8日零時3分"
    assert normalize_text("2024/05/08 00") == "2024年5月8日00"
    assert normalize_text("2024/05/08 0:30") == "2024年5月8日零時30分"
    assert normalize_text("2024年05月01日") == "2024年5月1日"
    assert normalize_text("2024/05/01") == "2024年5月1日"
    assert normalize_text("2024年05月01日") == "2024年5月1日"
    assert normalize_text("2024年05月01日 03時00分0秒") == "2024年5月1日三時零分零秒"
    assert normalize_text("2024年05月01日 03時0分0秒") == "2024年5月1日三時零分零秒"
    assert normalize_text("2024年05月08日 03時06分08秒") == "2024年5月8日三時6分八秒"
    assert normalize_text("2024年05月08日 00時00分30秒") == "2024年5月8日零時零分三十秒"
    assert normalize_text("2024年05月08日 00時00分０0秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024年05月08日 00時00分00秒") == "2024年5月8日零時零分零秒"
    assert normalize_text("2024年05月08日 00時00分") == "2024年5月8日零時"
    assert normalize_text("2024年05月08日 03時00分") == "2024年5月8日三時"
    assert normalize_text("2024年05月08日 03時00分0秒") == "2024年5月8日三時零分零秒"
    assert normalize_text("2024年05月08日 03時00分") == "2024年5月8日三時"
    assert normalize_text("2024年05月08日 03時01分") == "2024年5月8日三時1分"
    assert normalize_text("2024年05月08日 03:01") == "2024年5月8日三時1分"
    assert normalize_text("2024/05/08 03:01") == "2024年5月8日三時1分"
    assert normalize_text("2024/05/08 03:01:00") == "2024年5月8日三時1分零秒"
    assert normalize_text("2024/05/08 3時1分00") == "2024年5月8日三時1分00"
    assert normalize_text("2024/05/08 3時1分0秒") == "2024年5月8日三時1分零秒"
    assert normalize_text("2024/05/08 3時1分60秒") == "2024年5月8日三時1分六十"
    assert normalize_text("2024/05/08 3時1分59秒") == "2024年5月8日三時1分五十九秒"
    assert normalize_text("27:59:00") == "二十七時59分零秒"
    assert normalize_text("27:59") == "二十七時59分"
    assert normalize_text("27:59:00") == "二十七時59分零秒"
    assert normalize_text("28:59:00") == "二十八タイ五十九タイ零"
    assert normalize_text("28:59") == "二十八タイ五十九"
    assert normalize_text("03:03:03") == "三時3分三秒"
    assert normalize_text("30:1:03") == "三十タイ一タイ三"
    assert normalize_text("30:1:03:5") == "三十タイ一タイ三,5"
    assert normalize_text("30:1:03:5:07") == "三十タイ一タイ三,五時7分"
    assert normalize_text("1:3:4") == "一タイ三タイ四"
    assert normalize_text("1:3:4:5") == "一タイ三タイ四,5"
    assert normalize_text("1:3:4:5:6") == "一タイ三タイ四,五タイ六"
    assert normalize_text("1:3:4:5:6:7") == "一タイ三タイ四,五タイ六タイ七"
    assert normalize_text("1:3:4:5:6:7:8:9") == "一タイ三タイ四,五タイ六タイ七,八タイ九"
    assert normalize_text("16:9") == "十六タイ九"
    assert normalize_text("4:3") == "四タイ三"
    assert normalize_text("3:2") == "三タイ二"
    assert normalize_text("03:02") == "三時2分"
    assert normalize_text("03:2") == "三タイ二"
    assert normalize_text("3:02") == "三時2分"
    assert normalize_text("03:02") == "三時2分"
    assert normalize_text("03:2") == "三タイ二"
    assert normalize_text("03:02") == "三時2分"
    assert normalize_text("03:02:00") == "三時2分零秒"
    assert normalize_text("03:00:00") == "三時零分零秒"
    assert normalize_text("24:00:00") == "二十四時零分零秒"
    assert normalize_text("24:00") == "二十四時"
    assert normalize_text("27:59:00") == "二十七時59分零秒"
    # タイムスタンプの小数秒はマラソン計時のように秒の直後へ数字で続ける
    assert normalize_text("00:00:00.123") == "零時零分零秒123"
    assert normalize_text("00:34:05.101") == "零時34分5秒101"
    assert normalize_text("01:02:03.456") == "一時2分3秒456"
    assert normalize_text("03:04:05.6") == "三時4分5秒6"
    assert normalize_text("10:20:30.5") == "十時20分30秒5"
    assert normalize_text("00:00:00,123") == "零時零分零秒123"
    assert (
        normalize_text("経過時間は00:34:05.101です") == "経過時間は零時34分5秒101です"
    )
    # 時間省略の経過時間も同じ方針で読む
    assert normalize_text("34:05.101") == "34分5秒101"
    assert normalize_text("05:30.5") == "5分30秒5"
    assert normalize_text("00:05.101") == "零分5秒101"
    assert normalize_text("12:00 PM") == "十二時ピーエム"
    assert normalize_text("12:00 AM") == "十二時エーエム"
    assert normalize_text("深夜03時") == "深夜3時"
    assert normalize_text("未明04時") == "未明4時"
    assert normalize_text("早朝05時") == "早朝5時"
    assert normalize_text("夜09時") == "夜9時"
    assert normalize_text("正午") == "正午"
    assert normalize_text("正午12時") == "正午12時"
    assert normalize_text("0:00:00") == "零時零分零秒"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 連続時間表記 (`1h25m23s` 系)
        ("1h3m5s", "1時間3分5秒"),
        ("1h25m23s", "1時間25分23秒"),
        ("24d23h50m4s", "24日23時間50分4秒"),
        ("50m4s", "50分4秒"),
        ("1h3m", "1時間3分"),
        ("1h5s", "1時間5秒"),
        ("2h30m", "2時間30分"),
        ("45m30s", "45分30秒"),
        ("3d12h", "3日12時間"),
        ("3d12h5m", "3日12時間5分"),
        ("1H25M23S", "1時間25分23秒"),
        ("1.5h30m", "1.5時間30分"),
        # `m2` / `ms` への誤変換回帰 (`25m23s` が平方メートルにならないこと)
        ("25m23s", "25分23秒"),
        ("50m", "50メートル"),
        ("50m2", "50平方メートル"),
        ("50m3", "50立方メートル"),
        ("50ms", "50ミリ秒"),
        ("100m/h", "100メートル毎時"),
        # タイムスタンプ (`00:34:05.101` 系)
        ("00:34:05.101", "零時34分5秒101"),
        ("00:00:00.123", "零時零分零秒123"),
        ("00:00:00,123", "零時零分零秒123"),
        ("01:02:03.456", "一時2分3秒456"),
        ("03:04:05.6", "三時4分5秒6"),
        ("10:20:30.5", "十時20分30秒5"),
        ("1:02:03.040", "一時2分3秒040"),
        ("12:34:56.789", "十二時34分56秒789"),
        ("00:34:05", "零時34分五秒"),
        # 時間省略の経過時間 (`34:05.101` 系)
        ("34:05.101", "34分5秒101"),
        ("05:30.5", "5分30秒5"),
        ("00:05.101", "零分5秒101"),
        ("27:59.99", "27分59秒99"),
        ("90:00.001", "90分零秒001"),
        ("34:00.101", "34分零秒101"),
        ("34:05.000", "34分5秒000"),
        # 数値+単位から一般語へ続く範囲 (`月1度～毎週` 系)
        ("月1度～毎週", "月1度から毎週"),
        ("月1度〜毎週", "月1度から毎週"),
        ("年1回～月2回", "年1回から月2回"),
        ("週3日〜毎日", "週3日から毎日"),
        # Unicode 減算記号の変数間読み
        ("c−a", "cマイナスa"),
        ("x−y", "xマイナスy"),
        ("A−B", "AマイナスB"),
        # 型番・英単語内部では減算読みを挿入しない
        ("E2A", "E2A"),
        ("E2−A", "E2A"),
        ("ATX−S10", "エーティーエックスS10"),
        (
            "non−inertialcavitation",
            "ノンイナーシャルキャビテーション",
        ),
    ],
)
def test_normalize_text_duration_timestamp_and_related_patterns(
    text: str,
    expected: str,
) -> None:
    """
    連続時間表記、タイムスタンプ、経過時間、混合範囲、記号的減算の各パターンが適切に正規化されることを確認する。
    """

    assert normalize_text(text) == expected


def test_normalize_text_duration_timestamp_in_context() -> None:
    """
    連続時間表記やタイムスタンプが文中に埋め込まれている場合に、前後の文脈を壊さず正しく正規化されることを確認する。
    """

    assert normalize_text("経過1h25m23sでゴール") == "経過1時間25分23秒でゴール"
    assert (
        normalize_text("経過時間は00:34:05.101です") == "経過時間は零時34分5秒101です"
    )
    assert normalize_text("ラップは34:05.101でした") == "ラップは34分5秒101でした"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1h25m23s", "1時間25分23秒"),
        ("00:34:05.101", "零時34分5秒101"),
        ("34:05.101", "34分5秒101"),
        ("月1度～毎週", "月1度から毎週"),
    ],
)
def test_normalize_text_duration_timestamp_for_irodori(
    text: str,
    expected: str,
) -> None:
    """
    時間表記およびタイムスタンプの正規化が、Irodori-TTS 向けの経路でも同様に正しく行われることを確認する。
    """

    assert normalize_text(text, for_irodori=True) == expected


def test_normalize_text_duration_timestamp_negative_cases() -> None:
    """
    時間表記やタイムスタンプの正規化ルールが、比率や型番などの非時間表現に対して誤適用されないことを確認する。
    """

    # 両側が数字の範囲は従来の数値範囲変換を維持する
    assert normalize_text("100m〜200m") == "100メートルから200メートル"
    # 小数秒のない時刻・アスペクト比は従来どおり
    assert normalize_text("14:05") == "十四時5分"
    assert normalize_text("09:12") == "九時12分"
    assert normalize_text("16:9") == "十六タイ九"
    assert normalize_text("1920:1080") == "千九百二十タイ千八十"


def test_normalize_text_fractions():
    """
    日付と誤認されない文脈において、スラッシュ表記の分数が「○分の○」という日本語の分数読みへ正規化されることを確認する。
    """

    assert normalize_text("123/456") == "四百五十六ぶんの百二十三"
    assert normalize_text("1/100") == "百ぶんの一"
    assert normalize_text("13/32") == "三十二ぶんの十三"
    # 分数を含む文
    assert normalize_text("材料の2/30を使用した。") == "材料の三十ぶんの二を使用した."
    assert normalize_text("残り時間は1/40です。") == "残り時間は四十ぶんの一です."

    # 追加のテストケース
    assert normalize_text("16/9") == "九ぶんの十六"
    assert normalize_text("1/2") == "1月2日"  # 日付として解釈される
    assert normalize_text("1/32") == "三十二ぶんの一"
    assert normalize_text("9/16") == "9月16日"  # 日付として解釈される


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 計量の語や分量の助数詞と組む「1/2」は、日付ではなく分数として読む
        ("小さじ１／２", "小さじ二ぶんの一"),
        ("砂糖大さじ２分１を加える", "砂糖大さじ二ぶんの一を加える"),
        ("塩小さじ 4分3を量る", "塩小さじ四ぶんの三を量る"),
        ("小さじ12分10", "小さじ十二ぶんの十"),
        ("作業時間は2分1秒", "作業時間は2分1秒"),
        ("約１／２カップ", "約二ぶんの一カップ"),
        ("にんじん１／３本", "にんじん三ぶんの一本"),
        ("にんじん1/2本分", "にんじん二ぶんの一本分"),
        ("玉ねぎ１／２個", "玉ねぎ二ぶんの一個"),
        ("しめじ１／２パック", "しめじ二ぶんの一パック"),
        ("小さじ１と１／２", "小さじ1と二ぶんの一"),
        # 熟語と紛れない助数詞なら、後ろに「分」「弱」などの漢字が続いても分数として読む
        ("レモン汁１／２個分", "レモン汁二ぶんの一個分"),
        ("1/2カップ弱", "二ぶんの一カップ弱"),
        # 分量の文脈がない「1/2」は、従来どおり日付として読む
        ("１／２に開催", "1月2日に開催"),
        ("１／２から", "1月2日から"),
        # 「本」は直後が漢字でないときと「本分」のときだけ助数詞とし、「本番」「本日」「本社」は日付のまま読む
        # 「株」「合」は「株主」「合唱」のような熟語と紛れるので、助数詞として数えず日付のまま読む
        ("１／５本番", "1月5日本番"),
        ("１／５本日発売", "1月5日本日発売"),
        # 計量の語や助数詞との間に空白があっても分数として読む
        ("小さじ 1/2", "小さじ二ぶんの一"),
        ("1/2 カップ", "二ぶんの一カップ"),
        # 「契約」の「約」や「本社」「株主」「合唱」の先頭の字は計量の語や助数詞ではないので、日付のまま読む
        ("契約1/2締結", "契約1月2日締結"),
        ("1/2本社休業", "1月2日本社休業"),
        ("1/2株主総会", "1月2日株主総会"),
        ("1/2合唱団の公演", "1月2日合唱団の公演"),
        # 割合を表す語の後や、比べる語・増減・長さや束の単位が続く「1/2」も、日付ではなく分数として読む
        ("確率は1/2です", "確率は二ぶんの一です"),
        ("全体の1/3が賛成した", "全体の三ぶんの一が賛成した"),
        ("全体の１／４が賛成した", "全体の四ぶんの一が賛成した"),
        ("直径5/8インチのねじ", "直径八ぶんの五インチのねじ"),
        ("ほうれん草1/4束", "ほうれん草四ぶんの一束"),
        ("曲の2/3節を歌う", "曲の三ぶんの二節を歌う"),
        # 「節分」「節句」「節目」のように「節」で始まる語は助数詞ではないので、日付のまま読む
        ("開催日は2/3節分祭", "開催日は2月3日節分祭"),
        ("2/3節分の豆まき", "2月3日節分の豆まき"),
        ("3/3節句", "3月3日節句"),
        ("1/3節目の日", "1月3日節目の日"),
        ("2/3以上の賛成", "三ぶんの二以上の賛成"),
        ("1/10に減った", "十ぶんの一に減った"),
        # 割合や単位の手がかりがない「1/2」は、従来どおり日付として読む
        ("8/1から働く", "8月1日から働く"),
        ("明日6/9観に行く", "明日6月9日観に行く"),
        ("来週の6/9に会う", "来週の6月9日に会う"),
        ("1/2丁目", "1月2日丁目"),
        # 「½」のような分数の文字は、日付と紛れないので常に分数として読み、前に整数があれば「と」でつなぐ
        ("全体の½が賛成した", "全体の二ぶんの一が賛成した"),
        ("¼", "四ぶんの一"),
        ("1½カップ", "1と二ぶんの一カップ"),
    ],
)
def test_normalize_text_quantity_fractions(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「小さじ１／２」「１／３本」のように分量を表す「1/2」が「1月2日」と日付に読まれないよう、
    計量の語や分量の助数詞と組むときだけ「二ぶんの一」と分数に書き換えることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # かぎ針の号数は「3/0号」と書き、分母が0の分数ではない
        ("3/0号かぎ針", "3/0号かぎ針", "3/0号かぎ針"),
        ("かぎ針7/0号", "かぎ針7/0号", "かぎ針7/0号"),
        (
            "前端、えりぐり(縁編み)5/0号針",
            "前端,えりぐり'縁編み'5/0号針",
            "前端、えりぐり（縁編み）5/0号針",
        ),
        # 分母が0の分数は意味をなさないので、号数以外でも分数にしない
        ("5/0", "5/0", "5/0"),
        # 直後に「号」が続く番号は号数なので、分母が0でなくても分数にしない
        ("13/15号", "13/15号", "13/15号"),
        # 号数の文脈でない分数は、従来どおり「ぶんの」で読む
        ("16/9", "九ぶんの十六", "九ぶんの十六"),
        (
            "材料の2/30を使用した。",
            "材料の三十ぶんの二を使用した.",
            "材料の三十ぶんの二を使用した。",
        ),
        # 月日として成り立つ「2/3号」は、雑誌の日付の号として従来どおり日付で読む
        ("2/3号", "2月3日号", "2月3日号"),
        # 「2/2.5G」の「2.5」は小数なので、「2/2」を日付や分数にしない (コアは「ニ、ニーテンゴジー」と読む)
        ("2/2.5G", "2/2.5G", "2/二ー点五G"),
        (
            "２／２．５Ｇのネットワーク",
            "2/2.5Gのネットワーク",
            "2/二ー点五Gのネットワーク",
        ),
        ("1/2", "1月2日", "1月2日"),
    ],
)
def test_normalize_text_fraction_excludes_zero_denominator_and_gou_number(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    かぎ針の号数「3/0号」が「零ぶんの三号」と分数に読まれず、書き換えずにコアへ渡されて「サン、ゼロゴー」と読まれることを確認する。
    分母が0の「5/0」や、直後に「号」が続く「13/15号」も分数にせず、号数の文脈でない「16/9」は従来どおり「九ぶんの十六」と読む。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 浮動小数点で展開すると「六千十九垓九千九百九十九京…」と誤差の桁が出るので、10進数のまま展開する
        ("6.02e23", "六千二十垓", "六千二十垓"),
        (
            "アボガドロ定数は6.02e23です。",
            "アボガドロ定数は六千二十垓です.",
            "アボガドロ定数は六千二十垓です。",
        ),
        ("9.99e30", "九百九十九穣", "九百九十九穣"),
        ("N=6.02e23", "Nイコール六千二十垓", "Nイコール六千二十垓"),
        # 誤差の出ない指数も、従来どおり展開する
        ("1e-7メートル", "零点零零零零零零一メートル", "零点零零零零零零一メートル"),
        ("3.0E8m/s", "三億m/s", "三億m/s"),
        ("1.5e+3", "千五百", "千五百"),
        ("2e10", "二百億", "二百億"),
        ("1.23e-4", "零点零零零一二三", "零点零零零一二三"),
        # 小数の桁も浮動小数点を通さずに展開し、17桁目の「1」を丸めない
        (
            "1.0000000000000001e-1",
            "零点一零零零零零零零零零零零零零零零一",
            "零点一零零零零零零零零零零零零零零零一",
        ),
        # 読める桁数を超える指数や、10進数で表せない指数は、正規化を止めずに元の表記のまま残す
        ("1e9999999999999999999", "1e9999999999999999999", "1e9999999999999999999"),
        ("1e60", "1e60", "1e60"),
        # 英字やギリシャ文字の直後の「1e1」はベクトルの添字や識別子なので、指数として展開しない
        ("ξ1e1", "ξ1e1", "ξ1e1"),
        (
            "いまξ１ｅ１＋ξ２ｅ２の形",
            "いまξ1e1プラスξ2e2の形",
            "いまξ1e1プラスξ2e2の形",
        ),
        ("x1e1", "x1e1", "x1e1"),
    ],
)
def test_normalize_text_exponent_notation(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「6.02e23」のような指数表記が、浮動小数点の誤差なく「六千二十垓」と展開されることを確認する。
    「ξ1e1+ξ2e2」のように英字やギリシャ文字に続く「1e1」はベクトルの成分の添字なので、「ξ十」と指数に展開せずそのまま残し、コアに「クシーイチイーイチ」と読ませる。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_phone_numbers():
    """
    市外局番や携帯電話番号などの電話番号表記が、カタカナに書き換えられず、数字の組をハイフンでつないだ形でコアに渡されることを確認する。

    電話番号の1桁ずつの読み (「ゼロサン、イチニーサンヨン、…」)、組の間の休止、3桁の組の末尾の「2」「5」を伸ばさない読みや真ん中の0の「マル」は、
    コアの NJD が読みとアクセントを付ける。正規化がカタカナにすると、アクセントがカタカナの語として付け直されて崩れる。
    区切りのない携帯電話などの番号も、組の間に休止が入るようハイフンでつなぐ。
    """

    # --- 固定電話（ハイフン区切り） ---
    # 2桁市外局番（東京 03、大阪 06）: 0X-XXXX-XXXX
    assert normalize_text("03-1234-5678") == "03-1234-5678"
    assert normalize_text("06-9876-5432") == "06-9876-5432"
    # 3桁市外局番（横浜 045、名古屋 052、仙台 022）: 0XX-XXX-XXXX
    assert normalize_text("045-123-4567") == "045-123-4567"
    assert normalize_text("052-987-6543") == "052-987-6543"
    assert normalize_text("022-222-3333") == "022-222-3333"
    # 4桁市外局番（川崎 044、大分 097）: 0XXX-XX-XXXX
    assert normalize_text("044-12-3456") == "044-12-3456"
    assert normalize_text("097-56-7890") == "097-56-7890"
    # 5桁市外局番（地方小規模）: 0XXXX-X-XXXX
    assert normalize_text("0291-1-2345") == "0291-1-2345"

    # --- 携帯電話（ハイフン区切り）: 0X0-XXXX-XXXX ---
    assert normalize_text("080-4205-7491") == "080-4205-7491"
    assert normalize_text("090-1111-2222") == "090-1111-2222"
    assert normalize_text("070-9876-5432") == "070-9876-5432"
    # 060 は 2026年7月以降に割り当て予定の新プレフィックス
    assert normalize_text("060-1234-5678") == "060-1234-5678"

    # --- IP 電話: 050-XXXX-XXXX ---
    assert normalize_text("050-1234-5678") == "050-1234-5678"

    # --- フリーダイヤル: 0120-XXX-XXX ---
    assert normalize_text("0120-982-954") == "0120-982-954"
    assert normalize_text("0120-000-111") == "0120-000-111"

    # --- フリーコール: 0800-XXX-XXXX ---
    assert normalize_text("0800-123-4567") == "0800-123-4567"

    # --- ナビダイヤル: 0570-XXX-XXX ---
    assert normalize_text("0570-012-345") == "0570-012-345"

    # --- 文中の電話番号 ---
    assert (
        normalize_text("お電話は03-1234-5678までお願いします。")
        == "お電話は03-1234-5678までお願いします."
    )
    assert (
        normalize_text("03-1234-5678にお電話ください。")
        == "03-1234-5678にお電話ください."
    )
    assert (
        normalize_text("03-1234-5678 までお電話ください。")
        == "03-1234-5678,までお電話ください."
    )
    assert (
        normalize_text("電話番号は0120-456-789です。") == "電話番号は0120-456-789です."
    )
    assert normalize_text("携帯は090-1111-2222です。") == "携帯は090-1111-2222です."
    assert (
        normalize_text("受付時間内に050-9999-0000へご連絡ください。")
        == "受付時間内に050-9999-0000へご連絡ください."
    )

    # --- ハイフンなし電話番号（携帯・フリーダイヤル等の既知パターン） ---
    # 携帯（0X0 から始まる11桁）
    assert normalize_text("08042057491") == "080-4205-7491"
    assert normalize_text("09011112222") == "090-1111-2222"
    # フリーダイヤル（0120 + 6桁 = 10桁）
    assert normalize_text("0120982954") == "0120-982-954"
    # フリーコール（0800 + 7桁 = 11桁）
    assert normalize_text("08001234567") == "0800-123-4567"
    # ナビダイヤル（0570 + 6桁 = 10桁）
    assert normalize_text("0570012345") == "0570-012-345"
    # IP 電話（050 + 8桁 = 11桁）
    assert normalize_text("05012345678") == "050-1234-5678"
    # ハイフンなし携帯が文中に出現
    assert normalize_text("連絡先は09011112222です。") == "連絡先は090-1111-2222です."

    # --- 電話番号として判定されないケース ---
    # 固定電話のハイフンなしは市外局番の桁数が不明なので変換しない
    assert normalize_text("0312345678") == "0312345678"
    # 先頭が 0 でないハイフン区切り数字は電話番号ではない
    assert normalize_text("33-4") == "33-4"
    # 桁数が合わないもの
    assert normalize_text("03-12345-6789") == "03-12345-6789"
    # 普通の数字列は変換しない
    assert normalize_text("12345678") == "12345678"


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 中黒で区切った電話番号は、「電話番号」の見出しがないとコアが位取りで読むので、ハイフンにそろえる
        (
            "☎０９７４・６３・１５４１",
            "0974-63-1541",
            "0974-63-1541",
        ),
        (
            "同美術館（０４２８・７７・７０５１）。",
            "同美術館'0428-77-7051'.",
            "同美術館（0428-77-7051）。",
        ),
        (
            "事務局０３・５９５０・０７６５へ",
            "事務局03-5950-0765へ",
            "事務局03-5950-0765へ",
        ),
        # 市外局番を括弧で区切った電話番号も、ハイフンでつないでコアに1桁ずつ読ませる
        (
            "ファクス０９２（７１１）６２４２",
            "ファクス092-711-6242",
            "ファクス092-711-6242",
        ),
        # 電話番号の後に空白と数が続くときは読点で区切り、「'」と隣り合って番号のハイフンが消えないようにする
        (
            "☎０９６７（４４）０３３６　１泊２食",
            "0967-44-0336,1泊2食",
            "0967-44-0336、1泊2食",
        ),
        # フリーダイヤルの番号をプレフィックスの後だけで区切った表記も、組ごとにハイフンでつなぐ
        (
            "（０１２０・６０２７５３）で申し込み",
            "'0120-602-753'で申し込み",
            "（0120-602-753）で申し込み",
        ),
        # カタカナの社名の直後の番号も、部屋番号と取り違えずに電話番号として読む
        (
            "問ミズノ０１２０・３２０７９９。",
            "問ミズノ0120-320-799.",
            "問ミズノ0120-320-799。",
        ),
        # 電話の記号の直後なら、市外局番を省いた番号もハイフンでつないでコアに1桁ずつ読ませる
        (
            "毎月第１水曜、☎３３５０・６８４０。",
            "毎月第1水曜,3350-6840.",
            "毎月第1水曜、3350-6840。",
        ),
        # 〒の後の中黒で区切った郵便番号も、ハイフンでつないでコアに「ノ」と「マル」で読ませる
        (
            "〒１０４・８０１１　朝日新聞",
            "郵便番号104-8011,朝日新聞",
            "郵便番号104-8011、朝日新聞",
        ),
        (
            "〒０６０・００６３　札幌市",
            "郵便番号060-0063,札幌市",
            "郵便番号060-0063、札幌市",
        ),
        # 後ろの組が5桁以上あるものは郵便番号ではないので、郵便番号の形に変えない
        ("〒１２３・４５６７８", "郵便番号123,45678", "郵便番号123、45678"),
        # 建物名に続けて中黒やハイフンで並べた部屋番号は、電話番号ではないので従来どおり部屋番号として桁読みの漢数字にする
        (
            "石田ハイツ101・102",
            "石田ハイツ'一〇一,'一〇二",
            "石田ハイツ「一〇一、」一〇二",
        ),
        (
            "石田ハイツ101-102",
            "石田ハイツ'一〇一-102",
            "石田ハイツ'一〇一-102",
        ),
        # 中黒の直後の4桁の数字を、カタカナの建物名に続く部屋番号と取り違えない
        ("東京・2024年大会", "東京,2024年大会", "東京、2024年大会"),
        # 2つにしか区切られていない数や、0から始まらない数の並びは電話番号として読まない
        ("０３・５９５０", "03,5950", "03、5950"),
        ("１・２・３", "1,2,3", "1、2、3"),
        # 日付の日の並びや、年の範囲の並びに入った0から始まる数も、電話番号として読まない
        ("２００２年０４月０７・１４日", "2002年4月07,14日", "2002年4月07、14日"),
        ("２００３－０５・０７－１０年", "2003-05,07-10年", "2003-05、07-10年"),
        # 電話の記号がない4桁どうしの並びは、年の並びなどと区別できないので電話番号として読まない
        ("１９９８・２００２年", "1998,2002年", "1998、2002年"),
        ("１（２３４）５６７８", "1'234'5678", "1（234）5678"),
    ],
)
def test_normalize_text_dotted_and_parenthesized_phone_numbers(
    text: str, expected: str, expected_irodori: str, for_irodori: bool
) -> None:
    """
    「０９７４・６３・１５４１」「０９２（７１１）６２４２」のように中黒や括弧で区切った電話番号が、
    区切りごとに「ロクジューサン」のような位取りの数や部屋番号の読みに崩れず、
    ハイフン区切りの電話番号と同じ形でコアに渡されることを確認する。
    """

    if for_irodori is True:
        expected = expected_irodori
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 英字の見出しと空白の後の電話番号も、組をつなぐハイフンを残す
        ("TEL 03-1234-5678", "テル03-1234-5678"),
        ("FAX 06-9876-5432", "ファックス06-9876-5432"),
        ("Tel 03-1234-5678", "テル03-1234-5678"),
        ("Phone 03-1234-5678", "フォン03-1234-5678"),
        ("TEL 03・1234・5678", "テル03-1234-5678"),
        ("TEL 0120-123-456", "テル0120-123-456"),
        ("TEL 090-1234-5678", "テル090-1234-5678"),
        # 1文に見出しと番号の組が2つあっても、どちらの番号も組を保つ
        (
            "Fax 03-1234-5678 TEL 03-1234-5679",
            "ファックス03-1234-5678,テル03-1234-5679",
        ),
        ("〒100-0001 TEL 03-1234-5678", "郵便番号100-0001,テル03-1234-5678"),
        # 空白のない「TEL03-1234-5678」と、コロンで区切った見出しは従来どおり
        ("TEL03-1234-5678", "テル03-1234-5678"),
        ("TEL:03-1234-5678", "テル,03-1234-5678"),
    ],
)
def test_normalize_text_phone_numbers_after_english_words(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「TEL 03-1234-5678」のように英字の見出しの後に空白を挟んで書いた電話番号が、
    英単語のカタカナ変換で「TEL 03」と1つの語にまとめられて組のハイフンを失い、「テル0312345678」と1つの数としてコアに渡されないことを確認する。
    組のハイフンが残れば、コアが1桁ずつの読みと組の間の休止を付ける。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


def test_normalize_text_postal_codes():
    """
    「〒」記号や「3 桁-4 桁」の郵便番号表記が、カタカナに書き換えられず、「郵便番号」の見出しと数字の組をハイフンでつないだ形でコアに渡されることを確認する。

    郵便番号の1桁ずつの読み、ハイフンの「ノ」、前 3 桁の真ん中の0の「マル」(「304」は「サンマルヨン」) は、コアの NJD が読みとアクセントを付ける。
    「〒」記号は「郵便番号」と書き、コアが郵便番号と判定する手がかりにする。
    """

    # --- 〒 付き郵便番号 ---
    assert normalize_text("〒304-0002") == "郵便番号304-0002"
    assert normalize_text("〒100-0001") == "郵便番号100-0001"
    assert normalize_text("〒984-0054") == "郵便番号984-0054"
    assert normalize_text("〒802-0838") == "郵便番号802-0838"
    # 〒 の後にスペースがある場合
    assert normalize_text("〒 304-0002") == "郵便番号304-0002"

    # --- 〒 なしの郵便番号（3桁-4桁は郵便番号として推定） ---
    assert normalize_text("304-0002") == "304-0002"
    assert normalize_text("100-0001") == "100-0001"
    assert normalize_text("985-0054") == "985-0054"

    # --- 文中の郵便番号 ---
    assert (
        normalize_text("〒304-0002 茨城県下妻市今泉")
        == "郵便番号304-0002,茨城県下妻市今泉"
    )
    assert normalize_text("郵便番号は304-0002です。") == "郵便番号は304-0002です."

    # --- 郵便番号と住所の組み合わせ ---
    assert (
        normalize_text("〒304-0002 茨城県下妻市今泉613-6")
        == "郵便番号304-0002,茨城県下妻市今泉613の6"
    )

    # --- 全角数字の郵便番号 ---
    assert normalize_text("〒３０４−０００２") == "郵便番号304-0002"

    # --- 郵便番号として判定されないケース ---
    # 4桁-3桁は郵便番号ではない
    assert normalize_text("1234-567") == "1234-567"
    # 3桁-3桁は郵便番号ではない（ハイフン区切りなので分数にもならない）
    assert normalize_text("123-456") == "123-456"
    # 2桁-4桁は郵便番号ではない
    assert normalize_text("12-3456") == "12-3456"


def test_normalize_text_addresses():
    """
    地名に続く住所番地表記において、ハイフンの「の」への置換や部屋番号の書き分けなど、住所文脈に即した表記へ正規化されることを確認する。

    漢字地名の直後にある「数字-数字(-数字)(-数字)」パターンを住所番地として検出する。
    ハイフンは「の」に変換される。
    各要素のうち 2 桁以下はそのまま（pyopenjtalk が通常読み）とし、4 要素目が 3 桁以上の場合は部屋番号として扱う。
    部屋番号の読みとアクセントはコアに任せ、正規化はカナを作らずに表記だけを書き分ける:
      3 桁の部屋番号は、数字のままだとコアが位取りで読むので、桁読みの漢数字にする。
      例: 409→四〇九（コアは「ヨンマルキュー」と読む）、041→〇四一。
      4 桁以上の部屋番号は位取りで読む方が短く自然なので、数字のまま渡す。
      例: 1203→1203（コアは「センニヒャクサン」と読む）。
    号室の番号は、数字のままでもコアが 3 桁を桁読みするので、桁数にかかわらず数字のまま渡す。

    これらの規則に基づき、各種住所表記が適切に読み上げ用テキストへ正規化されることを確認する。
    """

    # --- 地番形式の住所（地番-枝番: 2要素） ---
    # 地名は実在するが番地の組み合わせは架空
    assert normalize_text("茨城県下妻市今泉613-6") == "茨城県下妻市今泉613の6"
    assert (
        normalize_text("福岡県北九州市小倉南区石田町399-18")
        == "福岡県北九州市小倉南区石田町399の18"
    )
    assert normalize_text("静岡県焼津市岡当目588-15") == "静岡県焼津市岡当目588の15"
    assert normalize_text("静岡県静岡市葵区千代362-17") == "静岡県静岡市葵区千代362の17"
    assert normalize_text("奈良県桜井市鹿路341-18") == "奈良県桜井市鹿路341の18"
    assert normalize_text("福島県二本松市在師29-7") == "福島県二本松市在師29の7"
    assert (
        normalize_text("宮城県仙台市若林区裏柴田町36-1")
        == "宮城県仙台市若林区裏柴田町36の1"
    )
    assert (
        normalize_text("富山県高岡市福岡町加茂784-4") == "富山県高岡市福岡町加茂784の4"
    )
    # 「條」は異体字変換で「条」に変換される
    assert normalize_text("奈良県五條市野原東2-889-2") == "奈良県五条市野原東2の889の2"
    assert (
        normalize_text("千葉県長生郡長南町棚毛51-13") == "千葉県長生郡長南町棚毛51の13"
    )

    # --- 住居表示形式の住所（丁目-番-号: 3要素） ---
    assert normalize_text("東京都港区六本木1-2-3") == "東京都港区六本木1の2の3"
    assert normalize_text("大阪府大阪市北区梅田3-1-3") == "大阪府大阪市北区梅田3の1の3"
    assert normalize_text("赤坂9-7-1") == "赤坂9の7の1"
    assert normalize_text("六本木1-2-3") == "六本木1の2の3"

    # --- 住居表示 + 部屋番号（4要素目が3桁なら桁読みの漢数字） ---
    assert normalize_text("赤坂1-2-3-609") == "赤坂1の2の3の六〇九"
    # 0 の位置によらず、3桁の部屋番号は桁読みの漢数字にする
    assert normalize_text("赤坂1-2-3-410") == "赤坂1の2の3の四一〇"
    assert normalize_text("赤坂1-2-3-041") == "赤坂1の2の3の〇四一"
    assert normalize_text("赤坂1-2-3-102") == "赤坂1の2の3の一〇二"
    assert normalize_text("赤坂1-2-3-304") == "赤坂1の2の3の三〇四"
    # 4桁の部屋番号
    assert normalize_text("赤坂1-2-3-1001") == "赤坂1の2の3の1001"
    assert normalize_text("赤坂1-2-3-1409") == "赤坂1の2の3の1409"
    # 4要素目でも2桁以下は通常読み（枝番として扱う）
    assert normalize_text("赤坂1-2-3-40") == "赤坂1の2の3の40"

    # --- マンション・アパート名 + 号室 ---
    # 号室の番号は数字のまま渡し、建物名の直後の番号は3桁なら桁読みの漢数字にする
    assert (
        normalize_text("茨城県下妻市今泉613-6 コーポ今泉301号室")
        == "茨城県下妻市今泉613の6,コーポ今泉'301号室"
    )
    assert normalize_text("石田ハイツ101") == "石田ハイツ'一〇一"
    assert normalize_text("グリーンコート赤坂409号室") == "グリーンコート赤坂'409号室"
    # 「号」のみの表記
    # 〇〇 は数値コンテキスト外のため「マルマル」に変換される
    assert normalize_text("〇〇マンション205号") == "マルマルマンション'二〇五号"
    # 2桁の号室は通常読み（pyopenjtalk に任せる）
    assert normalize_text("〇〇荘12号室") == "マルマル荘12号室"
    # 号室は文中でも変換する
    assert (
        normalize_text("お客様は309号室にお住まいなのですね。")
        == "お客様は'309号室にお住まいなのですね."
    )
    # 「号」は住所文脈がある場合のみ変換し、列車番号などは変換しない
    assert normalize_text("こだま309号が発車します") == "こだま309号が発車します"

    # --- 市区町村名などを省略した住所表記 ---
    # 4要素の standalone（2-11-3-309）はバージョン番号等と区別がつかないため変換しない
    assert normalize_text("2-11-3-309") == "2-11-3-309"
    # 3要素 + 空白 + 数字: 接尾辞（号/号室）なしは曖昧なため変換しない
    assert normalize_text("2-11-3 309") == "2-11-3'309"
    # 3要素 + 空白 + 号/号室: 接尾辞ありの場合のみ変換する
    assert normalize_text("2-11-3 309号") == "2の11の3の三〇九号"
    assert normalize_text("2-11-3 1205号室") == "2の11の3の1205号室"
    # 文頭以外でも、住所文脈（明示語彙）があれば standalone 住所を変換する
    assert normalize_text("住所は2-11-3 309号です") == "住所は2の11の3の三〇九号です"
    assert normalize_text("プラウド武蔵小杉1205号室") == "プラウド武蔵小杉'1205号室"
    # 住所 + 建物固有名詞 + 号 / 号室
    assert (
        normalize_text(
            "神奈川県川崎市幸区鹿島田3-105 パークハイム鹿島田スカイタワー 809号"
        )
        == "神奈川県川崎市幸区鹿島田3の105,パークハイム鹿島田スカイタワー'八〇九号"
    )
    assert (
        normalize_text(
            "神奈川県川崎市幸区鹿島田3-105 パークハイム鹿島田スカイタワー 809号室"
        )
        == "神奈川県川崎市幸区鹿島田3の105,パークハイム鹿島田スカイタワー'809号室"
    )
    # 県や市を省略しても問題なく変換される
    assert (
        normalize_text("川崎市幸区鹿島田3-105 パークハイム鹿島田スカイタワー 809号室")
        == "川崎市幸区鹿島田3の105,パークハイム鹿島田スカイタワー'809号室"
    )
    assert (
        normalize_text("高津区溝口2-34-5 パークタワー溝口 1009号室")
        == "高津区溝口2の34の5,パークタワー溝口'1009号室"
    )

    # --- 文中の住所 ---
    assert (
        normalize_text("住所は奈良県桜井市鹿路341-18です。")
        == "住所は奈良県桜井市鹿路341の18です."
    )
    assert (
        normalize_text("東京都港区六本木1-2-3にあります。")
        == "東京都港区六本木1の2の3にあります."
    )
    assert (
        normalize_text(
            "福岡県北九州市小倉南区石田町399-18 石田ハイツ101にお届けします。"
        )
        == "福岡県北九州市小倉南区石田町399の18,石田ハイツ'一〇一にお届けします."
    )

    # --- 「丁目」「番地」「号」表記は既存処理のまま通過する ---
    assert normalize_text("六本木3丁目10番4号") == "六本木3丁目10番4号"
    assert normalize_text("下妻市今泉613番地6") == "下妻市今泉613番地6"

    # --- 住所として判定されないケース ---
    # 漢字地名の直後でなければ住所とみなさない
    assert normalize_text("1-2-3") == "1-2-3"
    # iPhone 11 のように英単語+11までの数字パターンでは原則英語読みするようにしているが、1-2 のように直後にハイフン+数字が来る場合に
    # 英語読みするとは思えないので、英語読み変換の対象外となるべき
    assert normalize_text("ABC1-2-3") == "エービーシー1-2-3"
    # 住所の番地が1要素のみ（ハイフンなし）は住所変換対象外
    assert normalize_text("六本木7") == "六本木7"


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 図表番号のハイフンは住所の「の」にせず、コアに「ズヒョーイチ、ゴ、ハチ」と区切って読ませる
        ("図表1-5-8", "図表1-5-8", "図表1-5-8"),
        ("(図表1-5-8)。", "'図表1-5-8'.", "（図表1-5-8）。"),
        (
            "図表1-5-8　事業用借地権の利用目的",
            "図表1-5-8,事業用借地権の利用目的",
            "図表1-5-8、事業用借地権の利用目的",
        ),
        ("図1-2", "図1-2", "図1-2"),
        ("表3-1-2に示す", "表3-1-2に示す", "表3-1-2に示す"),
        ("式2-3", "式2-3", "式2-3"),
        ("第2節1-3", "第2節1-3", "第2節1-3"),
        ("第3章1-2", "第3章1-2", "第3章1-2"),
        # 地名の後の番地は、従来どおり住所としてハイフンを「の」にする
        ("東京都港区赤坂1-2-3", "東京都港区赤坂1の2の3", "東京都港区赤坂1の2の3"),
        (
            "奈良県桜井市鹿路341-18",
            "奈良県桜井市鹿路341の18",
            "奈良県桜井市鹿路341の18",
        ),
    ],
)
def test_normalize_text_figure_and_section_numbers_are_not_addresses(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「図表1-5-8」「表3-1-2」「第2節1-3」のように図・表・式・節・章に続く番号が、漢字の直後の「数字-数字」という形だけで住所と判定されて「図表1の5の8」と読まれないことを確認する。
    図表番号はハイフンのまま渡すと、コアが「ズヒョーイチ、ゴ、ハチ」と番号ごとに区切って読む。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 号室の番号はカナにせず数字のまま渡し、3桁はコアが「ハチマルゴゴーシツ」と桁読み、4桁は「センイチゴーシツ」と位取りで読む
        (
            "1001号室の方へお届けものです。",
            "'1001号室の方へお届けものです.",
            "'1001号室の方へお届けものです。",
        ),
        (
            "ホテルの部屋は805号室でした。",
            "ホテルの部屋は'805号室でした.",
            "ホテルの部屋は'805号室でした。",
        ),
        (
            "グリーンコート赤坂409号室",
            "グリーンコート赤坂'409号室",
            "グリーンコート赤坂'409号室",
        ),
        (
            "お客様は309号室にお住まいなのですね。",
            "お客様は'309号室にお住まいなのですね.",
            "お客様は'309号室にお住まいなのですね。",
        ),
        (
            "プラウド武蔵小杉1205号室",
            "プラウド武蔵小杉'1205号室",
            "プラウド武蔵小杉'1205号室",
        ),
        (
            "茨城県下妻市今泉613-6 コーポ今泉301号室",
            "茨城県下妻市今泉613の6,コーポ今泉'301号室",
            "茨城県下妻市今泉613の6、コーポ今泉'301号室",
        ),
        ("2-11-3 1205号室", "2の11の3の1205号室", "2の11の3の1205号室"),
        ("2-11-3 309号室", "2の11の3の309号室", "2の11の3の309号室"),
        # 住所の文脈の「号」は、数字のままだとコアが「サンビャクキューゴー」と位取りで読むので、3桁は桁読みの漢数字にする
        ("2-11-3 309号", "2の11の3の三〇九号", "2の11の3の三〇九号"),
        (
            "住所は2-11-3 309号です",
            "住所は2の11の3の三〇九号です",
            "住所は2の11の3の三〇九号です",
        ),
        (
            "港区赤坂1-2-3 パークハイム309号に届けてください。",
            "港区赤坂1の2の3,パークハイム'三〇九号に届けてください.",
            "港区赤坂1の2の3、パークハイム'三〇九号に届けてください。",
        ),
        (
            "マンション205号の住人です。",
            "マンション'二〇五号の住人です.",
            "マンション'二〇五号の住人です。",
        ),
        (
            "サンハイツ305号に住んでいます。",
            "サンハイツ'三〇五号に住んでいます.",
            "サンハイツ'三〇五号に住んでいます。",
        ),
        (
            "神奈川県川崎市幸区鹿島田3-105 パークハイム鹿島田スカイタワー 809号",
            "神奈川県川崎市幸区鹿島田3の105,パークハイム鹿島田スカイタワー'八〇九号",
            "神奈川県川崎市幸区鹿島田3の105、パークハイム鹿島田スカイタワー'八〇九号",
        ),
        # 4桁の号は、数字のままでもコアが「センキューゴー」と位取りで短く読むので、数字のまま渡す
        (
            "神奈川県川崎市幸区鹿島田3-105 パークハイム鹿島田スカイタワー 1009号",
            "神奈川県川崎市幸区鹿島田3の105,パークハイム鹿島田スカイタワー'1009号",
            "神奈川県川崎市幸区鹿島田3の105、パークハイム鹿島田スカイタワー'1009号",
        ),
        # 建物名の直後の番号は、正規化が部屋番号と判定した番号なので、4桁までを桁読みの漢数字にする
        ## 数字のまま渡すと、コアは号室の文脈を知らないので「サンゼンニヒャクヨンジューハチ」と位取りで読む
        (
            "送り先は石田ハイツ101です。",
            "送り先は石田ハイツ'一〇一です.",
            "送り先は石田ハイツ'一〇一です。",
        ),
        ("石田ハイツ1205", "石田ハイツ'一二〇五", "石田ハイツ'一二〇五"),
        (
            "石田ハイツ3248に住む",
            "石田ハイツ'三二四八に住む",
            "石田ハイツ'三二四八に住む",
        ),
        # 住所の4要素目は、3桁は桁読みの漢数字、4桁は数字のままにする
        (
            "石田ハイツ101・102",
            "石田ハイツ'一〇一,'一〇二",
            "石田ハイツ「一〇一、」一〇二",
        ),
        (
            "東京都港区赤坂1-2-3-405",
            "東京都港区赤坂1の2の3の四〇五",
            "東京都港区赤坂1の2の3の四〇五",
        ),
        (
            "東京都港区赤坂1-2-3-1205",
            "東京都港区赤坂1の2の3の1205",
            "東京都港区赤坂1の2の3の1205",
        ),
        # 住所と並ぶ電話番号は、ハイフンを住所の「の」にせず、電話番号の形のままコアに渡す
        (
            "東京都港区赤坂1-2-3-405 電話03-1234-5678",
            "東京都港区赤坂1の2の3の四〇五,電話03-1234-5678",
            "東京都港区赤坂1の2の3の四〇五、電話03-1234-5678",
        ),
        (
            "港区赤坂1-2-3 パークハイム309号 電話03-1234-5678",
            "港区赤坂1の2の3,パークハイム'三〇九号電話03-1234-5678",
            "港区赤坂1の2の3、パークハイム'三〇九号電話03-1234-5678",
        ),
        # 2桁の号室、号線、号車、住所の文脈のない列車の号は書き換えない
        ("〇〇荘12号室", "マルマル荘12号室", "マルマル荘12号室"),
        ("国道246号線", "国道246号線", "国道246号線"),
        ("5号車", "5号車", "5号車"),
        ("こだま309号", "こだま309号", "こだま309号"),
        (
            "東京都港区六本木1-2-3に届いた。こだま309号で帰る",
            "東京都港区六本木1の2の3に届いた.こだま309号で帰る",
            "東京都港区六本木1の2の3に届いた。こだま309号で帰る",
        ),
    ],
)
def test_normalize_text_room_numbers_are_left_to_core_reading(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    号室・住所の文脈の号・建物名の直後の番号を、正規化がカナ (「イチゼロゼロイチ」など) に書き換えないことを確認する。
    カナにすると MeCab の未知語になってアクセントが崩れるので、読みはコアに任せ、正規化は桁読みか位取りかを書き方で示すだけにする。
    号室は数字のまま渡せばコアが3桁を桁読み、4桁を位取りで読む。
    号と建物名の直後の番号は、数字のままだと3桁が位取りで読まれるので、3桁だけ「三〇九」のように桁読みの漢数字にする。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 鉄道車両の「形式-番号」は、形式を位取り、ハイフンを「の」、車両番号を桁読みで読む
        ("モハ205-3248が走る", "モハ205の三二四八が走る", "モハ205の三二四八が走る"),
        ("クハ211-3001", "クハ211の三〇〇一", "クハ211の三〇〇一"),
        ("キハ40-2001", "キハ40の二〇〇一", "キハ40の二〇〇一"),
        # 形式だけのときも、建物名の後の部屋番号と取り違えずに位取りで読む
        ("モハ205の車内", "モハ205の車内", "モハ205の車内"),
        # 車両の形式でないカタカナの後の番号は、従来どおり部屋番号として扱う
        ("石田ハイツ101", "石田ハイツ'一〇一", "石田ハイツ'一〇一"),
    ],
)
def test_normalize_text_railway_car_numbers(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「モハ205-3248」のような鉄道車両の形式と車両番号が、郵便番号の「205-3248」や部屋番号の「二〇五」と読まれず、
    「モハニヒャクゴノサンニーヨンハチ」と読まれる書き方になることを確認する。
    形式番号は位取り、車両番号は桁読みで読むことが多い。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_room_number_digit_patterns():
    """
    住所の4要素目や建物名の後の部屋番号が、桁数と接尾辞に応じて読み方の決まる表記へ正規化されることを確認する。

    部屋番号の読みとアクセントはコアに任せ、正規化はカナを作らずに表記だけを書き分ける:
    - 住所の4要素目と「号」の3桁の番号は、数字のままだとコアが「ヨンヒャクキュー」と位取りで読むので、「四〇九」のように桁読みの漢数字にする。
    - 4桁の番号は位取りで読む方が短く自然なので（「1205」は「センニヒャクゴ」）、数字のまま渡す。
    - 「号室」の番号は、数字のままでもコアが3桁を「ヨンマルキュー」と桁読みするので、桁数にかかわらず数字のまま渡す。

    0 を含む位置や末尾の数字によらず、同じ規則で書き分けられることを確認する。
    """

    # ========== 3桁の部屋番号: 桁読みの漢数字 ==========
    # 中間に0を含む番号
    assert normalize_text("赤坂1-2-3-101") == "赤坂1の2の3の一〇一"
    assert normalize_text("赤坂1-2-3-301") == "赤坂1の2の3の三〇一"
    assert normalize_text("赤坂1-2-3-409") == "赤坂1の2の3の四〇九"
    assert normalize_text("赤坂1-2-3-507") == "赤坂1の2の3の五〇七"
    assert normalize_text("赤坂1-2-3-608") == "赤坂1の2の3の六〇八"
    assert normalize_text("赤坂1-2-3-903") == "赤坂1の2の3の九〇三"
    # 末尾が2の番号
    assert normalize_text("赤坂1-2-3-102") == "赤坂1の2の3の一〇二"
    assert normalize_text("赤坂1-2-3-302") == "赤坂1の2の3の三〇二"
    assert normalize_text("赤坂1-2-3-902") == "赤坂1の2の3の九〇二"
    # 末尾が5の番号
    assert normalize_text("赤坂1-2-3-105") == "赤坂1の2の3の一〇五"
    assert normalize_text("赤坂1-2-3-205") == "赤坂1の2の3の二〇五"
    assert normalize_text("赤坂1-2-3-805") == "赤坂1の2の3の八〇五"
    # 0を含まない番号
    assert normalize_text("赤坂1-2-3-123") == "赤坂1の2の3の一二三"
    assert normalize_text("赤坂1-2-3-456") == "赤坂1の2の3の四五六"
    assert normalize_text("赤坂1-2-3-789") == "赤坂1の2の3の七八九"
    # 先頭が0の番号
    assert normalize_text("赤坂1-2-3-041") == "赤坂1の2の3の〇四一"
    # 末尾が0の番号
    assert normalize_text("赤坂1-2-3-410") == "赤坂1の2の3の四一〇"
    assert normalize_text("赤坂1-2-3-520") == "赤坂1の2の3の五二〇"
    # 0が続く番号
    assert normalize_text("赤坂1-2-3-100") == "赤坂1の2の3の一〇〇"
    assert normalize_text("赤坂1-2-3-200") == "赤坂1の2の3の二〇〇"
    assert normalize_text("赤坂1-2-3-500") == "赤坂1の2の3の五〇〇"

    # ========== 4桁の部屋番号: 数字のまま ==========
    # 中間に0を含む番号
    assert normalize_text("赤坂1-2-3-1409") == "赤坂1の2の3の1409"
    assert normalize_text("赤坂1-2-3-1205") == "赤坂1の2の3の1205"
    assert normalize_text("赤坂1-2-3-1302") == "赤坂1の2の3の1302"
    assert normalize_text("赤坂1-2-3-1507") == "赤坂1の2の3の1507"
    assert normalize_text("赤坂1-2-3-1608") == "赤坂1の2の3の1608"
    assert normalize_text("赤坂1-2-3-1901") == "赤坂1の2の3の1901"
    # 0を含まない番号
    assert normalize_text("赤坂1-2-3-1234") == "赤坂1の2の3の1234"
    assert normalize_text("赤坂1-2-3-5678") == "赤坂1の2の3の5678"
    # 0が続く番号
    assert normalize_text("赤坂1-2-3-1001") == "赤坂1の2の3の1001"
    assert normalize_text("赤坂1-2-3-2001") == "赤坂1の2の3の2001"
    # 末尾が2の番号
    assert normalize_text("赤坂1-2-3-1102") == "赤坂1の2の3の1102"
    assert normalize_text("赤坂1-2-3-1502") == "赤坂1の2の3の1502"
    # 末尾が5の番号
    assert normalize_text("赤坂1-2-3-1205") == "赤坂1の2の3の1205"
    assert normalize_text("赤坂1-2-3-1305") == "赤坂1の2の3の1305"

    # ========== 号室表記付きのハードコードテスト ==========
    # マンション名 + 3桁の号室・号（号室は数字のまま、号は桁読みの漢数字）
    assert normalize_text("サクラハイツ101号室") == "サクラハイツ'101号室"
    assert normalize_text("グリーンコート205号") == "グリーンコート'二〇五号"
    assert normalize_text("パークハイム502号室") == "パークハイム'502号室"
    assert normalize_text("フォレストタワー809号室") == "フォレストタワー'809号室"
    # マンション名 + 4桁号室
    assert normalize_text("プラウド武蔵小杉1205号室") == "プラウド武蔵小杉'1205号室"
    assert normalize_text("ネクサス鹿島田1607号室") == "ネクサス鹿島田'1607号室"
    assert normalize_text("サクラコート1302号") == "サクラコート'1302号"
    assert normalize_text("フォレストタワー2001号室") == "フォレストタワー'2001号室"

    # ========== 住所 + 建物名 + 号室 の複合テスト（ハードコード期待値） ==========
    assert (
        normalize_text("東京都港区六本木1-2-3 サクラハイツ604号室")
        == "東京都港区六本木1の2の3,サクラハイツ'604号室"
    )
    assert (
        normalize_text("大阪府大阪市北区梅田3-1-3 グリーンコート1205号室")
        == "大阪府大阪市北区梅田3の1の3,グリーンコート'1205号室"
    )
    assert (
        normalize_text("赤坂9-7-1 パークタワー1108号")
        == "赤坂9の7の1,パークタワー'1108号"
    )
    assert (
        normalize_text("六本木1-2-3 フォレストタワー502号室")
        == "六本木1の2の3,フォレストタワー'502号室"
    )
    # 住所 + 建物名 + 4桁号室（末尾が2）
    assert (
        normalize_text("宮城県仙台市若林区裏柴田町36-1 柴田ハイツ1302号室")
        == "宮城県仙台市若林区裏柴田町36の1,柴田ハイツ'1302号室"
    )
    # 住所 + 建物名 + 4桁号室（末尾が5）
    assert (
        normalize_text("静岡県焼津市岡当目588-15 プラウド焼津1205号室")
        == "静岡県焼津市岡当目588の15,プラウド焼津'1205号室"
    )


def test_normalize_text_standalone_address_false_positive():
    """
    地名を伴わない一般的な数字の列や計算式などが、誤って独立した住所番地パターンとして判定されないことを確認する。
    """

    # --- X-Y-Z-NNN（4要素）の非住所パターン ---
    # バージョン番号
    assert (
        normalize_text("バージョン1-2-3-456にアップデートした")
        == "バージョン1-2-3-456にアップデートした"
    )
    # 管理番号・シリアル番号
    assert (
        normalize_text("管理番号3-7-12-309で登録した") == "管理番号3-7-12-309で登録した"
    )
    # 各要素が3桁以上で日付パターン（\d{2}-\d{1,2}-\d{1,2}）に引っかからない形式を使用
    assert (
        normalize_text("シリアル番号100-200-300-456で管理されている")
        == "シリアル番号100-200-300-456で管理されている"
    )
    # ロッカーの組み合わせ番号
    assert normalize_text("ロッカーは4-8-2-105です") == "ロッカーは4-8-2-105です"
    # 座席番号
    assert (
        normalize_text("座席番号7-2-14-100になります") == "座席番号7-2-14-100になります"
    )

    # --- X-Y-Z NNN（3要素 + 空白 + 数字）の非住所パターン ---
    # 住所としての変換（XのYのZ...）が行われないことが重要
    # 空白 → ポーズマーカー（'）の変換は正規化パイプラインの標準動作
    # スポーツのスコア + 得点
    assert (
        normalize_text("戦績は5-3-2 200試合のものです")
        == "戦績は5-3-2'200試合のものです"
    )
    # フォーメーション + 選手番号
    assert (
        normalize_text("フォーメーションは4-3-3 309番の選手が担当")
        == "フォーメーションは4-3-3'309番の選手が担当"
    )
    # 接尾辞が「号」であっても、非住所文脈では住所変換しない
    # またこのようにスペースを削除すると数字同士が隣り合うパターンでは ' を入れる
    assert normalize_text("試合結果5-3-2 309号を記録") == "試合結果5の3の2'309号を記録"
    # 3桁超の数字でも同様に数字連結を防ぐ
    assert normalize_text("試合結果5-3-2 2309を記録") == "試合結果5の3の2'2309を記録"
    # 桁区切りカンマを含む数字でも、先頭境界の ' は維持したまま数字のカンマだけ除去する
    assert normalize_text("試合結果5-3-2 1,000を記録") == "試合結果5の3の2'1000を記録"
    # 2要素住所変換（X-Y）でも、後続の数字と連結しない
    assert normalize_text("区画3-2 100を確認") == "区画3の2'100を確認"


def test_normalize_text_gou_false_positive_composite():
    """
    同一文中に住所表記と非住所の号数表記が混在する場合に、前方の住所文脈が後方の非住所の「号」へ波及して誤って部屋番号として桁読みされないことを確認する。
    """

    # 住所 → 列車号数: 住所文脈が列車号数に波及する
    assert (
        normalize_text("東京都港区六本木1-2-3に届いた。こだま309号で帰る")
        == "東京都港区六本木1の2の3に届いた.こだま309号で帰る"
    )
    assert (
        normalize_text("横浜市の店舗で買い物をした後、のぞみ205号に乗った")
        == "横浜市の店舗で買い物をした後,のぞみ205号に乗った"
    )
    assert (
        normalize_text("大阪府のホテルにチェックインして、ひかり305号で東京に帰る")
        == "大阪府のホテルにチェックインして,ひかり305号で東京に帰る"
    )
    # 住所文脈 + 建物名キーワード → 非住所の号
    assert (
        normalize_text("品川区のビルで会議をして、作品309号を納品した")
        == "品川区のビルで会議をして,作品309号を納品した"
    )
    assert (
        normalize_text("バージョン1-2-3 beta 309号を確認した")
        == "バージョン1-2-3ベータ309号を確認した"
    )
    assert (
        normalize_text("千代田区の図書館で借りた本の管理番号は603号です")
        == "千代田区の図書館で借りた本の管理番号は603号です"
    )
    # 住所文脈が遠く離れた非住所「号」に波及する
    assert (
        normalize_text("奈良県の旅館に宿泊して翌日、連載305号の原稿を書いた")
        == "奈良県の旅館に宿泊して翌日,連載305号の原稿を書いた"
    )


def test_normalize_text_gou_false_positive_sentence_boundary():
    """
    句点や感嘆符などの文境界文字を挟んだ場合に、直前の住所文脈が遮断され、後続の文にある非住所の「号」が部屋番号として誤変換されないことを確認する。
    """

    # 「。」で住所文脈が遮断される（最終出力では「.」になる）
    assert (
        normalize_text("東京都港区六本木1-2-3に届いた。こだま309号で帰る")
        == "東京都港区六本木1の2の3に届いた.こだま309号で帰る"
    )
    # 「！」（z2h で「!」に変換済み）で住所文脈が遮断される
    assert (
        normalize_text("赤坂9-7-1で大事件！列車205号が遅延した")
        == "赤坂9の7の1で大事件!列車205号が遅延した"
    )
    # 「？」（z2h で「?」に変換済み）で住所文脈が遮断される
    assert (
        normalize_text("大阪府大阪市北区梅田3-1-3は何だっけ？こだま334号に乗ろう")
        == "大阪府大阪市北区梅田3の1の3は何だっけ?こだま334号に乗ろう"
    )
    # 改行で住所文脈が遮断される（最終出力では「.」になる）
    assert (
        normalize_text("六本木1-2-3が住所です\nチケット305号を受け取りました")
        == "六本木1の2の3が住所です.チケット305号を受け取りました"
    )
    # 全角コロン「：」（z2h で「:」に変換済み）で住所文脈が遮断される
    # 最終出力では replace_punctuation() により「:」が「,」に変換される
    assert (
        normalize_text("赤坂1-2-3の報告書：作品264号の出品が確認された")
        == "赤坂1の2の3の報告書,作品264号の出品が確認された"
    )
    # 半角コロン「:」で住所文脈が遮断される
    # 最終出力では replace_punctuation() により「:」が「,」に変換される
    assert (
        normalize_text("六本木1-2-3に住んでいる:こだま205号に乗った")
        == "六本木1の2の3に住んでいる,こだま205号に乗った"
    )
    # 「、」→「,」 は文境界として扱わないケースの確認（カンマは文境界ではない）
    # 住所文脈が「、」を挟んでも波及するケースがある（これは意図通り）
    # ただしここでは、文境界ではない「、」で住所コンテキストが遮断されないことを確認する
    # 住所パターン直後の建物名 + 号室は変換されるべき
    assert (
        normalize_text("六本木1-2-3、サクラハイツ309号室に届けてください")
        == "六本木1の2の3,サクラハイツ'309号室に届けてください"
    )


def test_normalize_text_floor_notation():
    """
    「3F」や「B1F」などのフロア階数表記が、「3階」や「地下1階」といった自然な日本語の階数表記へ正規化されることを確認する。
    """

    # --- 基本的なフロア表記 ---
    assert normalize_text("1F") == "1階"
    assert normalize_text("2F") == "2階"
    assert normalize_text("3F") == "3階"
    assert normalize_text("13F") == "13階"
    assert normalize_text("52F") == "52階"

    # --- 地下階 ---
    assert normalize_text("B1F") == "地下1階"
    assert normalize_text("B2F") == "地下2階"
    assert normalize_text("B3F") == "地下3階"

    # --- 文中のフロア表記 ---
    assert normalize_text("3Fのカフェ") == "3階のカフェ"
    assert normalize_text("B1Fにあります") == "地下1階にあります"
    assert normalize_text("エレベーターで13Fへ") == "エレベーターで13階へ"

    # --- ビル名 + フロア ---
    assert normalize_text("マルマルビル 13F") == "マルマルビル13階"
    assert normalize_text("六本木ヒルズ 52F") == "六本木ヒルズ52階"
    assert normalize_text("新宿パークタワー B2F") == "新宿パークタワー地下2階"

    # --- 住所の中のフロア表記 ---
    assert (
        normalize_text("東京都港区六本木1-2-3 六本木グランドタワー 13F")
        == "東京都港区六本木1の2の3,六本木グランドタワー13階"
    )
    assert (
        normalize_text("赤坂9-7-1 ミッドタウン B1F")
        == "赤坂9の7の1,ミッドタウン地下1階"
    )

    # --- フロア表記として判定されないケース ---
    # 後に英字が続く場合は変換しない
    assert normalize_text("5GHz") == "5ギガヘルツ"
    assert normalize_text("UTF-8") == "ユーティーエフエイト"
    assert normalize_text("PDF") == "ピーディーエフ"


def test_normalize_text_phone_postal_address_combined():
    """
    電話番号・郵便番号・住所番地が同一テキスト内に連続または混在して出現する場合に、各パターンが互いに干渉せず正しく正規化されることを確認する。
    """

    # 顧客の実用的な利用シーン: クリニック情報の読み上げ
    assert (
        normalize_text("〒304-0002 茨城県下妻市今泉613-6 TEL:0296-44-5678")
        == "郵便番号304-0002,茨城県下妻市今泉613の6,テル,0296-44-5678"
    )
    assert (
        normalize_text("〒100-0001 東京都千代田区千代田1-1 TEL:03-1234-5678")
        == "郵便番号100-0001,東京都千代田区千代田1の1,テル,03-1234-5678"
    )

    # マンション + 号室 + 電話番号
    # 郵便番号 802 の真ん中の0は、コアが「ハチマルニー」と「マル」で読む
    assert (
        normalize_text(
            "〒802-0838 福岡県北九州市小倉南区石田町399-18"
            " 石田ハイツ101 TEL:093-456-7890"
        )
        == "郵便番号802-0838,福岡県北九州市小倉南区石田町399の18,石田ハイツ'一〇一テル,093-456-7890"
    )

    # フリーダイヤルを含む
    assert (
        normalize_text("ご予約はフリーダイヤル0120-456-789まで。")
        == "ご予約はフリーダイヤル0120-456-789まで."
    )

    # ナビダイヤルを含む
    assert (
        normalize_text("お問い合わせは0570-012-345（ナビダイヤル）まで。")
        == "お問い合わせは0570-012-345'ナビダイヤル'まで."
    )

    # 複数の電話番号
    assert (
        normalize_text("固定電話03-1234-5678、携帯080-4205-7491")
        == "固定電話03-1234-5678,"
        "携帯080-4205-7491"
    )

    # ビル名 + フロア + 電話番号
    assert (
        normalize_text("宮城県仙台市若林区裏柴田町36-1 柴田ビル 3F TEL:022-222-3333")
        == "宮城県仙台市若林区裏柴田町36の1,"
        "柴田ビル3階テル,"
        "022-222-3333"
    )


def test_normalize_text_phone_postal_address_edge_cases():
    """
    電話番号・郵便番号・住所表記における境界値や特殊な区切り記号などのエッジケースが、意図通りに正規化または非変換として処理されることを確認する。
    """

    # --- 数式との区別 ---
    # 既存の数式処理: イコールが数字の間にあるので数式コンテキスト
    assert normalize_text("5-3=2") == "5マイナス3イコール2"
    # スペース付き数式（こちらは明確に数式）
    assert normalize_text("5 - 3 = 2") == "5マイナス3イコール2"
    # `=` がない加算・乗算・除算は数式として読ませる
    assert normalize_text("5 + 8 は") == "5プラス8は"
    assert normalize_text("5*8は") == "5かける8は"
    assert normalize_text("2*3*4は") == "2かける3かける4は"
    assert normalize_text("2 * 3 * 4 は") == "2かける3かける4は"
    assert normalize_text("2×3×4は") == "2かける3かける4は"
    assert normalize_text("2✖3✖4は") == "2かける3かける4は"
    assert normalize_text("2⨯3⨯4は") == "2かける3かける4は"
    assert normalize_text("2＊3＊4は") == "2かける3かける4は"
    assert normalize_text("1.5*2.5は") == "1.5かける2.5は"
    assert normalize_text("1920×1080×2") == "1920かける1080かける2"
    assert normalize_text("6 ÷ 2 は") == "6わる2は"
    assert normalize_text("2*3=6") == "2かける3イコール6"
    # `*` は数字同士のときだけ乗算として扱い、それ以外では従来通り記号として除去される
    assert normalize_text("A*B") == "AB"
    assert normalize_text("注釈*を確認") == "注釈を確認"
    assert normalize_text("2*abc") == "2エービーシー"
    assert normalize_text("abc*2") == "エービーシー2"
    assert normalize_text("2*3*abc") == "2かける3エービーシー"
    # 読めない記号を削除する場合でも、数字同士は別の数として分離する
    assert normalize_text("1_2") == "1'2"
    assert normalize_text("1_2", for_irodori=True) == "1'2"
    # OpenJTalk が読める記号は、数字に挟まれていても表層を残して分かち書きへ渡す
    assert normalize_text("1@2") == "1@2"
    assert normalize_text("1@2", for_irodori=True) == "1@2"
    # 数字に挟まれていない読めない記号は、従来どおり読みを挿入せずに削除する
    assert normalize_text("A_B") == "AB"
    assert normalize_text("A_2") == "A2"
    assert normalize_text("1_B") == "1B"
    # 価格やポイント表記の `＋` は数式として一体化せず、従来通り記号読みだけに留める
    assert normalize_text("53693円＋537ポイント") == "53693円プラス537ポイント"
    # `=` がない減算は住所・スコア・型番と衝突するため従来通り保持する
    assert normalize_text("5 - 3") == "5-3"
    # 先頭0のハイフン区切りは電話番号として処理される（数式より優先）
    assert normalize_text("03-1234-5678") == "03-1234-5678"

    # --- スコア的なパターン（数字-数字でバーサスや比率） ---
    # 先頭が0でない2要素ハイフン区切りは変換しない
    # （33-4 はスコアのように読まれうる）
    assert normalize_text("33-4") == "33-4"
    assert normalize_text("21-0") == "21-0"
    # ただし3桁-4桁は郵便番号として処理される
    assert normalize_text("100-0001") == "100-0001"

    # --- 日付との区別 ---
    # 日付パターンは既存処理が優先
    assert normalize_text("2024-01-01") == "2024年1月1日"
    # 日付に近いが電話番号のパターン（先頭0）
    assert normalize_text("03-12-3456") == "03-12-3456"

    # --- 単位付きのハイフンとの区別 ---
    assert normalize_text("100m〜200m") == "100メートルから200メートル"

    # --- 電話番号として判定されないパターン ---
    # 先頭が0でない
    assert normalize_text("12-3456-7890") == "12-3456-7890"
    # ハイフンなし固定電話は市外局番の桁数が不明なので変換しない
    assert normalize_text("0312345678") == "0312345678"
    # 桁数がどのパターンにも合わない
    assert normalize_text("03-123-45678") == "03-123-45678"
    assert normalize_text("080-123-4567") == "080-123-4567"
    # 12桁以上の数字列は電話番号ではない
    assert normalize_text("080-12345-67890") == "080-12345-67890"

    # --- 郵便番号として判定されないパターン ---
    # 3桁-3桁は郵便番号ではない（ハイフン区切りなので分数にもならない）
    assert normalize_text("123-456") == "123-456"
    # 2桁-4桁は郵便番号ではない
    assert normalize_text("12-3456") == "12-3456"

    # --- 住所のエッジケース ---
    # 部屋番号なしの住所
    assert normalize_text("赤坂9-7-1") == "赤坂9の7の1"
    # 地名の後に5要素以上のハイフン区切り（4要素まで住所として処理）
    assert normalize_text("赤坂1-2-3-4-5") == "赤坂1の2の3の4-5"

    # --- 全角数字・全角ハイフン ---
    assert normalize_text("０３−１２３４−５６７８") == "03-1234-5678"
    assert normalize_text("〒３０４−０００２") == "郵便番号304-0002"

    # --- 連続するパターンの処理順序 ---
    # 郵便番号の直後に住所、その後に電話番号
    assert (
        normalize_text("〒304-0002 下妻市今泉613-6 0296-44-5678") == "郵便番号304-0002,"
        "下妻市今泉613の6,"
        "0296-44-5678"
    )

    # --- 住所 + フロア表記の組み合わせ ---
    assert (
        normalize_text("六本木1-2-3 ABCビル B1F")
        == "六本木1の2の3,エービーシービル地下1階"
    )


def test_normalize_text_address_marker_propagation_prevention():
    """
    住所変換後のマーカー以降に助詞を含む文構造テキストが続く場合に、住所文脈が後続へ不適切に伝播しないことを確認する。
    """

    # マーカー後に助詞「で」を含む文構造テキスト → 住所文脈ではない
    assert (
        normalize_text("赤坂1-2-3 パークハイムの受付で309号の書類")
        == "赤坂1の2の3,パークハイムの受付で309号の書類"
    )
    # マーカー後に助詞「で」→ 住所文脈ではない
    assert (
        normalize_text("横浜市鶴見区5-3-2で買い物をした後、のぞみ205号に乗った")
        == "横浜市鶴見区5の3の2で買い物をした後,のぞみ205号に乗った"
    )
    # マーカー後に助詞「に」→ マーカー判定では住所文脈としないが、
    # check 5（建物名キーワード近接）で「ビル」が距離2以内にあるため変換される
    # これは check 5 の仕様通りの挙動（「ビルにて」の「にて」は2文字）
    assert (
        normalize_text("赤坂1-2-3 ABCビルにて309号の議案を審議")
        == "赤坂1の2の3,エービーシービルにて'三〇九号の議案を審議"
    )
    # 「で」は1文字なので check 5 も発動する（建物名キーワードから距離1）
    # 一方マーカー判定では「受付で」にひらがな「で」があるため住所文脈としない
    # → check 5 のみで変換される
    assert (
        normalize_text("赤坂1-2-3 ABCビルで309号の議案を審議")
        == "赤坂1の2の3,エービーシービルで'三〇九号の議案を審議"
    )
    # check 5 が発動しない距離（「ビルの受付にて」→ 距離5）ではマーカー判定の
    # 厳格パターンにより住所文脈としない
    assert (
        normalize_text("赤坂1-2-3 ABCビルの受付にて309号の議案を審議")
        == "赤坂1の2の3,エービーシービルの受付にて309号の議案を審議"
    )

    # 対照: マーカー後が建物名のみ → 住所文脈として変換される
    assert (
        normalize_text("赤坂1-2-3 パークハイム 309号")
        == "赤坂1の2の3,パークハイム'三〇九号"
    )
    assert (
        normalize_text("赤坂1-2-3 パークハイム鹿島田スカイタワー 309号")
        == "赤坂1の2の3,パークハイム鹿島田スカイタワー'三〇九号"
    )
    # マーカー後に「の」のみ含む建物名 → 許容される
    assert (
        normalize_text("赤坂1-2-3 パークタワーの丘 309号")
        == "赤坂1の2の3,パークタワーの丘'三〇九号"
    )


def test_normalize_text_building_brand_names():
    """
    主要なマンション・ビルブランド名が建物名として認識され、後続する部屋番号の「号」が適切に桁読みへ正規化されることを確認する。
    """

    # 野村不動産: プラウド
    assert normalize_text("プラウド武蔵小杉1205号室") == "プラウド武蔵小杉'1205号室"
    # 東急不動産: ブランズ
    assert normalize_text("ブランズ横浜1302号室") == "ブランズ横浜'1302号室"
    # 東京建物: ブリリア
    assert normalize_text("ブリリア目黒809号室") == "ブリリア目黒'809号室"
    # 大和ハウス: プレミスト
    assert normalize_text("プレミスト新宿1002号室") == "プレミスト新宿'1002号室"
    # 三井不動産: パークホームズ
    assert (
        normalize_text("パークホームズ西立川502号室") == "パークホームズ西立川'502号室"
    )
    # 三井不動産: パークコート
    assert normalize_text("パークコート青山2001号室") == "パークコート青山'2001号室"
    # 三菱地所: パークハウス
    assert normalize_text("パークハウス渋谷502号室") == "パークハウス渋谷'502号室"
    # 住友不動産: シティタワー
    assert normalize_text("シティタワー品川1705号室") == "シティタワー品川'1705号室"
    # 住友不動産: グランドヒルズ（ブランド名直後に部屋番号）
    assert normalize_text("グランドヒルズ205号") == "グランドヒルズ'二〇五号"
    # グランドヒルズ + 地名: 「グランドヒルズ」がまとめてマッチするため、
    # 後続の地名「白金台」(3文字) で距離が3となり check 5 は発動しない
    # ただし住所文脈があれば変換される
    assert (
        normalize_text("港区白金台3-1-2 グランドヒルズ白金台 205号")
        == "港区白金台3の1の2,グランドヒルズ白金台'二〇五号"
    )
    # 大京: ライオンズ
    assert normalize_text("ライオンズ武蔵小杉1302号室") == "ライオンズ武蔵小杉'1302号室"
    # タカラレーベン: レーベン
    assert normalize_text("レーベン川崎809号室") == "レーベン川崎'809号室"
    # 一般建物種別語の追加分: ガーデン
    assert normalize_text("サクラガーデン101号室") == "サクラガーデン'101号室"
    # 一般建物種別語の追加分: メゾン（号室 → 5b 無条件変換パス）
    assert normalize_text("メゾン青葉台205号室") == "メゾン青葉台'205号室"
    # メゾン + 号（号室なし → 5c パス、has_address_context 必要）
    # 地名が2文字以内なら check 5 が発動する
    assert normalize_text("メゾン目黒205号") == "メゾン目黒'二〇五号"
    # 一般建物種別語の追加分: ヴィラ
    assert normalize_text("ヴィラ世田谷309号室") == "ヴィラ世田谷'309号室"
    # ブランド名 + 住所との複合テスト
    assert (
        normalize_text("東京都港区六本木1-2-3 ブリリア六本木1205号室")
        == "東京都港区六本木1の2の3,ブリリア六本木'1205号室"
    )
    assert (
        normalize_text("神奈川県川崎市宮前区梶が谷5-30-2 パークホームズ新宮前平 809号")
        == "神奈川県川崎市宮前区梶が谷5の30の2,パークホームズ新宮前平'八〇九号"
    )


def _convert_room_digits_for_expected(digits: str, suffix: str) -> str:
    """
    号室・部屋番号の期待値文字列を生成する。
    """

    # 号室の番号は数字のまま渡し、コアに3桁は桁読み・4桁は位取りで読ませる
    if suffix == "号室":
        return digits
    # 号の番号は、数字のままだと3桁が位取りで読まれるので、3桁だけ桁読みの漢数字にする
    ## 例: 3桁 "409" → 四〇九, 4桁 "1203" → 1203
    if len(digits) == 3:
        return digits.translate(str.maketrans("0123456789", "〇一二三四五六七八九"))
    return digits


def _build_room_context_positive_cases() -> list[tuple[str, str]]:
    """
    住所、建物名、号室が組み合わさったテストケースの一覧を生成する。
    """

    address_cases = [
        ("神奈川県川崎市幸区鹿島田1-34-5", "神奈川県川崎市幸区鹿島田1の34の5"),
        ("東京都港区赤坂1-2-3", "東京都港区赤坂1の2の3"),
        ("大阪府大阪市北区梅田3-1-3", "大阪府大阪市北区梅田3の1の3"),
        ("福岡県北九州市小倉南区石田町399-18", "福岡県北九州市小倉南区石田町399の18"),
        ("宮城県仙台市若林区裏柴田町36-1", "宮城県仙台市若林区裏柴田町36の1"),
        ("静岡県焼津市岡当目588-15", "静岡県焼津市岡当目588の15"),
        ("奈良県桜井市鹿路341-18", "奈良県桜井市鹿路341の18"),
        ("六本木1-2-3", "六本木1の2の3"),
        ("赤坂9-7-1", "赤坂9の7の1"),
        ("神奈川県川崎市幸区鹿島田１−３４−５", "神奈川県川崎市幸区鹿島田1の34の5"),
    ]
    building_names = [
        "パークハイム鹿島田第一",
        "プラウド武蔵小杉",
        "鹿島田第一",
        "ネクサス鹿島田",
        "サクラコート",
        "フォレストタワー",
    ]
    room_numbers = [
        # 3桁: 中間に0を含む番号
        "101",
        "204",
        "309",
        "405",
        "502",
        "802",
        # 3桁: 0を含まない番号
        "123",
        "789",
        # 3桁: 先頭以外に0が続く番号と、末尾が0の番号
        "100",
        "410",
        # 4桁: 号室・号とも数字のまま渡す番号
        "1205",
        "1409",
        "1302",
        "1001",
        "2505",
    ]
    suffixes = [
        "号",
        "号室",
    ]
    spaces = [
        " ",
    ]

    cases: list[tuple[str, str]] = []
    for address_input, normalized_address in address_cases:
        for space_between_address_and_building in spaces:
            for building_name in building_names:
                for space_between_building_and_room in spaces:
                    for room_number in room_numbers:
                        for suffix in suffixes:
                            room_text = _convert_room_digits_for_expected(
                                room_number, suffix
                            )
                            text = (
                                f"{address_input}"
                                f"{space_between_address_and_building}"
                                f"{building_name}"
                                f"{space_between_building_and_room}"
                                f"{room_number}{suffix}"
                            )
                            expected = (
                                f"{normalized_address},{building_name}"
                                f"'{room_text}{suffix}"
                            )
                            cases.append((text, expected))
    return cases


def _build_room_context_negative_cases() -> list[tuple[str, str]]:
    """
    住所文脈を持たない「号」の非変換テストケースの一覧を生成する。
    """

    prefixes = [
        "こだま",
        "のぞみ",
        "ひかり",
        "特急",
        "急行",
        "第",
        "作品",
        "問題",
        "章",
        "話",
        "案件",
        "型番",
        "規格",
        "便",
        "列車",
        "ルール",
        "プロトコル",
        "プレイリスト",
        "任務",
        "講義",
        "イベント",
        "チャンネル",
        "プラン",
        "フォーマット",
        "テンプレート",
        "シリーズ",
        "ファイル",
        "プロジェクト",
        "テスト",
        "モデル",
    ]
    room_numbers = [
        "101",
        "204",
        "309",
        "405",
        "502",
        "1205",
        "1409",
        "1302",
    ]
    suffixes = [
        "号",
    ]

    cases: list[tuple[str, str]] = []
    for prefix in prefixes:
        for room_number in room_numbers:
            for suffix in suffixes:
                text = f"{prefix}{room_number}{suffix}を参照してください"
                expected = text
                cases.append((text, expected))
    return cases


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_room_context_positive_cases(),
)
def test_normalize_text_room_context_positive_massive(text: str, expected: str):
    """
    住所文脈を伴う「号」「号室」の表記が、多様な組み合わせにおいて部屋番号の桁読みへ正しく正規化されることを確認する。
    """

    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_room_context_negative_cases(),
)
def test_normalize_text_room_context_negative_massive(text: str, expected: str):
    """
    住所文脈を持たない文章中の「号」表記が、誤って部屋番号の桁読みへ変換されないことを確認する。
    """

    assert normalize_text(text) == expected


def test_normalize_text_address_without_admin_name_wontfix():
    """
    市区町村名などの行政区画名を伴わない住所表記において、一般語との誤判定を防ぐために「号」が桁読みされない仕様通りの挙動を確認する。

    「赤坂1-2-3 309号」のように市区町村を省略した住所表記では、
    住所変換マーカーは付与されるが、マーカー直後のテキストが空（スペースのみ）のため、
    has_address_context() のフォールバックが明示的な住所語彙または行政区画名を要求し、
    号の桁読み変換が行われない。

    これは意図的な設計判断であり、以下の理由により Wont fix としている。

    1. 偽陽性の防止:
       「赤坂」と「結果」のように、住所に使われる漢字と一般語の漢字を区別する手段がない。
       マーカーを無条件に信頼すると「試合結果5-3-2 309号を記録」の 309号 が
       誤って桁読みされてしまう（__ADDRESS_PATTERN が任意の漢字 + 数字パターンにマッチするため）。

    2. 実用上の影響が極めて小さい:
       日本語の住所表記では市区町村（最低でも「市」や「区」）を省略することはほぼない。
       TTS の入力として「赤坂1-2-3 309号」のように区画名なしで入力されることは稀であり、
       通常は「港区赤坂1-2-3 309号」のように記載される。
       行政区画名があれば has_address_context() の
       __ADDRESS_ADMINISTRATIVE_NAME_PATTERN で正しく住所文脈と判定される。

    3. 建物名がある場合は正しく変換される:
       「赤坂1-2-3 パークハイム 309号」のように建物名を伴う場合は、
       マーカー後テキストの厳格パターン検証を通過し、正しく変換される。

    これらの理由から、行政区画名のない住所表記において「号」が桁読みされない仕様通りの挙動を確認する。
    """

    # 行政区画名なし + 号（5c パス: has_address_context 必要）→ 変換されない
    # 住所変換マーカーは付与されるが、empty-tail フォールバックで
    # 「赤坂」は行政区画接尾辞（都/道/府/県/市/区/町/村）を持たないため不一致
    assert normalize_text("赤坂1-2-3 309号") == "赤坂1の2の3'309号"
    assert normalize_text("鹿島田588-15 309号") == "鹿島田588の15'309号"
    assert normalize_text("六本木1-2-3 409号") == "六本木1の2の3'409号"

    # 対照: 行政区画名ありなら正しく変換される
    assert normalize_text("港区赤坂1-2-3 309号") == "港区赤坂1の2の3,'三〇九号"
    assert normalize_text("幸区鹿島田588-15 309号") == "幸区鹿島田588の15,'三〇九号"
    assert normalize_text("港区六本木1-2-3 409号") == "港区六本木1の2の3,'四〇九号"

    # 対照: 建物名ありなら行政区画名なしでも変換される
    assert (
        normalize_text("赤坂1-2-3 パークハイム 309号")
        == "赤坂1の2の3,パークハイム'三〇九号"
    )

    # 対照: 号室（5b パス: has_address_context 不要, 無条件で変換される）
    assert normalize_text("赤坂1-2-3 309号室") == "赤坂1の2の3,'309号室"

    # この設計判断により防がれている偽陽性
    # __ADDRESS_PATTERN は「結果5-3-2」も住所として変換してしまう（漢字マッチの広さ）ため、
    # マーカーを無条件に信頼すると以下が誤変換される
    assert normalize_text("試合結果5-3-2 309号を記録") == "試合結果5の3の2'309号を記録"


def _build_gou_false_positive_admin_kanji_cases() -> list[tuple[str, str]]:
    """
    行政区画に用いられる漢字を含みつつ住所文脈ではない文章において、「号」が誤変換されないことを検証するためのテストケース一覧を生成する。
    """

    # {num} に数字を埋め込み、{num}号 が変換されないことを確認する
    # 各テンプレートには行政区画漢字を含む非住所語が含まれている
    templates = [
        # 「市」: 市場, 市民, 都市, 市販, 闇市, 朝市, 市街地, 市長
        "市場で{num}号の整理券を受け取った",
        "市民ホールで公演{num}号が開催された",
        "都市計画のプロジェクト{num}号を推進する",
        "市販の製品カタログ{num}号を確認した",
        "闇市で仕入れた品物の管理タグ{num}号がある",
        "朝市で整理券{num}号を配布している",
        "市街地で特別列車{num}号を見かけた",
        "市長が法案{num}号に署名した",
        # 「区」: 区別, 区間, 地区, 学区, 区画, 特区
        "区別がつかないので識別番号{num}号を振った",
        "区間快速の列車{num}号に乗った",
        "地区大会で選手番号{num}号が優勝した",
        "学区の会報誌{num}号が届いた",
        "区画整理の事業認可{num}号が下りた",
        "特区に指定された施設の登録番号{num}号です",
        # 「町」: 下町, 城下町, 門前町, 町内, 町工場
        "下町の店で整理券{num}号を配布した",
        "城下町を散策して記念コイン{num}号を購入した",
        "門前町の伝統祭りでくじ{num}号を引いた",
        "町内会の回覧板{num}号を回してください",
        "町工場で部品{num}号を製造している",
        # 「村」: 村上, 村田, 農村, 漁村, 山村
        "村上春樹の短編集で作品{num}号が好きだ",
        "村田製作所の製品カタログ{num}号を確認した",
        "農村の暮らしを描いた絵画{num}号が入賞した",
        "漁村で水揚げされた漁獲管理番号{num}号を確認した",
        "山村留学のパンフレット{num}号を取り寄せた",
        # 「道」: 北海道, 柔道, 書道, 鉄道, 水道, 歩道, 茶道
        "北海道の名産品カタログ{num}号を取り寄せた",
        "柔道の段位証書{num}号を授与された",
        "書道展に出品された作品{num}号が入賞した",
        "鉄道ファンの雑誌{num}号を購読している",
        "水道の検査レポート{num}号を提出した",
        "歩道の改善要望書{num}号が受理された",
        "茶道の免状で認定番号{num}号を受けた",
        # 「府」: 政府, 幕府, 府中, 内閣府
        "政府が発表した政令{num}号を確認した",
        "幕府が発布した法令{num}号を調査した",
        "府中の競馬場でレース{num}号が開催された",
        "内閣府の告示{num}号を参照してください",
        # 「県」: 県庁, 県民, 県警, 県道
        "県庁で申請書{num}号を提出した",
        "県民アンケート{num}号を集計した",
        "県警が捜査資料{num}号を公開した",
        "県道の標識を管理番号{num}号で登録した",
        # 「都」: 都合, 首都, 都営, 都心, 都度
        "都合が悪いので予約{num}号をキャンセルした",
        "首都高速の路線{num}号が渋滞している",
        "都営バスの系統{num}号に乗車した",
        "都心で開催されたイベントのブース{num}号に出展した",
        "都度払いで請求書{num}号を発行した",
    ]

    numbers = ["309", "205", "1205"]

    cases: list[tuple[str, str]] = []
    for template in templates:
        for num in numbers:
            text = template.format(num=num)
            # 期待値: NNN号 が桁読みに変換されない（入力と同一）
            cases.append((text, text))
    return cases


def _build_gou_false_positive_building_keyword_cases() -> list[tuple[str, str]]:
    """
    建物名キーワードを含みつつ住所文脈ではない文章において、「号」が誤変換されないことを検証するためのテストケース一覧を生成する。
    """

    templates = [
        # 「タワー」: 東京タワー, タワーレコード, タワーディフェンス
        "東京タワーの入場券{num}号を持っている",
        "タワーレコードの注文番号{num}号が発送された",
        "タワーディフェンスゲームのステージ{num}号をクリアした",
        # 「ビル」: ビルド, ビルダー, ビル（人名）
        "ビルドエラーのチケット{num}号を修正した",
        "ビルダーパターンのプルリクエスト{num}号をマージした",
        # 「館」: 体育館, 図書館, 映画館, 美術館, 水族館, 博物館
        "体育館でロッカー番号{num}号を使った",
        "図書館の蔵書番号{num}号を借りた",
        "映画館でシアター{num}号に入場した",
        "美術館の展示品番号{num}号が修復中だ",
        "水族館のチケット{num}号で入場した",
        "博物館の収蔵品{num}号を展示する予定だ",
        # 「荘」: 荘厳, 荘園
        "荘厳な雰囲気の演奏会プログラム{num}号が始まった",
        "荘園の歴史を記した文献{num}号を参照した",
        # 「コート」: テニスコート, バスケットボールコート, コート（衣類）
        "テニスコートの利用予約番号{num}号を取った",
        "コートを着て外出し整理券{num}号を受け取った",
        # 「棟」: 棟梁
        "棟梁が手がけた建築の文化財指定{num}号を受けた",
    ]

    numbers = ["309", "205", "1205"]

    cases: list[tuple[str, str]] = []
    for template in templates:
        for num in numbers:
            text = template.format(num=num)
            cases.append((text, text))
    return cases


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_gou_false_positive_admin_kanji_cases(),
)
def test_normalize_text_gou_false_positive_admin_kanji(
    text: str,
    expected: str,
):
    """
    行政区画に用いられる漢字を含む非住所文脈において、「号」表記が誤って部屋番号の桁読みへ変換されないことを確認する。
    """

    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    _build_gou_false_positive_building_keyword_cases(),
)
def test_normalize_text_gou_false_positive_building_keywords(
    text: str,
    expected: str,
):
    """
    建物名キーワードを含む非住所文脈において、「号」表記が誤って部屋番号の桁読みへ変換されないことを確認する。
    """

    assert normalize_text(text) == expected


def test_normalize_text_cross_mark_context_dependent() -> None:
    """
    「×」などの各種バツ記号について、前後の文字種（漢字・カタカナ・数字・英字）に応じて「かける」と「バツ」へ適切に読み分けられることを確認する。

    判定ルール:
      × の両側が漢字・カタカナ・数字・アルファベットの場合 → 「かける」。
      それ以外（ひらがな・空白・句読点・文頭文末等） → 「バツ」。

    これらの条件に従い、文脈に応じた適切な読みへ正規化されることを確認する。
    """

    # --- 「かける」になるケース: 両側が漢字・カタカナ・数字・アルファベット ---
    # 数学（数式パターン: digit × digit = digit）
    assert normalize_text("3×5=15") == "3かける5イコール15"
    assert normalize_text("2×3=6") == "2かける3イコール6"

    # 寸法（数字×数字、= なし）
    assert normalize_text("1920×1080") == "1920かける1080"

    # コラボレーション（漢字×カタカナ）
    assert normalize_text("きのこの山×タケノコの里") == "きのこの山かけるタケノコの里"

    # 漢字×漢字
    assert normalize_text("猫×犬") == "猫かける犬"

    # アルファベット×アルファベット
    assert normalize_text("A×B") == "AかけるB"

    # スピーカー仕様（アルファベット×数字）
    assert "8Wかける2基" in normalize_text("8W×2基のスピーカーを搭載")

    # ✖ (U+2716) バリエーション
    assert normalize_text("3✖5=15") == "3かける5イコール15"

    # ⨯ (U+2A2F) バリエーション
    assert normalize_text("1920⨯1080") == "1920かける1080"

    # --- 「バツ」になるケース: 片側以上がひらがな・空白・句読点等 ---
    # ○×の対比（ひらがな「か」に挟まれる）
    assert normalize_text("○か×か") == "マルかバツか"

    # 単体使用
    assert normalize_text("×") == "バツ"

    # ひらがなに隣接
    assert normalize_text("答えは×です") == "答えはバツです"

    # ❌ (U+274C) も同様のヒューリスティックで処理される
    assert normalize_text("❌") == "バツ"
    assert normalize_text("正解は❌") == "正解はバツ"

    # ❌ のバリエーションセレクタ付き
    assert normalize_text("❌\ufe0f") == "バツ"


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 人名や作品名どうしをつなぐ「×」は、空白や括弧を挟んでも「かける」と読む
        (
            "対談　落合陽一×羽生結弦",
            "対談,落合陽一かける羽生結弦",
            "対談、落合陽一かける羽生結弦",
        ),
        ("落合陽一 × 羽生結弦", "落合陽一かける羽生結弦", "落合陽一かける羽生結弦"),
        ("落合陽一　×　羽生結弦", "落合陽一かける羽生結弦", "落合陽一かける羽生結弦"),
        (
            "嵐山光三郎（作家）×篠原勝之（ゲージツ家）",
            "嵐山光三郎'作家'かける篠原勝之'ゲージツ家'",
            "嵐山光三郎（作家）かける篠原勝之（ゲージツ家）",
        ),
        (
            "「鬼滅の刃」×「呪術廻戦」",
            "'鬼滅の刃'かける'呪術廻戦'",
            "「鬼滅の刃」かける「呪術廻戦」",
        ),
        # 伏せ字の「××」と記号の対比、掛け算は従来どおり読む
        ("××病院", "バツバツ病院", "バツバツ病院"),
        ("○か×か", "マルかバツか", "マルかバツか"),
        ("2×3", "2かける3", "2かける3"),
        ("円周率×母線", "円周率かける母線", "円周率かける母線"),
    ],
)
def test_normalize_text_cross_mark_between_names(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「落合陽一 × 羽生結弦」「「鬼滅の刃」×「呪術廻戦」」のように、空白や括弧を挟んで人名や作品名をつなぐコラボ表記の「×」が、「バツ」ではなく「かける」と読まれることを確認する。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5mL・kg−1・min−1", "5ミリリットルマイキログラムマイフン"),
        ("五ｍｌ・ｋｇ−１・ｍｉｎ−１", "五ミリリットルマイキログラムマイフン"),
        ("8mg・kg^-1", "8ミリグラムマイキログラム"),
        ("2L·s⁻¹", "2リットルマイ秒"),
    ],
)
def test_normalize_reciprocal_compound_units(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    複合単位の負の1乗表記が「マイ」を伴う読みへ変換され、元の数量と単位の関係が保たれて正規化されることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == expected


def test_reciprocal_compound_unit_replacement_details() -> None:
    """
    複合単位の読み替えにおいて、元の単位表記と対応する区間が details に正確に記録されて返されることを確認する。
    """

    result = normalize_text("5mL・kg−1・min−1", return_details=True)
    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == [
        ("unit", "5mL・kg−1・min−1", "5ミリリットルマイキログラムマイフン"),
    ]


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mL・kg−1・min−1", "ミリリットルマイキログラムマイフン"),
        ("L・min−1", "リットルマイフン"),
        ("L・s−1", "リットルマイビョー"),
    ],
)
def test_reciprocal_compound_units_keep_time_unit_readings(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    複合単位に含まれる時間単位が適切な時間の読みで発音され、「分」が不自然に分割されずに正規化されることを確認する。
    """

    normalized = normalize_text(text, for_irodori=for_irodori)
    assert (
        pyopenjtalk.g2p(
            normalized, kana=True, use_sudachi_kanji_yomi=False, predict_nani=False
        )
        == expected
    )


def test_reciprocal_compound_unit_rejects_other_exponents() -> None:
    """
    「マイ」を用いた読み替えの対象が既知の単位の負の1乗に限定され、その他の指数表記には適用されないことを確認する。
    """

    for text in ["mL・kg−10", "mL・kg−1.5", "xml・kg−1"]:
        assert "マイ" not in normalize_text(text)


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        ("（≧∇≦）", "''", "（）"),
        ("(≧∇≦)", "''", "（）"),
        ("（≧▽≦）", "''", "（）"),
        ("（≧ω≦）プッ！！", "''プッ!!", "（）プッ！！"),
        ("（≧△≦）", "''", "（）"),
        ("（≧_≦）", "''", "（）"),
        ("（ ≧ ∇ ≦ ）", "''", "（）"),
        ("(≥▽≤)", "''", "（）"),
        ("（＞▽＜）", "''", "（）"),
        ("嬉しい（≧∇≦）です", "嬉しい''です", "嬉しい（）です"),
        ("でしたぁ♪ヾ（≧▽≦）ノ", "でしたぁ''", "でしたぁ（）"),
        ("ヾ(≧ω≦)ﾉ", "''", "（）"),
        ("1（≧∇≦）2", "1''2", "1（）2"),
    ],
)
def test_normalize_text_kaomoji(
    text: str, expected: str, expected_irodori: str, for_irodori: bool
) -> None:
    """
    括弧に囲まれた顔文字の構成要素（目・口・手など）が除去され、括弧がポーズ表現として維持された上で前後の語句が正規化されることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        ("（x≧0）", "'x大なりイコール0'", "（x大なりイコール0）"),
        ("a≦b", "a小なりイコールb", "a小なりイコールb"),
        (
            "（≧x≦）",
            "'大なりイコールx小なりイコール'",
            "（大なりイコールx小なりイコール）",
        ),
        (
            "（≧３≦）",
            "'大なりイコール3小なりイコール'",
            "（大なりイコール3小なりイコール）",
        ),
        (
            "（≧π≦）",
            "'大なりイコールパイ小なりイコール'",
            "（大なりイコールパイ小なりイコール）",
        ),
        ("（∇）", "'ナブラ'", "（ナブラ）"),
        ("（笑）", "'笑'", "（笑）"),
        (
            "（≧∇≦",
            "'大なりイコールナブラ小なりイコール",
            "（大なりイコールナブラ小なりイコール",
        ),
        ("楽しい（＾▽＾）", "楽しい''", "楽しい（）"),
    ],
)
def test_normalize_text_kaomoji_non_targets(
    text: str, expected: str, expected_irodori: str, for_irodori: bool
) -> None:
    """
    英数字を含む不等式や、顔文字の構成要件を満たさない記号列が誤って顔文字として除去されず、本来の読みが保たれることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        (
            "嬉しい(>_<)ノートを買った",
            "嬉しい''ノートを買った",
            "嬉しい（）ノートを買った",
        ),
        ("(>_<)ノイズ", "''ノイズ", "（）ノイズ"),
        ("(>_<)ノンストップ", "''ノンストップ", "（）ノンストップ"),
        ("(>_<)ﾉｰﾄ", "''ノート", "（）ノート"),
        ("（≧∇≦）ノート", "''ノート", "（）ノート"),
        ("ヾ(≧ω≦)ﾉｲｽﾞ", "''ノイズ", "（）ノイズ"),
        ("(>_<)ノ", "''", "（）"),
        ("(>_<)ﾉ", "''", "（）"),
        ("(>_<)ノ。", "''.", "（）。"),
        ("(>_<)ノ ありがとう", "''ありがとう", "（）ありがとう"),
    ],
)
def test_normalize_text_kaomoji_keeps_following_katakana_words(
    text: str, expected: str, expected_irodori: str, for_irodori: bool
) -> None:
    """
    顔文字の直後に続くカタカナ語の先頭文字が保持され、手として使われている単独の「ノ」や「ﾉ」のみが適切に除去されることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


def test_normalize_text_symbols():
    """
    矢印記号や特殊記号などの各種記号が、文脈に合わせた適切なカタカナ読みへ正規化されることを確認する。
    """

    # 基本的な記号
    assert normalize_text("@") == "@"
    assert normalize_text("＠") == "@"
    assert normalize_text("&") == "&"
    assert normalize_text("＆") == "&"
    assert normalize_text("A@B") == "A@B"
    assert normalize_text("A＠B") == "A@B"
    assert normalize_text("A&B") == "A&B"
    assert normalize_text("A＆B") == "A&B"
    assert normalize_text("ABC+ABC") == "エービーシープラスエービーシー"
    assert normalize_text("ABC&ABC") == "エービーシー&エービーシー"
    assert normalize_text("abc+abc") == "エービーシープラスエービーシー"
    assert normalize_text("abc&abc") == "エービーシー&エービーシー"
    assert normalize_text("OpenAI&ChatGPT") == "オープンエーアイ&チャットジーピーティー"
    assert (
        normalize_text("OpenAI & ChatGPT") == "オープンエーアイ&チャットジーピーティー"
    )
    assert (
        normalize_text("OpenAPI-Specification")
        == "オープンエーピーアイスペシフィケーション"
    )
    # 数式
    assert normalize_text("1+1=2") == "1プラス1イコール2"
    assert normalize_text("5-3=2") == "5マイナス3イコール2"
    assert normalize_text("2×3=6") == "2かける3イコール6"
    assert normalize_text("6÷2=3") == "6わる2イコール3"
    # 比較演算子
    assert normalize_text("5>3") == "5大なり3"
    assert normalize_text("5≥3") == "5大なりイコール3"
    assert normalize_text("2<4") == "2小なり4"
    assert normalize_text("2≤4") == "2小なりイコール4"


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 見出しの前後などで連続する「＋」や、記号に挟まれた「＋」は装飾なので読まない
        ("＋＋お知らせ＋＋", "お知らせ", "お知らせ"),
        ("++重要++", "重要", "重要"),
        ("★＋☆＋★", "", ""),
        ("無理かな・・・・・・・・・＋", "無理かな.........", "無理かな………"),
        ("＋＋＋＋＋＋＋　お知らせ　＋＋＋＋＋＋", ".お知らせ.", "。お知らせ。"),
        # 語の後や語と語の間の「＋」と、英字の後の「++」は、従来どおり「プラス」と読む
        ("C++", "Cプラスプラス", "Cプラスプラス"),
        ("Notepad++", "ノートパッドプラスプラス", "ノートパッドプラスプラス"),
        ("Streaming+", "ストリーミングプラス", "ストリーミングプラス"),
        (
            "これ＋これを買ってきて",
            "これプラスこれを買ってきて",
            "これプラスこれを買ってきて",
        ),
        ("Alt＋F4", "アルトプラスF4", "アルトプラスF4"),
        ("1+1", "1プラス1", "1プラス1"),
        ("+1", "プラス1", "プラス1"),
        # 括弧や鉤括弧で囲んだ語どうしの「＋」と、遺伝子型の「/＋」も「プラス」と読む
        ("「Alt」＋「F4」", "'アルト'プラス'F4'", "「アルト」プラス「F4」"),
        (
            "（税引き営業利益）＋（減価償却費）",
            "'税引き営業利益'プラス'減価償却費'",
            "（税引き営業利益）プラス（減価償却費）",
        ),
        ("モカ/＋", "モカ/プラス", "モカ/プラス"),
        # 鉤括弧や括弧で囲んだ記号の「＋」と、英字や数字の前の「++」も、記号の説明や演算子なので「プラス」と読む
        (
            "「＋」キーを押してください",
            "'プラス'キーを押してください",
            "「プラス」キーを押してください",
        ),
        ("(++i)を計算する", "'プラスプラスi'を計算する", "（プラスプラスi）を計算する"),
        # 顔文字の「（＋＿＋）」の「＋」は、括弧の中でも記号そのものを指していないので装飾として除く
        ("お許しを（（＋＿＋））", "お許しを''''", "お許しを（（））"),
        (
            "［税引き営業利益］＋［減価償却費］",
            "'税引き営業利益'プラス'減価償却費'",
            "「税引き営業利益」プラス「減価償却費」",
        ),
    ],
)
def test_normalize_text_decorative_plus_signs(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「＋＋お知らせ＋＋」「★＋☆＋★」のように装飾として並べた「＋」が「プラスプラス」と読まれず除かれることを確認する。
    「C++」「Streaming+」「これ＋これ」のような語に付く「＋」は、従来どおり「プラス」と読む。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected_details"),
    [
        ("#06", [("number", "#06", "ナンバー06")]),
        ("白＃二十八", [("number", "＃二十八", "ナンバー二十八")]),
        ("＃タグ", [("symbol", "＃タグ", "タグ")]),
        ("「#」キー", [("symbol", "#", "シャープ")]),
        ("#7119に電話", [("number", "#7119", "シャープ、七一一九")]),
    ],
)
def test_normalize_text_return_details_number_sign(
    text: str, expected_details: list[tuple[str, str, str]]
) -> None:
    """
    「＃」を「ナンバー」と読む番号、読まずに除くハッシュタグ、「シャープ」と読む単独の「#」が、置換区間の details に実際の出力のとおり記録されることを確認する。
    「#06」の「#」を削除したと記録すると、details から原文を復元するときに「#ナンバー06」と文字が増える。
    """

    result = normalize_text(text, for_irodori=True, return_details=True)

    assert [
        (detail.category, detail.original_text, detail.normalized_text)
        for detail in result.details
    ] == expected_details
    for detail in result.details:
        assert text[detail.original_start : detail.original_end] == detail.original_text
        assert (
            result.text[detail.normalized_start : detail.normalized_end]
            == detail.normalized_text
        )


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 数字の前の「＃」は番号なので「ナンバー」と読む
        (
            "＃５＜アトラクション＞",
            "ナンバー5アトラクション",
            "ナンバー5アトラクション",
        ),
        ("#06", "ナンバー06", "ナンバー06"),
        ("白＃二十八", "白ナンバー二十八", "白ナンバー二十八"),
        # HTML の文字参照の「&#」は番号ではないので、「ナンバー」と読まない
        ("＆＃一万二千五百七十二；", "&一万二千五百七十二,", "&一万二千五百七十二、"),
        # ハッシュタグや顔文字の「＃」は読まない
        ("＃タグ", "タグ", "タグ"),
        ("#タグ付けして投稿", "タグ付けして投稿", "タグ付けして投稿"),
        ("（＃＾．＾＃）", "'.'", "（。）"),
        ("#おうち時間", "おうち時間", "おうち時間"),
        # 漢数字で始まっても、後ろに漢字が続くハッシュタグは番号ではないので読まない
        ("#千葉", "千葉", "千葉"),
        ("#一人旅", "一人旅", "一人旅"),
        # 鉤括弧で囲んだり助詞を続けたりして「#」そのものを指すときは、「シャープ」と読む
        (
            "「#」キーを押してください",
            "'シャープ'キーを押してください",
            "「シャープ」キーを押してください",
        ),
        ("#を押してください", "シャープを押してください", "シャープを押してください"),
        # 英字の後の「#」は、プログラミング言語や音名なので「シャープ」と読む
        ("C#", "Cシャープ", "Cシャープ"),
        ("F#の曲", "Fシャープの曲", "Fシャープの曲"),
        ("ド♯", "ドシャープ", "ドシャープ"),
    ],
)
def test_normalize_text_number_sign(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「＃５」「#06」のように数字の前の「＃」が「ナンバー」と読まれ、ハッシュタグや顔文字の「＃」は「シャープ」と読まれずに除かれることを確認する。
    「「#」キー」「#を押して」のように「#」そのものを指すときは除かずに「シャープ」と読み、操作の対象が消えないようにする。
    「C#」「F#」のように英字の後の「#」は、従来どおり「シャープ」と読む。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 「#」の後に7〜9で始まる4桁が続く公共の短縮ダイヤルは、「シャープ」と桁読みの漢数字にする
        ## 「シャープ」と番号の間は読点で区切る。「'」で区切ると、Irodori-TTS 向けの経路で1文に2つあるとき鉤括弧の組になる
        ("#7119", "シャープ,七一一九", "シャープ、七一一九"),
        ("＃７１１９に電話", "シャープ,七一一九に電話", "シャープ、七一一九に電話"),
        (
            "急がなくてもいい相談は「＃９１１０」に",
            "急がなくてもいい相談は'シャープ,九一一ゼロ'に",
            "急がなくてもいい相談は「シャープ、九一一ゼロ」に",
        ),
        (
            "#7119または#8000",
            "シャープ,七一一九またはシャープ,八ゼロゼロゼロ",
            "シャープ、七一一九またはシャープ、八ゼロゼロゼロ",
        ),
        ("#8000", "シャープ,八ゼロゼロゼロ", "シャープ、八ゼロゼロゼロ"),
        ("＃８００８", "シャープ,八ゼロゼロ八", "シャープ、八ゼロゼロ八"),
        ("#8103", "シャープ,八一ゼロ三", "シャープ、八一ゼロ三"),
        ("#8778", "シャープ,八七七八", "シャープ、八七七八"),
        ("#8891", "シャープ,八八九一", "シャープ、八八九一"),
        ("#9910", "シャープ,九九一ゼロ", "シャープ、九九一ゼロ"),
        # 7〜9で始まらない4桁、3桁、5桁以上は、番号として「ナンバー」と読む
        ("#1234", "ナンバー1234", "ナンバー1234"),
        ("#719", "ナンバー719", "ナンバー719"),
        ("#71190", "ナンバー71190", "ナンバー71190"),
        # 英字の後の「#」は、従来どおり「シャープ」と読む
        ("C#7119", "Cシャープ7119", "Cシャープ7119"),
    ],
)
def test_normalize_text_short_dial_numbers(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「#7119」(救急安心電話相談) や「#9110」(警察相談) のような公共の短縮ダイヤルが、「ナンバー7119」と位取りで読まれず、
    「シャープ」と桁読みの番号に書き換えられて、コアに「シャープナナイチイチキュー」と読まれることを確認する。
    0は「〇」だとコアが「マル」と読むので、「#8000」は「シャープ、八ゼロゼロゼロ」と書いて「シャープ、ハチゼロゼロゼロ」と読ませる。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_currency():
    """
    円・ドル・ユーロ・ポンドなどの各種通貨記号を伴う金額表記が、適切な日本語の読みへ正規化されることを確認する。
    """

    # 各種通貨記号
    assert normalize_text("$100") == "100ドル"
    assert normalize_text("¥100") == "100円"
    assert normalize_text("€100") == "100ユーロ"
    assert normalize_text("£100") == "100ポンド"
    assert normalize_text("₩1000") == "1000ウォン"
    # 通貨記号の位置による違い
    assert normalize_text("100$") == "100ドル"
    assert normalize_text("100¥") == "100円"
    # 金額の桁区切り
    assert normalize_text("¥1,234,567") == "1234567円"
    # 読点を桁区切りに使った「1、000名」も、単位や助数詞が続くときは桁区切りとして除き、「千名」と読ませる
    assert normalize_text("参加者は1、000名") == "参加者は1000名"
    assert normalize_text("29、002フィート") == "29002フィート"
    assert normalize_text("12、345人") == "12345人"
    # 単位や助数詞が続かない「1、2、3」「2、000」は、数の並びなので読点のまま残す
    assert normalize_text("1、2、3") == "1,2,3"
    assert normalize_text("2、000") == "2,000"
    assert normalize_text("$1,234.56") == "1234.56ドル"
    # 通貨の単位
    assert normalize_text("1億円") == "1億円"
    assert normalize_text("100万ドル") == "100万ドル"
    # 特殊な通貨
    assert normalize_text("₿1.5") == "1.5ビットコイン"
    assert normalize_text("₹100") == "100ルピー"
    assert normalize_text("₽50") == "50ルーブル"
    assert normalize_text("₺25") == "25リラ"
    assert normalize_text("฿1000") == "1000バーツ"
    assert normalize_text("₱100") == "100ペソ"
    assert normalize_text("₴50") == "50フリヴニャ"
    assert normalize_text("₫1000") == "1000ドン"
    assert normalize_text("₪100") == "100シェケル"
    assert normalize_text("₦500") == "500ナイラ"
    assert normalize_text("₡1000") == "1000コロン"


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 丸数字と後ろの数の間に足した読点は、桁区切りではないので残す
        ("①100円", "1,100円"),
        ("①12、345人", "1,12345人"),
        # 原文で読点を桁区切りに使った数は、丸数字の後でも1つの数にまとめる
        ("②1、000円", "2,1000円"),
        # 電話番号の後の空白から足した読点も、後ろの金額とまとめない
        ("0120-123-456 100円", "0120-123-456,100円"),
        ("03-1234-5678 2、000人", "03-1234-5678,2000人"),
        # 数どうしの間の全角空白から足した読点も、1つの数にまとめない
        ("100　200円", "100,200円"),
        # 原文の読点の桁区切りは、従来どおり除く
        ("参加者は1、000名", "参加者は1000名"),
    ],
)
def test_normalize_text_inserted_commas_are_not_digit_separators(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    「①100円」の丸数字の後や、「0120-123-456 100円」の電話番号の後の空白に、正規化が数の連結を防ぐために足した読点が、
    後段で「1、000名」のような原文の読点の桁区切りと取り違えられて除かれ、「1100円」「0120-123-456100円」と別の金額や番号に変わらないことを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        ("1ha", "1ヘクタール", "1ヘクタール"),
        ("8.5ha", "8.5ヘクタール", "八点五ヘクタール"),
        ("１ｈａ以上", "1ヘクタール以上", "1ヘクタール以上"),
        # 漢数字の後の「ｈａ」も、面積の単位として読む
        (
            "三百ｈａにおよぶ園内",
            "三百ヘクタールにおよぶ園内",
            "三百ヘクタールにおよぶ園内",
        ),
        ("百六十八万３千ｈａ", "百六十八万3千ヘクタール", "百六十八万3千ヘクタール"),
        # 数のない「ha」は、ほかの単位の「kg」などと同じく英字のまま残す
        ("ha", "ha", "ha"),
    ],
)
def test_normalize_text_hectare_unit(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「1ha」「８．５ｈａ」「三百ｈａ」のように数の後の「ha」が、英語の「ハー」ではなく面積の単位の「ヘクタール」と読まれることを確認する。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 数の直後の単位記号は、英単語や州名の略語ではなく単位として読む
        ("それは1ctです", "それは1カラットです", "それは1カラットです"),
        ("それは2calです", "それは2カロリーです", "それは2カロリーです"),
        ("それは3ftです", "それは3フィートです", "それは3フィートです"),
        ("それは4lxです", "それは4ルクスです", "それは4ルクスです"),
        ("それは5inです", "それは5インチです", "それは5インチです"),
        ("それは6ktです", "それは6ノットです", "それは6ノットです"),
        # アールの「a」は、数式の係数の「3a」と区別できないので単位にしない
        ("3aの2乗b", "3aの2乗b", "3aの2乗b"),
        ("0.5ct", "0.5カラット", "零点五カラット"),
        # 数のない語や、後ろに英数字が続く語は単位にしない
        ("2in1", "2in1", "2in1"),
        ("made in Japan", "メイドインジャパン", "メイドインジャパン"),
        ("a pen", "アペン", "アペン"),
        # 「cc」はコアが助数詞として「シーシー」とアクセント付きで読むので、カタカナにせずに渡す
        ("それは1ccです", "それは1ccです", "それは1ccです"),
    ],
)
def test_normalize_text_units_after_numbers(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「1ct」「2cal」「3ft」「4lx」「5in」「6kt」のように数の直後に置いた単位記号が、
    「コネチカット」「キャル」「エフティー」「イン」「キロトン」と読まれず、カラット・カロリー・フィート・ルクス・インチ・ノットと読まれることを確認する。
    「in」は英語の前置詞と同じ綴りなので、数の直後にあって後ろに英数字が続かないときだけ単位にする。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_units():
    """
    長さ・重さ・体積・速度・データ容量などの各種単位記号が、数値と組み合わさった際に適切な日本語の読みへ正規化されることを確認する。
    """

    # ページ数表記
    assert normalize_text("40pをご覧ください。") == "40ページをご覧ください."
    assert normalize_text("本文は28p、資料は40p、付録は128pです。") == (
        "本文は28ページ,資料は40ページ,付録は128ページです."
    )
    assert normalize_text("全48pの冊子です。") == "全48ページの冊子です."
    assert normalize_text("発表は3pまで進みました。") == "発表は3ページまで進みました."
    assert normalize_text("第2版は84p構成です。") == "第2版は84ページ構成です."
    assert normalize_text("見出しは1pあたり2段組です。") == (
        "見出しは1ページあたり2段組です."
    )
    assert normalize_text("この本はA4判・96p・フルカラーです。") == (
        "この本はA4判,96ページ,フルカラーです."
    )
    assert normalize_text("漫画は全200P、設定資料集は52Pです。") == (
        "漫画は全200ページ,設定資料集は52ページです."
    )
    assert normalize_text("ページ表記の揺れとして40Ｐや52ｐも含みます。") == (
        "ページ表記の揺れとして40ページや52ページも含みます."
    )
    assert normalize_text("1p目だけ差し替えました。") == "1ページ目だけ差し替えました."
    # ページ数表記として扱わないケース
    assert normalize_text("メモリ使用量は4pではなく4PBです。") == (
        "メモリ使用量は4ページではなく4ペタバイトです."
    )
    assert normalize_text("実験条件はpH7.4です。") == "実験条件はペーハー7.4です."
    assert normalize_text("商品コードはXP40Proです。") == "商品コードはXP40プロです."
    assert normalize_text("URLはhttps://example.com/p/40です。") == (
        "ユーアールエルはエイチティーティーピーエス,イグザンプルドットコム,スラッシュ,p,スラッシュ,40です."
    )
    assert normalize_text("午後3pmに集合です。") == "午後3ピーエムに集合です."
    assert normalize_text("通し番号はNo.40pではありません。") == (
        "通し番号はノー40pではありません."
    )
    assert normalize_text("40ptsのフォントを使う。") == (
        "40ピーティーエスのフォントを使う."
    )

    # 基本的な単位
    assert normalize_text("100m") == "100メートル"
    assert normalize_text("100cm") == "100センチメートル"
    assert normalize_text("1000.19mm") == "1000.19ミリメートル"
    assert normalize_text("1km") == "1キロメートル"
    assert normalize_text("500mL") == "500ミリリットル"
    assert normalize_text("1L") == "1リットル"
    assert normalize_text("1000.19kL") == "1000.19キロリットル"
    assert normalize_text("1000.19mg") == "1000.19ミリグラム"
    assert normalize_text("100g") == "100グラム"
    assert normalize_text("2kg") == "2キログラム"
    assert normalize_text("200万t以上") == "200万トン以上"
    assert normalize_text("21万tを市場に出す") == "21万トンを市場に出す"
    assert normalize_text("697万tから711万t") == "697万トンから711万トン"
    assert normalize_text("200ｍｇ") == "200ミリグラム"  # 全角英数
    # データ容量
    assert normalize_text("51B") == "51バイト"
    assert normalize_text("51KB") == "51キロバイト"
    assert normalize_text("51MB") == "51メガバイト"
    assert normalize_text("51GB") == "51ギガバイト"
    assert normalize_text("51TB") == "51テラバイト"
    assert normalize_text("51PB") == "51ペタバイト"
    assert normalize_text("51EB") == "51エクサバイト"
    assert normalize_text("100KiB") == "100キビバイト"
    assert normalize_text("1000.11MiB") == "1000.11メビバイト"
    assert normalize_text("1000.11GiB") == "1000.11ギビバイト"
    assert normalize_text("1000.11TiB") == "1000.11テビバイト"
    assert normalize_text("1000.11PiB") == "1000.11ペビバイト"
    assert normalize_text("1000.11EiB") == "1000.11エクスビバイト"
    # 面積・体積
    assert normalize_text("100m2") == "100平方メートル"
    assert normalize_text("1km2") == "1平方キロメートル"
    assert normalize_text("50m3") == "50立方メートル"
    # 温度・角度
    assert normalize_text("50℃") == "50度"
    assert normalize_text("5.6°C") == "5.6度"
    assert normalize_text("0.3°c") == "0.3度"
    assert normalize_text("30°C以上") == "30度以上"
    assert normalize_text("1.5°C高い") == "1.5度高い"
    assert normalize_text("32℉") == "32度"
    assert normalize_text("98.6°F") == "98.6度"
    assert normalize_text("100°f") == "100度"
    assert normalize_text("30° F") == "30度"
    assert normalize_text("180°回転") == "180度回転"
    assert normalize_text("-3℃") == "マイナス3度"
    assert normalize_text("ー10.7℃") == "マイナス10.7度"
    assert normalize_text("－0.3°C") == "マイナス0.3度"
    assert normalize_text("+1.5°C") == "プラス1.5度"
    assert normalize_text("-40°F") == "マイナス40度"
    assert normalize_text("+30°") == "プラス30度"
    assert normalize_text("5.6°C", for_irodori=True) == "五ー点六度"
    assert normalize_text("98.6°F", for_irodori=True) == "九十八点六度"
    # 単位付き指数
    assert normalize_text("1.23e-6") == "零点零零零零零一二三"
    assert normalize_text("1.23e+4") == "一万二千三百"
    assert normalize_text("1.23e-4") == "零点零零零一二三"
    assert normalize_text("1e6") == "百万"
    # 単位付きの範囲
    assert normalize_text("100m〜200m") == "100メートルから200メートル"
    assert normalize_text("1kg〜2kg") == "1キログラムから2キログラム"
    assert normalize_text("100dL〜200dL") == "100デシリットルから200デシリットル"
    # ヘルツ
    assert normalize_text("100Hz") == "100ヘルツ"
    assert normalize_text("100kHz") == "100キロヘルツ"  # k が小文字
    assert normalize_text("100KHz") == "100キロヘルツ"  # K が大文字
    assert normalize_text("100MHz") == "100メガヘルツ"
    assert normalize_text("100GHz") == "100ギガヘルツ"
    assert normalize_text("100THz") == "100テラヘルツ"
    assert normalize_text("45.56kHz") == "45.56キロヘルツ"
    # ヘルツ (hz が小文字、表記揺れ対策)
    assert normalize_text("100hz") == "100ヘルツ"
    assert normalize_text("100khz") == "100キロヘルツ"
    assert normalize_text("100Khz") == "100キロヘルツ"
    assert normalize_text("100Mhz") == "100メガヘルツ"
    assert normalize_text("100Ghz") == "100ギガヘルツ"
    assert normalize_text("100Thz") == "100テラヘルツ"
    assert normalize_text("45.56khz") == "45.56キロヘルツ"
    # ヘクトパスカル
    assert normalize_text("100hPa") == "100ヘクトパスカル"
    assert normalize_text("100hpa") == "100ヘクトパスカル"
    assert normalize_text("100HPa") == "100ヘクトパスカル"
    # アンペア
    assert normalize_text("100A") == "100アンペア"
    assert normalize_text("100mA") == "100ミリアンペア"
    assert normalize_text("100kA") == "100キロアンペア"
    assert normalize_text("45.56mA") == "45.56ミリアンペア"
    assert normalize_text("5000mAh") == "5000ミリアンペアアワー"
    assert normalize_text("1.2Ah") == "1.2アンペアアワー"
    # 路線番号や型番の内部にある数字+英字は物理単位として切り出さない
    assert normalize_text("E2A") == "E2A"
    assert normalize_text("65Wh") == "65ワットアワー"
    # bps
    assert normalize_text("100.55bps") == "100.55ビーピーエス"
    assert normalize_text("100kbps") == "100キロビーピーエス"
    assert normalize_text("100Mbps") == "100メガビーピーエス"
    assert normalize_text("100Gbps") == "100ギガビーピーエス"
    assert normalize_text("100Tbps") == "100テラビーピーエス"
    assert normalize_text("100Pbps") == "100ペタビーピーエス"
    assert normalize_text("100Ebps") == "100エクサビーピーエス"
    # ビット
    assert normalize_text("100bit") == "100ビット"
    assert normalize_text("100kbit") == "100キロビット"
    assert normalize_text("100Mbit") == "100メガビット"
    assert normalize_text("100Gbit") == "100ギガビット"
    assert normalize_text("100Tbit") == "100テラビット"
    assert normalize_text("100Pbit") == "100ペタビット"
    assert normalize_text("100Ebit") == "100エクサビット"
    # スラッシュ付き単位（毎分・毎秒）
    assert normalize_text("100m/h") == "100メートル毎時"
    assert normalize_text("100km/h") == "100キロメートル毎時"
    assert normalize_text("5000m/h") == "5000メートル毎時"
    assert normalize_text("3.5km/h") == "3.5キロメートル毎時"
    assert normalize_text("30.56B/h") == "30.56バイト毎時"
    assert normalize_text("30.56kB/h") == "30.56キロバイト毎時"
    assert normalize_text("30.56KB/h") == "30.56キロバイト毎時"
    assert normalize_text("30.56MB/h") == "30.56メガバイト毎時"
    assert normalize_text("30.56GB/h") == "30.56ギガバイト毎時"
    assert normalize_text("30.56TB/h") == "30.56テラバイト毎時"
    assert normalize_text("30.56EB/h") == "30.56エクサバイト毎時"
    assert normalize_text("30.56b/h") == "30.56ビット毎時"
    assert normalize_text("30.56Kb/h") == "30.56キロビット毎時"
    assert normalize_text("30.56Mb/h") == "30.56メガビット毎時"
    assert normalize_text("30.56Gb/h") == "30.56ギガビット毎時"
    assert normalize_text("30.56Tb/h") == "30.56テラビット毎時"
    assert normalize_text("30.56Eb/h") == "30.56エクサビット毎時"
    assert normalize_text("100m/s") == "100メートル毎秒"
    assert normalize_text("100km/h") == "100キロメートル毎時"
    assert normalize_text("5000m/s") == "5000メートル毎秒"
    assert normalize_text("3.5km/s") == "3.5キロメートル毎秒"
    assert normalize_text("30.56B/s") == "30.56バイト毎秒"
    assert normalize_text("30.56kB/s") == "30.56キロバイト毎秒"
    assert normalize_text("30.56KB/s") == "30.56キロバイト毎秒"
    assert normalize_text("30.56MB/s") == "30.56メガバイト毎秒"
    assert normalize_text("30.56GB/s") == "30.56ギガバイト毎秒"
    assert normalize_text("30.56TB/s") == "30.56テラバイト毎秒"
    assert normalize_text("30.56EB/s") == "30.56エクサバイト毎秒"
    assert normalize_text("30.56b/s") == "30.56ビット毎秒"
    assert normalize_text("30.56Kb/s") == "30.56キロビット毎秒"
    assert normalize_text("30.56Mb/s") == "30.56メガビット毎秒"
    assert normalize_text("30.56Gb/s") == "30.56ギガビット毎秒"
    assert normalize_text("30.56Tb/s") == "30.56テラビット毎秒"
    assert normalize_text("30.56Eb/s") == "30.56エクサビット毎秒"
    # スラッシュ付き単位（毎分・毎秒以外の意図的に変換せず pyopenjtalk に任せるパターン）
    assert normalize_text("100kL/m") == "100kL/m"
    assert normalize_text("100g/㎥") == "100g/m3"
    # スラッシュ付き単位ではないので通常通り変換するパターン (dB は変換対象外の単位)
    assert normalize_text("100m/100.50mL/50dB") == "100メートル/100.50ミリリットル/50dB"
    assert normalize_text("100m/秒") == "100メートル/秒"
    # 英単語の後に単位が来るケース
    assert normalize_text("up to 8GB") == "アップトゥー8ギガバイト"
    # 追加のテストケース
    assert normalize_text("100tトラック") == "100トントラック"
    assert normalize_text("100.1919tトラック") == "100.1919トントラック"
    assert normalize_text("345.56t") == "345.56トン"
    assert normalize_text("345.56test") == "345.56テスト"
    assert normalize_text("345.56t") == "345.56トン"
    assert normalize_text("345.56ms") == "345.56ミリ秒"
    assert normalize_text("345ms") == "345ミリ秒"
    assert normalize_text("24hを") == "24時間を"
    assert normalize_text("24h営業") == "24時間営業"
    assert normalize_text("24ms営業") == "24ミリ秒営業"
    assert normalize_text("24s営業") == "24秒営業"
    assert normalize_text("500Kがある") == "500Kがある"
    assert normalize_text("50℃") == "50度"
    assert normalize_text("50ms") == "50ミリ秒"
    assert normalize_text("50s") == "50秒"
    assert normalize_text("50ns") == "50ナノ秒"
    assert normalize_text("50μs") == "50マイクロ秒"
    assert normalize_text("50ms") == "50ミリ秒"
    assert normalize_text("50s") == "50秒"
    assert normalize_text("50h") == "50時間"
    assert normalize_text("1h") == "1時間"
    # 連続時間表記は d/h/m/s をまとめて日本語の時間単位へ変換する
    assert normalize_text("1h3m5s") == "1時間3分5秒"
    assert normalize_text("1h25m23s") == "1時間25分23秒"
    assert normalize_text("24d23h50m4s") == "24日23時間50分4秒"
    assert normalize_text("50m4s") == "50分4秒"
    assert normalize_text("1h3m") == "1時間3分"
    assert normalize_text("1h5s") == "1時間5秒"
    assert normalize_text("300s") == "300秒"
    assert normalize_text("3hで") == "3時間で"
    assert normalize_text("3hzで") == "3ヘルツで"
    assert normalize_text("30d") == "30日"
    assert normalize_text("30dでなんとかした") == "30日でなんとかした"
    assert normalize_text("でも30dでなんとかした") == "でも30日でなんとかした"
    assert normalize_text("でも30daysでなんとかした") == "でも30デイズでなんとかした"
    assert normalize_text("でも30date") == "でも30デート"
    assert normalize_text("でも30dで") == "でも30日で"
    assert normalize_text("でも30Dで") == "でも30Dで"
    assert normalize_text("\\100") == "100円"
    assert normalize_text("$100で") == "100ドルで"
    assert normalize_text("€100で") == "100ユーロで"
    assert normalize_text("5㎞") == "5キロメートル"
    assert normalize_text("5㎡") == "5平方メートル"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 摂氏記号
        ("50℃", "50度"),
        ("5.6℃", "5.6度"),
        ("30℃以上", "30度以上"),
        ("1.5℃高い", "1.5度高い"),
        # 摂氏の ASCII 表記
        ("50°C", "50度"),
        ("50° C", "50度"),
        ("50°c", "50度"),
        ("50° c", "50度"),
        ("30°C以上", "30度以上"),
        ("1.5°C高い", "1.5度高い"),
        # 既に `°` が `度` へ置換された後の表記
        ("50度C", "50度"),
        ("50度 C", "50度"),
        ("50度c", "50度"),
        ("50度 c", "50度"),
        # 華氏記号
        ("32℉", "32度"),
        ("98.6℉", "98.6度"),
        ("100℉以上", "100度以上"),
        ("1.5℉高い", "1.5度高い"),
        # 華氏の ASCII 表記
        ("98.6°F", "98.6度"),
        ("98.6° F", "98.6度"),
        ("100°f", "100度"),
        ("100° f", "100度"),
        ("32°F以上", "32度以上"),
        ("1.5°F高い", "1.5度高い"),
        # 既に `°` が `度` へ置換された後の華氏表記
        ("98.6度F", "98.6度"),
        ("98.6度 F", "98.6度"),
        ("100度f", "100度"),
        ("100度 f", "100度"),
        # 角度記号
        ("90°", "90度"),
        ("180°回転", "180度回転"),
        ("45°傾ける", "45度傾ける"),
        ("90度", "90度"),
        ("180度回転", "180度回転"),
        # 符号付き表記
        ("-3℃", "マイナス3度"),
        ("−3℃", "マイナス3度"),
        ("－3℃", "マイナス3度"),
        ("ー3℃", "マイナス3度"),
        ("+1.5°C", "プラス1.5度"),
        ("-40°F", "マイナス40度"),
        ("+30°", "プラス30度"),
        # 全角英数記号
        ("５０℃", "50度"),
        ("５０°Ｃ", "50度"),
        ("９８．６°Ｆ", "98.6度"),
        ("１８０°", "180度"),
    ],
)
def test_normalize_text_degree_units(text: str, expected: str):
    """
    摂氏・華氏や角度などの度数表記が、口頭読みの「度」へ適切に正規化されることを確認する。
    """

    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Irodori-TTS 向けでも単位の読みは SBV2 と同じく「度」に畳む
        ("5.6℃", "五ー点六度"),
        ("5.6°C", "五ー点六度"),
        ("5.6度C", "五ー点六度"),
        ("98.6℉", "九十八点六度"),
        ("98.6°F", "九十八点六度"),
        ("98.6度F", "九十八点六度"),
        ("180°回転", "180度回転"),
        ("+30°", "プラス30度"),
    ],
)
def test_normalize_text_degree_units_for_irodori(text: str, expected: str):
    """
    Irodori-TTS 向けにおいても、摂氏・華氏や角度などの度数表記が同じ読みへ適切に正規化されることを確認する。
    """

    assert normalize_text(text, for_irodori=True) == expected


def test_normalize_text_chemical_formula_like_words():
    """
    化学式風の英数字表記において、元素記号や数値が崩れずに適切な読み上げテキストへ正規化されることを確認する。
    """

    assert normalize_text("CO2濃度を測定する。") == "シーオーツー濃度を測定する."
    assert normalize_text("CO2") == "シーオーツー"
    assert (
        normalize_text("CO2センサーを校正する。") == "シーオーツーセンサーを校正する."
    )
    assert normalize_text("CO2/CH4/O2を比較する。") == (
        "シーオーツー/シーエイチフォー/オーツーを比較する."
    )
    assert normalize_text("H2Oを加える。") == "エイチツーオーを加える."
    assert normalize_text("N2を封入する。") == "エヌツーを封入する."
    assert normalize_text("NO2濃度に注意する。") == "エヌオーツー濃度に注意する."
    assert (
        normalize_text("NH3センサーを交換する。")
        == "エヌエイチスリーセンサーを交換する."
    )
    assert normalize_text("SO2排出量を監視する。") == "エスオーツー排出量を監視する."
    assert normalize_text("HClで洗浄する。") == "エイチシーエルで洗浄する."
    assert normalize_text("NaCl水溶液を作る。") == "エヌエーシーエル水溶液を作る."
    assert normalize_text("O3発生器を停止する。") == "オースリー発生器を停止する."
    assert normalize_text("CO/CO2/O2を同時測定する。") == (
        "シーオー/シーオーツー/オーツーを同時測定する."
    )
    # 化学式風でない英数字は従来通りの挙動を維持する
    assert normalize_text("pH7.4を維持する。") == "ペーハー7.4を維持する."
    assert normalize_text("CPUを8基で学習した。") == "シーピーユーを8基で学習した."


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("❹③に重しをし、", "4,3に重しをし,"),
        ("❷①を水で洗って", "2,1を水で洗って"),
        ("❹①に②と③を加えて", "4,1に2と3を加えて"),
        ("➋①", "2,1"),
        ("⓫③", "11,3"),
        ("❹➂", "4,3"),
        ("❿⑩", "10,10"),
        ("➓㉑", "10,21"),
        ("❹③❷①", "4,3,2,1"),
    ],
)
def test_normalize_text_black_and_white_circled_numbers(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    黒丸数字から白丸数字へと続く手順番号が読点で区切られ、複数の組がある場合でも各数字が分離して読まれることを確認する。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("❶", "1"),
        ("①", "1"),
        ("❶❷❸❹❺❻❼❽❾❿", "12345678910"),
        ("⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴", "11121314151617181920"),
        ("➊➋➌➍➎➏➐➑➒➓", "12345678910"),
        ("①②③④⑤⑥⑦⑧⑨⑩", "12345678910"),
        ("➀➁➂➃➄➅➆➇➈➉", "12345678910"),
        ("①1回", "1,1回"),
        ("12③", "12,3"),
        # 丸数字の直後の漢字やカタカナの語は、数に付く助数詞ではなく項目の中身なので、読点で区切る
        ("②薬を飲む", "2,薬を飲む"),
        ("③報告", "3,報告"),
        ("①社内調査", "1,社内調査"),
        ("②一賃金", "2,一賃金"),
        ("④テスト", "4,テスト"),
        # 丸数字の後の助詞は項目の番号を受けるので、区切らない
        ("②と③を加えて", "2と3を加えて"),
    ],
)
def test_normalize_text_circled_number_non_targets(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    同種の丸数字が連続する場合や、数字に隣接する読点の区切り方が意図通りに保持されることを確認する。
    「②薬」のように丸数字の直後に漢字やカタカナの語が続くと、コアが「2薬」を数と助数詞の「ニヤク」と読むので、読点で区切る。
    """

    if for_irodori is True:
        expected = expected.replace(",", "、")
    assert normalize_text(text, for_irodori=for_irodori) == expected


def test_normalize_text_enclosed_characters():
    """
    丸数字や囲み英数字、囲み漢字などの各種囲み文字が、適切な文字や単語へ正規化されることを確認する。
    """

    # 丸付き数字
    assert normalize_text("①②③④⑤⑥⑦⑧⑨⑩") == "12345678910"
    assert normalize_text("⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳") == "11121314151617181920"
    # NFKC で展開されない装飾付き数字も、通常の丸付き数字と同じ数値へ変換する
    assert normalize_text("❶❷❸❹❺❻❼❽❾❿") == "12345678910"
    assert normalize_text("⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴") == "11121314151617181920"
    assert normalize_text("➀➁➂➃➄➅➆➇➈➉") == "12345678910"
    assert normalize_text("➊➋➌➍➎➏➐➑➒➓") == "12345678910"
    # 英単語直後の装飾付き数字を Python の int() へ直接渡さず、先に ASCII 数字へ直す
    assert normalize_text("Chapter❻", for_irodori=True) == "チャプターシックス"
    assert normalize_text("❻今後の課題", for_irodori=True) == "6、今後の課題"
    # 囲み文字（漢字）
    assert normalize_text("㈱") == "株式会社"
    assert normalize_text("㈲") == "有限会社"
    assert normalize_text("㈳") == "社団法人"
    assert normalize_text("㈴") == "合名会社"
    assert normalize_text("㈵") == "特殊法人"
    assert normalize_text("㈶") == "財団法人"
    assert normalize_text("㈷") == "祝日"
    assert normalize_text("㈸") == "労働組合"
    assert normalize_text("㈹") == "代表電話"
    assert normalize_text("㈺") == "呼出し電話"
    assert normalize_text("㈻") == "学校法人"
    assert normalize_text("㈼") == "監査法人"
    assert normalize_text("㈽") == "企業組合"
    assert normalize_text("㈾") == "合資会社"
    assert normalize_text("㈿") == "協同組合"
    assert normalize_text("㊤㊥㊦") == "上中下"
    assert normalize_text("㊧㊨") == "左右"
    assert normalize_text("㊩") == "医療法人"
    assert normalize_text("㊪") == "宗教法人"
    assert normalize_text("㊫") == "学校法人"
    assert normalize_text("㊬") == "監査法人"
    assert normalize_text("㊭") == "企業組合"
    assert normalize_text("㊮") == "合資会社"
    assert normalize_text("㊯") == "協同組合"


@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        # 図表番号の区切りを消して「424」と1つの数にまとめない
        ("４‐２‐４‐十三図", "4-2-4-十三図", "4-2-4-十三図"),
        ("２‐９‐十五表", "2-9-十五表", "2-9-十五表"),
        # 節番号の2つ目の点を消して「2.43」と小数にまとめない
        ## 区切りごとの数を「点」でつなぎ、Irodori-TTS 向けには漢数字にする
        ("２．４．３　節の内容", "2点4点3,節の内容", "二点四点三、節の内容"),
        (
            "バージョン７．２．１が利用できる",
            "バージョン7点2点1が利用できる",
            "バージョン七点二点一が利用できる",
        ),
        ("６．３．１．　組織", "6点3点1.,組織", "六点三点一。、組織"),
        # 3桁以上の項や先頭のゼロを含む番号は、整数に読み替えず区切りを保つ
        ("バージョン1.02.003", "バージョン1.02.003", "バージョン1。02。003"),
        ("192.168.0.1", "192.168.0.1", "192。168。0。1"),
        # 丸数字の直後の数字と合わせて「11回」のように1つの数にしない
        ("①１回の額", "1,1回の額", "1、1回の額"),
        ("②５ｍｍごとに", "2,5ミリメートルごとに", "2、5ミリメートルごとに"),
        # 1文に丸数字と数字の組が2つあっても、区切りが鉤括弧の組に変わらない
        ("①１月の額、②２月の額", "1,1月の額,2,2月の額", "1、1月の額、2、2月の額"),
        # 区切りのない数や小数はそのまま読む
        ("４‐２‐４", "4-2-4", "4-2-4"),
        ("①回目", "1回目", "1回目"),
        # 後段で分数や時刻に変わる数字とも、丸数字を連結しない
        ("①1/2カップ", "1,二ぶんの一カップ", "1、二ぶんの一カップ"),
        ("①12:30に集合", "1,十二時30分に集合", "1、十二時30分に集合"),
        # 条番号に続く項の丸数字も、直前の数と合わせて「2341」のような1つの数にしない
        ("（所法二百三十四①）", "'所法二百三十四,1'", "（所法二百三十四、1）"),
        ("第２３条①項", "第23条1項", "第23条1項"),
        ("１２③", "12,3", "12、3"),
        # 小数を含む比は、小数点で区切らず「タイ」でつなぐ
        (
            "比はおよそ１：１．６１８になる",
            "比はおよそ一タイ一点六一八になる",
            "比はおよそ一タイ一点六一八になる",
        ),
        ("比は１：１", "比は一タイ一", "比は一タイ一"),
        ("１２：３０．５", "12分30秒5", "12分30秒5"),
    ],
)
def test_normalize_text_separated_number_sequences(
    text: str, expected: str, expected_irodori: str
) -> None:
    """
    「４‐２‐４‐十三図」「２．４．３」「①１回」のように記号で区切った数字の並びが、
    英単語の変換や NFKC で区切りを失って「424」「2.43」「11」のような1つの数として読まれないことを確認する。
    """

    assert normalize_text(text) == expected
    assert normalize_text(text, for_irodori=True) == expected_irodori


def test_normalize_text_itaiji():
    """
    人名や地名などに用いられる旧字体や異体字が、標準的な新字体へ適切に変換されることを確認する。
    """

    # 基本的な旧字体→新字体の変換
    assert normalize_text("學校") == "学校"
    assert normalize_text("國語") == "国語"
    assert normalize_text("經濟") == "経済"
    assert normalize_text("醫學") == "医学"
    assert normalize_text("圖書館") == "図書館"
    assert normalize_text("鐵道") == "鉄道"
    assert normalize_text("歷史") == "歴史"
    assert normalize_text("實驗") == "実験"
    assert normalize_text("體育") == "体育"
    assert normalize_text("變化") == "変化"

    # 旧字体を含む文
    assert normalize_text("國語の學校で勉強する。") == "国語の学校で勉強する."
    assert normalize_text("圖書館で經濟學を學ぶ。") == "図書館で経済学を学ぶ."
    assert normalize_text("醫學部の實驗は嚴しい。") == "医学部の実験は厳しい."

    # 新字体のみの文はそのまま通過する
    assert normalize_text("学校で勉強する。") == "学校で勉強する."
    assert normalize_text("図書館で本を読む。") == "図書館で本を読む."

    # 旧字体と新字体が混在する文
    assert normalize_text("學校と図書館で勉強する。") == "学校と図書館で勉強する."

    # 旧字体と他の正規化処理が組み合わさるケース
    # 旧字体変換 + 日付正規化
    assert normalize_text("2024/01/01に學校へ行く。") == "2024年1月1日に学校へ行く."
    # 旧字体変換 + 英単語カタカナ変換
    assert normalize_text("經濟のNewsを讀む。") == "経済のニューズを読む."
    # 旧字体変換 + 単位変換
    assert normalize_text("學校まで3km歩く。") == "学校まで3キロメートル歩く."

    # 複数の旧字体が連続するケース
    assert normalize_text("國際經濟學") == "国際経済学"
    assert normalize_text("勸業銀行") == "勧業銀行"
    assert normalize_text("總務省") == "総務省"
    assert normalize_text("辯護士") == "弁護士"
    assert normalize_text("營業權") == "営業権"

    # 辨・瓣・辯 はいずれも「弁」に変換される
    assert normalize_text("辨當") == "弁当"
    assert normalize_text("花瓣") == "花弁"
    assert normalize_text("辯論") == "弁論"

    # 旧字体が含まれる固有名詞的な用例
    assert normalize_text("龍が如く") == "竜が如く"
    assert normalize_text("櫻の花が咲く。") == "桜の花が咲く."
    assert normalize_text("澤山の寶物") == "沢山の宝物"


def test_normalize_text_cjk_compatibility_ideographs():
    """
    NFKC 正規化で統合漢字に変換されない互換漢字や CJK 拡張漢字が、文字種クリーンアップ処理で欠落することなく保持されることを確認する。
    """

    # ITAIJI_MAP に登録済みの互換漢字・拡張漢字は通用字へ変換される
    assert normalize_text("黒﨑さん") == "黒崎さん"  # U+FA11
    assert normalize_text("﨔の木") == "欅の木"  # U+FA14
    assert normalize_text("𠮷野家") == "吉野家"  # U+20BB7 (CJK 拡張 B)
    assert normalize_text("𡈽井さん") == "土井さん"  # U+2123D (CJK 拡張 B)

    # NFKC で統合漢字へ正規化される互換漢字はそのまま変換される
    assert normalize_text("神社") == "神社"  # U+FA19 神 -> U+795E 神

    # ITAIJI_MAP 未登録の互換漢字・拡張漢字も削除されず未知語として残る
    assert normalize_text("﨎") == "﨎"  # U+FA0E (通用字対応が不明確な IBM 拡張漢字)

    # BMP 内の人名異体字 (削除対象になったことはないが、回帰防止として固定)
    assert normalize_text("髙橋さん") == "髙橋さん"  # U+9AD9 はしご高
    assert normalize_text("草彅剛") == "草彅剛"  # U+5F45


def test_normalize_text_japanese_unicode_blocks_keep_surface() -> None:
    """
    ひらがな・カタカナ・漢字をはじめ、日本語の表層として出現し得る各種 Unicode ブロックの文字が削除されず保持されることを確認する。
    """

    # 二の字点は pyopenjtalk が読みに展開しないため、直前の漢字を繰り返す
    assert normalize_text("人〻") == "人人"  # U+303B
    assert normalize_text("山〻") == "山山"  # U+303B

    # CJK 記号と句読点内の日本語文字は、辞書表層から欠けないように残す
    assert normalize_text("締〆") == "締〆"  # U+3006
    assert normalize_text("〱〲〳〴〵") == "〱〲〳〴〵"  # U+3031-U+3035

    # かな系の追加ブロックは、固有名詞や引用文の表層として残す
    assert normalize_text("変体仮名𛀁") == "変体仮名𛀁"  # U+1B001
    assert normalize_text("変体仮名𛄀") == "変体仮名𛄀"  # U+1B100
    assert normalize_text("小書き𛅐") == "小書き𛅐"  # U+1B150
    assert normalize_text("かな𚿰") == "かな𚿰"  # U+1AFF0

    # 部首・筆画は読めない可能性があっても、漢字表層の一部として残す
    assert normalize_text("部首⺅") == "部首⺅"  # U+2E85
    assert normalize_text("康熙⼀") == "康熙一"  # U+2F00 は NFKC で一へ統合
    assert normalize_text("筆画㇀") == "筆画㇀"  # U+31C0

    # CJK 拡張 G 以降も、既存の拡張 B〜F と同じく未知語として残す
    assert normalize_text("拡張G𰀀") == "拡張G𰀀"  # U+30000
    assert normalize_text("拡張H𱍐") == "拡張H𱍐"  # U+31350
    assert normalize_text("拡張J𲎰") == "拡張J𲎰"  # U+323B0

    # Irodori-TTS 側も同じ日本語文字範囲を使い、句読点だけ専用表記にする
    assert normalize_text("人〻、変体仮名𛀁", for_irodori=True) == "人人、変体仮名𛀁"


def test_normalize_text_non_japanese_unicode_blocks_are_removed() -> None:
    """
    日本語の読み上げ対象外として定義された Unicode ブロックの文字が、適切に除去されることを確認する。
    """

    # 外国語文字は日本語向け normalizer の対象外として削除する
    assert normalize_text("東京가서울") == "東京"  # Hangul Syllables
    assert normalize_text("東京ЖМосква") == "東京"  # Cyrillic
    assert normalize_text("東京عربي") == "東京"  # Arabic

    # 結合文字は NFKC で合成できる日本語濁点だけ合成し、単独の装飾用結合文字は削除する
    assert normalize_text("か\u3099") == "が"  # U+3099
    assert normalize_text("は\u309a") == "ぱ"  # U+309A
    assert normalize_text("あ\u0301あ") == "ああ"  # Combining Acute Accent

    # 異体字セレクタは発音を持たないため削除するが、直前の漢字は残す
    assert normalize_text("黒\ufe00崎") == "黒崎"  # Variation Selectors
    assert normalize_text("黒\U000e0100崎") == "黒崎"  # Variation Selectors Supplement


def test_normalize_text_english():
    """
    英単語や英文表記が、辞書や発音規則に基づいて適切なカタカナ読みへ正規化されることを確認する。
    """

    # 基本的な英単語
    assert normalize_text("Hello") == "ハロー"
    assert normalize_text("Good Morning") == "グッドモーニング"
    assert normalize_text("Node.js") == "ノードジェイエス"
    # 複数形
    assert normalize_text("computers") == "コンピューターズ"
    assert normalize_text("smartphones") == "スマートフォーンズ"
    assert (
        normalize_text("chatgpts")
        == "チャットジーピーティーズ"  # "chatgpt" しか辞書に含まれていない場合に自動的に s をつけて読む
    )
    # CamelCase
    assert normalize_text("JavaScript") == "ジャバスクリプト"
    assert normalize_text("TypeScript") == "タイプスクリプト"
    assert normalize_text("RockchipTechnologies") == "ロックチップテクノロジーズ"
    assert normalize_text("splitTextAndImage()") == "スプリットテキストアンドイメージ''"
    # 複合語
    assert normalize_text("e-mail") == "イーメール"
    assert normalize_text("YouTube") == "ユーチューブ"
    # 辞書にない単語の変換
    assert (
        # 小文字の場合は2単語へ分割して辞書で変換
        normalize_text("windsurfeditor") == "ウインドサーフエディター"
    )
    assert (
        # 大文字の場合は2単語への分割が行われないので、C2K によるカタカナ読み推定結果が返る
        normalize_text("WINDSURFEDITOR") == "ウィンサーフデディター"
    )
    assert normalize_text("DevinProgrammerAgents") == "デビンプログラマーエージェンツ"
    # クオートの正規化処理
    assert normalize_text("I'm") == "アイム"
    assert normalize_text("I’m") == "アイム"
    assert normalize_text("We've") == "ウイブ"
    assert normalize_text("We’ve") == "ウイブ"

    # 敬称の処理
    assert normalize_text("Mr. John") == "ミスタージョン"
    assert normalize_text("Mrs. Smith") == "ミセススミス"
    assert normalize_text("Ms. Jane") == "ミズジェーン"
    assert normalize_text("Dr. Brown") == "ドクターブラウン"
    assert normalize_text("John Smith Jr.") == "ジョンスミスジュニア"
    assert normalize_text("John Smith Sr.") == "ジョンスミスシニア"
    assert normalize_text("Mr Smith") == "ミスタースミス"  # ピリオドなし
    assert normalize_text("Dr Brown") == "ドクターブラウン"  # ピリオドなし
    assert normalize_text("Mr. and Mrs. Smith") == "ミスターアンドミセススミス"
    assert normalize_text("Dr. Smith Jr.") == "ドクタースミスジュニア"
    assert (
        normalize_text("Mr. John Smith Jr. PhD")
        == "ミスタージョンスミスジュニアピーエイチディー"
    )
    assert normalize_text("John's book") == "ジョンズブック"
    assert normalize_text("The company's policy") == "ザカンパニーズポリシー"

    # 英単語の後に数字が来る場合
    assert normalize_text("GPT-3") == "ジーピーティースリー"
    assert normalize_text("GPT-11") == "ジーピーティーイレブン"
    assert (
        # 小数点は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("GPT-4.5") == "ジーピーティー4.5"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("GPT-12") == "ジーピーティー12"
    )
    assert normalize_text("iPhone11") == "アイフォンイレブン"
    assert normalize_text("iPhone 8") == "アイフォンエイト"
    assert normalize_text("iPhone 9 Pro Max") == "アイフォンナインプロマックス"
    assert normalize_text("Claude 3") == "クロードスリー"
    assert normalize_text("Pixel 8") == "ピクセルエイト"
    assert (
        # 09 のように数字が0埋めされている場合は変換しない
        normalize_text("Pixel 09") == "ピクセル09"
    )
    assert (
        # 8a のように数字の後にスペースなしで何か付く場合は変換しない
        normalize_text("Pixel 8a") == "ピクセル8a"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("iPhone12") == "アイフォン12"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("Android 14") == "アンドロイド14"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("Windows 95") == "ウィンドウズ95"
    )
    assert (
        # 12 以降は変換しない (pyopenjtalk で日本語読みされる)
        normalize_text("Windows95") == "ウィンドウズ95"
    )
    assert normalize_text("Gemini-2") == "ジェミニツー"
    assert (
        # 小数点は pyopenjtalk に任せた方が良い読みになるので変換しない
        normalize_text("Gemini-1.5") == "ジェミニ1.5"
    )
    assert normalize_text("Gemini 2") == "ジェミニツー"
    assert (
        # 小数点は pyopenjtalk に任せた方が良い読みになるので変換しない
        normalize_text("Gemini 2.0") == "ジェミニ2.0"
    )
    assert (
        # ハイフンが Non-Breaking Hyphen になっている
        normalize_text("GPT‑4") == "ジーピーティーフォー"
    )

    # 単数を表す "a" の処理
    assert normalize_text("a pen") == "アペン"
    assert normalize_text("a book") == "アブック"
    assert normalize_text("a student") == "アスチューデント"
    assert normalize_text("not a pen") == "ノットアペン"
    assert (
        # OFDMEXA は辞書未収録の造語かつ全て大文字で、NGram によって英単語として読むべきと判定されるのでそのまま
        normalize_text("a OFDMEXA modular") == "アOFDMEXAモジュラー"
    )
    # e2k の2/3/4-gramを固定順で評価し、プロセスのハッシュ順による判定揺れを防ぐ
    ## EISEIGIYO は正しい重み順ではアルファベット読み、PFOA も意図どおり OpenJTalk 側へ委ねる
    assert normalize_text("EISEIGIYO") == "EISEIGIYO"
    assert normalize_text("PFOA") == "PFOA"
    assert normalize_text("a 123") == "a123"  # 数字の前の a はそのまま
    assert normalize_text("This is a pen.") == "ディスイズアペン."
    assert normalize_text("This is a good pen.") == "ディスイズアグッドペン."

    # ハイフンで区切られた英単語の処理
    assert normalize_text("pen") == "ペン"
    assert normalize_text("good-pen") == "グッドペン"
    assert normalize_text("OFDMEXA-modular") == "OFDMEXAモジュラー"
    assert (
        # "Bentol" は適当にでっち上げた造語なので C2K によってカタカナ推定が入り、それ以外は辞書からカタカナ表記が取得される
        normalize_text("Bentol-API-SpecificationResult2")
        == "ベントルエーピーアイスペシフィケーションリザルトツー"
    )
    assert (
        # "Paravoice" は辞書にない造語なので、C2K によってカタカナ推定が入る
        # "OTAMESHI" は辞書にない単語だがローマ字として解釈可能なため、ローマ字読みされる
        normalize_text("Paravoice 3を4GBまでOTAMESHIできます")
        == "パラヴォイススリーを4ギガバイトまでオータメシできます"
    )

    # ローマ字読み (辞書に存在するもの)
    assert normalize_text("Akagi") == "アカギ"
    assert normalize_text("Akagisan") == "アカギサン"
    assert normalize_text("Akahata") == "アカハタ"
    assert normalize_text("Akasaka") == "アカサカ"
    assert normalize_text("Akashi") == "アカシ"
    assert normalize_text("Akashiyaki") == "アカシヤキ"
    assert normalize_text("Akebono") == "アケボノ"
    assert normalize_text("Akihabara") == "アキハバラ"
    assert normalize_text("Akita") == "アキタ"
    assert normalize_text("Akitainu") == "アキタイヌ"
    assert normalize_text("Amae") == "アマエ"
    assert normalize_text("Amagasaki") == "アマガサキ"
    assert normalize_text("Amakusa") == "アマクサ"
    assert normalize_text("Amazake") == "アマザケ"
    assert normalize_text("Amezaiku") == "アメザイク"

    # ローマ字読み (C2K でいい感じに自動推定される)
    assert (
        normalize_text("KONO DENSHAWA YAMANOTESEN UCHIMAWARI")
        == "コノデンシャワヤマノテセンウチマワリ"
    )
    assert (
        normalize_text("Next Station Is Musashi-Mizonokuchi.")
        == "ネクストステイションイズムサシミゾノクチ."
    )
    assert normalize_text("ConoHa") == "コノハ"
    assert normalize_text("KonoHa") == "コノハ"
    assert normalize_text("QonoHa") == "コノハ"

    # 長い英文
    assert (
        normalize_text(
            "GPT-4 can solve difficult problems with greater accuracy, thanks to its broader general knowledge and problem solving abilities."
        )
        == "ジーピーティーフォーキャンソルブディフィカルトプロブレムズウィズグレーターアキュラシー,サンクストゥーイツブローダージェネラルナレッジアンドプロブレムソルビングアビリティーズ."
    )
    assert (
        # 小数点は pyopenjtalk に任せた方が良い読みになるので変換しない
        normalize_text(
            "We’re releasing a research preview of GPT‑4.5—our largest and best model for chat yet. GPT‑4.5 is a step forward in scaling up pre-training and post-training. By scaling unsupervised learning, GPT‑4.5 improves its ability to recognize patterns, draw connections, and generate creative insights without reasoning."
        )
        == "ウイアーリリーシングアリサーチプレビューオブジーピーティー4.5-アワーラージェストアンドベストモデルフォーチャットイェット.ジーピーティー4.5イズアステップフォーワードインスケーリングアッププリートレーニングアンドポストトレーニング.バイスケーリングアンスーパーバイズドラーニング,ジーピーティー4.5インプルーブズイツアビリティートゥーレコグナイズパターンズ,ドローコネクションズ,アンドジェネレートクリエイティブインサイツウィザウトリーズニング."
    )

    # 複雑な CamelCase と英文の混合パターン (TODO: 改善の余地あり)
    assert (
        normalize_text(
            "ではCinamicさん、WindsurfCascade-PriceはGemini+Claude&Deepseekesより安いか分かりますか？"
        )
        == "ではシナマイクさん,ウインドサーフカスケードプライスはジェミニプラスクロード&ディープシークエスより安いか分かりますか?"
    )
    assert (
        normalize_text("I'm human, with ApplePencil. Because, We have iPhone 8.")
        == "アイムヒューマン,ウィズアップルペンシル.ビコーズ,ウィーハブアイフォンエイト."
    )
    assert (
        # "GPT" は辞書に登録されているため "ジーピーティー" に変換される
        normalize_text("ModelTrainingWithGPT4TurboAndLlama3")
        == "モデルトレーニングウィズジーピーティー4ターボアンドラマスリー"
    )
    assert (
        normalize_text("NextGenCloudComputingSystem2024")
        == "ネクストジェンクラウドコンピューティングシステム2024"
    )
    assert (
        normalize_text("MachineLearning+DeepLearning=AI")
        == "マシンラーニングプラスディープラーニングイコールエーアイ"
    )
    assert (
        # "iPhone" は辞書に登録されているため "アイフォン" に変換される
        normalize_text("iPhoneProMax15-vs-GooglePixel8 Pro")
        == "アイフォンプロマックス15バーサスグーグルピクセルエイトプロ"
    )
    assert (
        normalize_text("WebDev2023: HTML5+CSS3+JavaScript")
        == "ウェブデブ2023,エイチティーエムエルファイブプラスシーエスエススリープラスジャバスクリプト"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 商品名の「POCARI SWEAT」は、推定で「ポカリスウェイト」になる
        ("POCARISWEATを飲む", "ポカリスエットを飲む"),
        # 遺伝子の「cDNA」と肝機能の検査値の「GOT」は、略語として1文字ずつ読む
        ("cDNAの合成", "シーディーエヌエーの合成"),
        ("GOTの値", "ジーオーティーの値"),
        ("got", "ゴット"),
    ],
)
def test_normalize_text_katakana_map_words(text: str, expected: str) -> None:
    """
    英字のカタカナ読みの辞書に足した商品名や略語が、推定の読み (「ポカリスウェイト」「シーディーナ」「ゴット」) にならず、辞書の読みで読まれることを確認する。
    「cDNA」「GOT」は大文字と小文字を区別する略語の行なので、英語の「got」は従来どおり「ゴット」と読む。
    """

    assert normalize_text(text) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ｎｏ．１", "ナンバー1"),
        ("No.124", "ナンバー124"),
        ("NO.二百七十三", "ナンバー二百七十三"),
        ("no.九十五", "ナンバー九十五"),
        ("ｎｏ．１２４", "ナンバー124"),
        ("No. 12", "ナンバー12"),
        ("No.　12", "ナンバー12"),
        ("No.0", "ナンバー0"),
        ("No.❸", "ナンバー3"),
        ("No.③", "ナンバー3"),
        ("番号No.12の札", "番号ナンバー12の札"),
        ("No.百二十四を紹介", "ナンバー百二十四を紹介"),
    ],
)
def test_normalize_text_number_labels(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    算用数字や漢数字が後続する「No.」表記が「ナンバー」へ書き換えられ、全角表記や丸数字にも同様に適用されることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == expected


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        ("No, thank you", "ノー,サンクユー", "ノー、サンクユー"),
        ("No.と書かれた札", "ノー.と書かれた札", "ノー。と書かれた札"),
        ("No.", "ノー.", "ノー。"),
        ("No.40p", "ノー40p", "ノー40p"),
        ("受付番号No.A-102", "受付番号ノーA102", "受付番号ノーA102"),
        ("Piano.1", "ピアノ1", "ピアノ1"),
        ("NO1", "ナンバーワン", "ナンバーワン"),
    ],
)
def test_normalize_text_number_label_non_targets(
    text: str, expected: str, expected_irodori: str, for_irodori: bool
) -> None:
    """
    数字が後続しない「No.」や英字を含む通し番号、ピリオドのない表記が「ナンバー」へ誤変換されず、英語読みが保たれることを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected", "expected_irodori"),
    [
        (
            "He said no. 3 people heard him.",
            "ヒーセッドノー.3ピープルハードヒム.",
            "ヒーセッドノー。3ピープルハードヒム。",
        ),
        (
            "She said No. 12 people left.",
            "シーセッドノー.12ピープルレフト.",
            "シーセッドノー。12ピープルレフト。",
        ),
        (
            "He answered no. 3 friends listened.",
            "ヒーアンサードノー.3フレンズリスンド.",
            "ヒーアンサードノー。3フレンズリスンド。",
        ),
        (
            "He said ｎｏ． ３ people heard him.",
            "ヒーセッドノー.3ピープルハードヒム.",
            "ヒーセッドノー。3ピープルハードヒム。",
        ),
        (
            "He said no. 1,000 people heard him.",
            "ヒーセッドノー.1000ピープルハードヒム.",
            "ヒーセッドノー。1000ピープルハードヒム。",
        ),
        (
            "He said no. 3.5 people heard him.",
            "ヒーセッドノー.3.5ピープルハードヒム.",
            "ヒーセッドノー。三点五ピープルハードヒム。",
        ),
        (
            "He said no. 3 people heard him. He said no. 3 people heard him.",
            "ヒーセッドノー.3ピープルハードヒム.ヒーセッドノー.3ピープルハードヒム.",
            "ヒーセッドノー。3ピープルハードヒム。ヒーセッドノー。3ピープルハードヒム。",
        ),
        ("No.124", "ナンバー124", "ナンバー124"),
        ("No. 12", "ナンバー12", "ナンバー12"),
        (
            "Please read No. 12.",
            "プリーズリードナンバー12.",
            "プリーズリードナンバー12。",
        ),
        ("page No.124", "ページナンバー124", "ページナンバー124"),
    ],
)
def test_normalize_text_number_labels_preserve_english_negation(
    text: str, expected: str, expected_irodori: str, for_irodori: bool
) -> None:
    """
    前後に英単語が存在する英文中の「no.」が否定語として維持され、番号ラベルへ誤変換されないことを確認する。
    """

    assert normalize_text(text, for_irodori=for_irodori) == (
        expected_irodori if for_irodori is True else expected
    )


@pytest.mark.parametrize("for_irodori", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ＯＬのとき", "オーエルのとき"),
        ("ＮＥＷＳのメンバー", "ニュースのメンバー"),
        ("ＭＣ", "エムシー"),
        ("ＦＡ", "エフエー"),
        ("ＴＯＫＩＯライブ", "トキオライブ"),
        ("ＣＯＰ６", "シーオーピーシックス"),
        ("ＣＯＰ３", "シーオーピースリー"),
        ("cop", "コップ"),
        ("fa", "ファ"),
        ("mc", "マック"),
        ("ol", "オル"),
        ("news", "ニューズ"),
        ("tokio", "トーキョー"),
        ("iPhone11", "アイフォンイレブン"),
        ("Ｖｏ．の２人", "ボーカル.の2人"),
        ("Vo", "ボーカル"),
        ("VO", "ブイオー"),
        ("Vo2", "ブイオーツー"),
        ("Vo2.", "ブイオーツー"),
        ("Vo2。", "ブイオーツー."),
        ("Ｖｏ２", "ブイオーツー"),
        ("VO2", "ブイオーツー"),
        ("OM-4", "オーエムフォー"),
        ("ＮＩＥ講習会", "エヌアイイー講習会"),
        ("ＭＴ車", "エムティー車"),
        ("ＴＯＶ", "ティーオーブイ"),
        ("ＮｅｗＳ", "ニュース"),
        ("ＮＯ１", "ナンバーワン"),
        ("RADEON", "ラデオン"),
        ("page", "ページ"),
        ("GPT-4.5", "ジーピーティー4.5"),
    ],
)
def test_normalize_text_uppercase_abbreviations(
    text: str, expected: str, for_irodori: bool
) -> None:
    """
    大文字略語や固有名詞・型番・音楽クレジットなどの英字略記が、文脈に応じた適切なカタカナ読みへ正規化されることを確認する。
    """

    if for_irodori is True and text == "GPT-4.5":
        expected = "ジーピーティー四点五"
    # 「Ｖｏ．の２人」の全角ピリオドは Irodori 向けでは「。」へ変換
    if for_irodori is True and text == "Ｖｏ．の２人":
        expected = "ボーカル。の2人"
    if for_irodori is True and text == "Vo2。":
        expected = "ブイオーツー。"
    assert normalize_text(text, for_irodori=for_irodori) == expected


def test_normalize_text_mixed_scripts():
    """
    漢字・ひらがな・カタカナ・英単語・数字・記号が混在する文において、各要素が破綻することなく適切に正規化されることを確認する。
    """

    # 漢字・ひらがな・カタカナの混在
    assert (
        normalize_text("漢字とひらがなとカタカナの混在文")
        == "漢字とひらがなとカタカナの混在文"
    )
    # 英数字との混在
    assert normalize_text("123と漢字とABCの混在") == "123と漢字とエービーシーの混在"
    # 記号との混在
    assert normalize_text("漢字+カタカナ=混在!?") == "漢字プラスカタカナイコール混在!?"
    # 特殊文字との混在
    assert normalize_text("①漢字②ひらがな③カタカナ") == "1,漢字2ひらがな3,カタカナ"
    # 単位との混在
    assert (
        normalize_text("漢字100kg+カタカナ500m")
        == "漢字100キログラムプラスカタカナ500メートル"
    )


def test_normalize_text_edge_cases():
    """
    空文字列や空白のみの入力、記号の連続など、多様なエッジケースにおいて例外が発生せず意図通りに正規化されることを確認する。
    """

    # 空文字列
    assert normalize_text("") == ""
    # 記号のみ
    assert normalize_text("...") == "..."
    assert normalize_text("!!!") == "!!!"
    assert normalize_text("???") == "???"
    # 数字のみ
    assert normalize_text("12345") == "12345"

    # 結合文字の濁点・半濁点
    assert normalize_text("か゛") == "か"  # 結合文字の濁点は削除
    assert normalize_text("は゜") == "は"  # 結合文字の半濁点は削除

    # 数字と数字の間のスペースは ' に変換される（数字連結防止）
    # "5090 32" が "509032" になって「ゴジュウマンキュウセンサンジュウニ」と読まれるのを防ぐ
    assert normalize_text("RTX 5090 32GB") == "アールティーエックス5090'32ギガバイト"
    assert (
        # H100 の H は単独では変換されない
        normalize_text("H100 96GB") == "H100'96ギガバイト"
    )
    assert normalize_text("100 200 300") == "100'200'300"
    # 英字と数字の間のスペースは連結しても問題ないため変換不要
    assert normalize_text("RTX 5090") == "アールティーエックス5090"

    # 極端に長い数値
    assert normalize_text("12345678901234567890") == "12345678901234567890"
    # 極端に長い英単語
    assert (
        normalize_text("supercalifragilisticexpialidocious")
        == "スーパーカリフラジリー"  # e2k ライブラリによる自動推定結果
    )
    # 特殊な文字の組み合わせ
    assert normalize_text("㊊㊋㊌㊍㊎㊏㊐") == "月火水木金土日"  # 曜日の丸文字
    assert (
        normalize_text("㍉㌔㌢㍍㌘㌧㌃㌶㍑㍗")
        == "ミリキロセンチメートルグラムトンアールヘクタールリットルワット"
    )


def test_normalize_text_complex():
    """
    日付・時刻・数量・URL・英単語などが複合的に含まれる長文テキストにおいて、すべての正規化処理が正しく組み合わさって適用されることを確認する。
    """

    # 日付・時刻・単位を含む文
    assert (
        normalize_text("2024/01/01(月)の14時30分に1.5kgの荷物を受け取った。")
        == "2024年1月1日月曜日の十四時30分に1.5キログラムの荷物を受け取った."
    )
    assert (
        normalize_text("MacBookで1080p/60fpsの動画を2GB保存した。")
        == "マックブックで1080p/60エフピーエスの動画を2ギガバイト保存した."
    )
    assert (
        normalize_text("¥1,000の商品を2個買うと、¥2,000です（1,000×2=2,000）。")
        == "1000円の商品を2個買うと,2000円です'1000かける2イコール2000'."
    )
    assert (
        normalize_text(
            "お問い合わせは、info@example.comまたはhttps://example.com/contactまで！"
        )
        == "お問い合わせは,インフォ,アットマーク,イグザンプルドットコムまたはエイチティーティーピーエス,イグザンプルドットコム,スラッシュ,コンタクトまで!"
    )
    assert (
        normalize_text("09:30に家を出発し、2km先のスーパーで500gのお肉を買った。")
        == "九時30分に家を出発し,2キロメートル先のスーパーで500グラムのお肉を買った."
    )
    assert (
        normalize_text(
            "2024/05/01にWindowsのアップデート（2GB+500MB=2.5GB）を実施する。"
        )
        == "2024年5月1日にウィンドウズのアップデート'2ギガバイトプラス500メガバイトイコール2.5ギガバイト'を実施する."
    )
    assert (
        normalize_text(
            "CPU使用率が50%を超え、メモリ消費が2GBに達した時点で、Windows Serverは自動的に再起動します。"
        )
        == "シーピーユー使用率が50パーセントを超え,メモリ消費が2ギガバイトに達した時点で,ウィンドウズサーバーは自動的に再起動します."
    )
    assert (
        normalize_text(
            "新商品のiPhone 15 Pro Max (256GB)が¥158,000(税込)で発売！9/22(金)午前10時から予約受付開始。"
        )
        == "新商品のアイフォン15プロマックス'256ギガバイト'が158000円'税込'で発売!9月22日金曜日午前10時から予約受付開始."
    )
    assert (
        normalize_text(
            "株式会社Deeptest(担当：山田)様、10/1(月)15:00〜17:00にWeb会議(https://meet.example.com/test)を設定しました。"
        )
        == "株式会社ディープテスト'担当,山田'様,10月1日月曜日十五時から十七時にウェブ会議'エイチティーティーピーエス,ミートドット,イグザンプルドットコム,スラッシュ,テスト'を設定しました."
    )
    assert (
        normalize_text(
            "材料(4人分)：牛肉250g、玉ねぎ1個、水300mL、醤油大さじ2(30mL)、砂糖20g。"
        )
        == "材料'4人分',牛肉250グラム,玉ねぎ1個,水300ミリリットル,醤油大さじ2'30ミリリットル',砂糖20グラム."
    )
    assert (
        normalize_text(
            "2つの数 a, b があり、a:b = 2:3 で、a + b = 10 のとき、a = 4, b = 6 となります。"
        )
        == "2つの数a,bがあり,a,bイコール二タイ三で,aプラスbイコール10のとき,aイコール4,bイコール6となります."
    )
    assert (
        normalize_text(
            "JavaScriptでArray.prototype.map()を使用し、配列の要素を2倍にする処理を1/100秒で実行。"
        )
        == "ジャバスクリプトでアレイプロトタイプマップ''を使用し,配列の要素を2倍にする処理を百ぶんの一秒で実行."
    )
    assert (
        normalize_text(
            "今日01/03（月）にですね、16:9の映像を1/128の確率で表示するイベントをやっていて、85/09/30の08月01日(金)にお会いした人と久々に会うんです"
        )
        == "今日1月3日月曜日にですね,十六タイ九の映像を百二十八ぶんの一の確率で表示するイベントをやっていて,1985年9月30日の8月1日金曜日にお会いした人と久々に会うんです"
    )
    assert (
        normalize_text(
            "path-to-model-file.onnx は事前学習済みの onnx モデルファイルです。 onnx_model/phoneme_transition_model.onnxにあります。 path-to-wav-file はサンプリング周波数 16kHz  のモノラル wav ファイルです。 path-to-phoneme-file は音素を空白区切りしたテキストが格納されたファイルのパスです。 NOTE: 開始音素と終了音素は pau である必要があります。"
        )
        == "パストゥーモデルファイルオニキスは事前学習済みのオニキスモデルファイルです.オニキスモデル/フォーニムトランジションモデルオニキスにあります.パストゥーワブファイルはサンプリング周波数16キロヘルツのモノラルワブファイルです.パストゥーフォーニムファイルは音素を空白区切りしたテキストが格納されたファイルのパスです.ノート,開始音素と終了音素はパウである必要があります."
    )
    assert (
        normalize_text(
            "Apple Watch Series 10も安くなっている。Amazonでの 販売価格は、昨年発売された新モデルということもあり、過去最安値となっている。42mmのジェットブラックモデル（Wi-Fi）の場合、5％OFFの\\53,693＋537ポイントで販売している。ヨドバシ.comとビックカメラ.comでもセール対象となっており、ポイント還元分を含んだ実質価格はAmazonと同等だ。"
        )
        == "アップルウォッチシリーズテンも安くなっている.アマゾンでの販売価格は,昨年発売された新モデルということもあり,過去最安値となっている.42ミリメートルのジェットブラックモデル'ワイファイ'の場合,5パーセントオフの53693円プラス537ポイントで販売している.ヨドバシ.コムとビックカメラ.コムでもセール対象となっており,ポイント還元分を含んだ実質価格はアマゾンと同等だ."
    )
    assert (
        normalize_text(
            "音質面では、8W×2基のスピーカーを搭載。立体音響フォーマットはDTS:Xに対応し、バーチャルサウンド技術のDTS:Virtualサウンドを活用した再生も可能だという。Google TVが導入されているため、YouTube／Prime Video／Netflixといった多数のVODサービスが楽しめる他、音声操作のGoogleアシスタントbuilt-in、スマートフォンなどのデバイスからテレビに映像をキャストするChromecast built-inなども採用されている。ユニボディデザインに極上メタリックフレームを採用したプレミアムなデザインも特徴的。付属リモコンは、Bluetooth接続タイプが投入されている。ワイヤレス機能は、Bluetooth ver5.0、Wi-Fi（5GHz/2.4GHz）に対応する。"
        )
        == "音質面では,8Wかける2基のスピーカーを搭載.立体音響フォーマットはディーティーエス,Xに対応し,バーチャルサウンド技術のディーティーエス,バーチャルサウンドを活用した再生も可能だという.グーグルティービーが導入されているため,ユーチューブ/プライムビデオ/ネットフリックスといった多数のブイーオーディーサービスが楽しめる他,音声操作のグーグルアシスタントビルトイン,スマートフォンなどのデバイスからテレビに映像をキャストするクロームキャストビルトインなども採用されている.ユニボディデザインに極上メタリックフレームを採用したプレミアムなデザインも特徴的.付属リモコンは,ブルートゥース接続タイプが投入されている.ワイヤレス機能は,ブルートゥースバージョン5.0,ワイファイ'5ギガヘルツ/2.4ギガヘルツ'に対応する."
    )
    assert (
        normalize_text(
            "Scopely（スコープリー）は『モノポリーGO』や『マーベル・ストライクフォース』などを配信している、アメリカのモバイルゲーム会社。2023年にサウジアラビアのSavvy Games Groupに49億ドルで買収されている。 一方のナイアンティックは、前述のゲーム事業の売却に合わせて、新会社となる”Niantic Spatial Inc.”（ナイアンティックスペーシャル）を設立。ジオスペーシャルAI事業として、空間コンピューティング、XR、地理情報システム（GIS）、AIを統合した、新たなプラットフォーム”Niantic Spatial Platform”へ注力するという。なお、『ポケモンGO』や『モンスターハンターNow』、『ピクミンブルーム』の事業はスコープリーへ移管されるものの、『Ingress Prime』や『Peridot』などの現実世界を舞台にしたARゲームは引き続き、ナイアンティック側で運営を行う。"
        )
        == "スコープリー'スコープリー'はモノポリーゴーやマーベル,ストライクフォースなどを配信している,アメリカのモバイルゲーム会社.2023年にサウジアラビアのサヴィゲームズグループに49億ドルで買収されている.一方のナイアンティックは,前述のゲーム事業の売却に合わせて,新会社となる'ナイアンティックスペイシャルインク.''ナイアンティックスペーシャル'を設立.ジオスペーシャルエーアイ事業として,空間コンピューティング,エックスアール,地理情報システム'ジーアイエス',エーアイを統合した,新たなプラットフォーム'ナイアンティックスペイシャルプラットフォーム'へ注力するという.なお,ポケモンゴーやモンスターハンターナウ,ピクミンブルームの事業はスコープリーへ移管されるものの,イングレスプライムやペリドットなどの現実世界を舞台にしたエーアールゲームは引き続き,ナイアンティック側で運営を行う."
    )
    assert (
        normalize_text(
            "ROCK5 is a series of Rockchip RK3588(s) based SBC(Single Board Computer) by Radxa. It can run Linux, Android, BSD and other distributions. ROCK5 comes in two models, Model A and Model B. Both models offer 4GB, 8GB, 16GB and 32GB options. For detailed difference between Model A and Model B, please check Specifications. ROCK5 features a Octa core ARM processor(4x Cortex-A76 + 4x Cortex-A55), 64bit 3200Mb/s LPDDR4, up to 8K@60 HDMI, MIPI DSI, MIPI CSI, 3.5mm jack with mic, USB Port, 2.5 GbE LAN, PCIe 3.0, PCIe 2.0, 40-pin color expansion header, RTC. Also, ROCK5 supports USB PD and QC powering."
        )
        == "ロックファイブイズアシリーズオブロックチップRK3588's'ベースドエスビーシー'シングルボードコンピューター'バイラダ.イットキャンランリナックス,アンドロイド,ビーエスディーアンドアザーディストリビューションズ.ロックファイブカムズインツーモデルズ,モデルAアンドモデルB.ボスモデルズオファー4ギガバイト,8ギガバイト,16ギガバイトアンド32ギガバイトオプションズ.フォーディテールズディファレンスビトゥイーンモデルAアンドモデルB,プリーズチェックスペシフィケーションズ.ロックファイブフィーチャーズアオクタコアアームプロセッサー'4xコーテックスA76プラス4xコーテックスA55',64ビット3200メガビット毎秒エルピーディーディーアールフォー,アップトゥー8K@60エイチディーエムアイ,ミピーディーエスアイ,ミピーシーエスアイ,3.5ミリメートルジャックウィズマイク,ユーエスビーポート,2.5ジービーイーラン,ピーシーアイイー3.0,ピーシーアイイー2.0,40ピンカラーエクスパンションヘッダー,アールティーシー.オルソ,ロックファイブサポーツユーエスビーピーディーアンドキューシーパワーリング."
    )


@pytest.mark.parametrize(
    ("input_text", "expected_text"),
    [
        (
            "2025/04/01(火)18:45開始、会場はB2F、入場料は¥2,480です。",
            "2025年4月1日火曜日十八時45分開始,会場は地下2階,入場料は2480円です.",
        ),
        (
            "速報: CPU使用率99%・温度78℃・消費電力120Wでも、Server-Xは24時間連続稼働した。",
            "速報,シーピーユー使用率99パーセント,温度78度,消費電力120Wでも,サーバーXは24時間連続稼働した.",
        ),
        (
            "新製品「AeroFit Pro 2」は、重さ98g、連続再生12.5時間、充電10分で3時間使えます。",
            "新製品'アエロフィットプロツー'は,重さ98グラム,連続再生12.5時間,充電10分で3時間使えます.",
        ),
        (
            "キャンペーン期間は2026/02/01〜26/2/28、先着1,000名様にAmazonギフト券500円分を進呈。",
            "キャンペーン期間は2026年2月1日から2026年2月28日,先着1000名様にアマゾンギフト券500円分を進呈.",
        ),
        (
            "売上は前年比128%、解約率は1.2%、NPSは+42を記録しました。",
            "売上は前年比128パーセント,解約率は1.2パーセント,エヌピーエスはプラス42を記録しました.",
        ),
        (
            "東京都千代田区1-2-3 8Fで、2024年12月31日(火)23:59にカウントダウン配信を行う。",
            "東京都千代田区1の2の3'8階で,2024年12月31日火曜日二十三時59分にカウントダウン配信を行う.",
        ),
        (
            "受付番号No.A-102をお持ちの方は、14:05までに3番窓口へお越しください。",
            "受付番号ノーA102をお持ちの方は,十四時5分までに3番窓口へお越しください.",
        ),
        (
            "β版アプリはiOS 18.1 / Android 16 / Windows 11に対応予定です。",
            "β版アプリはアイオーエス18.1/アンドロイド16/ウィンドウズイレブンに対応予定です.",
        ),
        (
            "身長170cm・体重62kg・体脂肪率11%の選手が、100mを10.8秒で走った。",
            "身長170センチメートル,体重62キログラム,体脂肪率11パーセントの選手が,100メートルを10.8秒で走った.",
        ),
        (
            "このSSDは読込7,400MB/s、書込6,500MB/s、容量4TBです。",
            "このエスエスディーは読込7400メガバイト毎秒,書込6500メガバイト毎秒,容量4テラバイトです.",
        ),
        (
            "開発コードネームProject Orion-Xでは、GPU を8基でLLMを学習した。",
            "開発コードネームプロジェクトオリオンXでは,ジーピーユーを8基でエルエルエムを学習した.",
        ),
        (
            "CEO登壇は17:30、配信URLはhttps://live.example.com/2026/keynote?lang=jaです。",
            "シーイーオー登壇は十七時30分,配信ユーアールエルはエイチティーティーピーエス,ライブドット,イグザンプルドットコム,スラッシュ,2026,スラッシュ,キーノート,クエスチョン,ラングイコールジャです.",
        ),
        (
            "R6.4.1時点で累計導入社数は321社、継続率は99.1%です。",
            "令和6年4月1日時点で累計導入社数は321社,継続率は99.1パーセントです.",
        ),
        (
            "実験条件はpH7.4、温度25℃、CO2濃度5%です。",
            "実験条件はペーハー7.4,温度25度,シーオーツー濃度5パーセントです.",
        ),
        (
            "ECサイトのCVRは2.35%、CPAは¥1,280、ROASは480%でした。",
            "イーシーサイトのCVRは2.35パーセント,CPAは1280円,ROASは480パーセントでした.",
        ),
        (
            "保存形式はPNG, JPEG, WebP、推奨サイズは1920×1080です。",
            "保存形式はピーエヌジー,ジェイペグ,ウェッピー,推奨サイズは1920かける1080です.",
        ),
        (
            "発表資料はver2.0、更新日は2025-11-03、総ページ数は48pです。",
            "発表資料はバージョン2.0,更新日は2025年11月3日,総ページ数は48ページです.",
        ),
        (
            "寄付総額は$12,345、参加者は国内47都道府県・海外12か国から集まりました。",
            "寄付総額は12345ドル,参加者は国内47都道府県,海外12か国から集まりました.",
        ),
        (
            "午前0時ちょうどに公開された動画は、公開3時間で再生回数10万回を突破。",
            "午前零時ちょうどに公開された動画は,公開3時間で再生回数10万回を突破.",
        ),
        (
            "メール件名に【至急】と入れ、dev-team@example.comへ20:30までに返信してください。",
            "メール件名に'至急'と入れ,デブハイフンチーム,アットマーク,イグザンプルドットコムへ二十時30分までに返信してください.",
        ),
        (
            "冷蔵庫は幅68cm×奥行63cm×高さ185cm、年間消費電力量は302kWh。",
            "冷蔵庫は幅68センチメートルかける奥行63センチメートルかける高さ185センチメートル,年間消費電力量は302キロワットアワー.",
        ),
        (
            "Q&Aセッションでは「なぜ今、生成AIなのか？」という質問が最多でした。",
            "Q&Aセッションでは'なぜ今,生成エーアイなのか?'という質問が最多でした.",
        ),
        (
            "2025/7/5(土) 7:05発の便で出発し、現地時間13:20に到着予定。",
            "2025年7月5日土曜日七時5分発の便で出発し,現地時間十三時20分に到着予定.",
        ),
        (
            "サーバー負荷が70%を超えたら自動でscale-outし、90%以上ならアラートを送信。",
            "サーバー負荷が70パーセントを超えたら自動でスケールアウトし,90パーセント以上ならアラートを送信.",
        ),
        (
            "限定1,500セット、1人2点まで、送料は全国一律660円です。",
            "限定1500セット,1人2点まで,送料は全国一律660円です.",
        ),
        (
            "初回出荷は2025/10/01、倉庫在庫は残り24箱、再入荷は10/15(水)予定です。",
            "初回出荷は2025年10月1日,倉庫在庫は残り24箱,再入荷は10月15日水曜日予定です.",
        ),
        (
            "4K/60fps対応カメラを2台、予算¥198,000以内で比較検討中。",
            "4K/60エフピーエス対応カメラを2台,予算198000円以内で比較検討中.",
        ),
        (
            "受付は午前8時30分から、最終入場は18:15、駐車場は第2-第4区画を利用してください。",
            "受付は午前八時30分から,最終入場は十八時15分,駐車場は第2第4区画を利用してください.",
        ),
        (
            "創業以来、累計5,000万DL、月間アクティブユーザーは320万人を突破。",
            "創業以来,累計5000万DL,月間アクティブユーザーは320万人を突破.",
        ),
        (
            "3/14(金)19:00開始の配信では、全12曲をノンストップで披露します。",
            "3月14日金曜日十九時開始の配信では,全12曲をノンストップで披露します.",
        ),
        (
            "このモバイルバッテリーは10,000mAh、最大出力は65W、重さは220gです。",
            "このモバイルバッテリーは10000ミリアンペアアワー,最大出力は65W,重さは220グラムです.",
        ),
        (
            "商品コードAB-1200Xは、倉庫C-3棚の上段にあります。",
            "商品コードエービー1200Xは,倉庫C3棚の上段にあります.",
        ),
        (
            "11/29(日)限定で、Mサイズ2枚+Lサイズ1枚のセットを¥3,980で販売。",
            "11月29日日曜日限定で,Mサイズ2枚プラスLサイズ1枚のセットを3980円で販売.",
        ),
        (
            "導入企業のうち、従業員1,000人以上の大企業は全体の42%を占めます。",
            "導入企業のうち,従業員1000人以上の大企業は全体の42パーセントを占めます.",
        ),
        (
            "視聴者プレゼントはA賞1名、B賞3名、C賞10名で、応募締切は23:59です。",
            "視聴者プレゼントはA賞1名,B賞3名,C賞10名で,応募締切は二十三時59分です.",
        ),
        (
            "〒150-0001 東京都渋谷区神宮前1-2-3のポップアップ会場は、03-1234-5678でお問い合わせを受け付けています。",
            "郵便番号150-0001,東京都渋谷区神宮前1の2の3のポップアップ会場は,03-1234-5678でお問い合わせを受け付けています.",
        ),
        (
            "新店舗の住所は〒060-0001 北海道札幌市中央区北1条西2-3 5F、電話は011-200-3000です。",
            "新店舗の住所は郵便番号060-0001,北海道札幌市中央区北1条西2の3'5階,電話は011-200-3000です.",
        ),
        (
            "サポート窓口は東京都港区赤坂1-2-3 赤坂タワー1205号室にあり、代表番号は03-5555-6789です。",
            "サポート窓口は東京都港区赤坂1の2の3,赤坂タワー'1205号室にあり,代表番号は03-5555-6789です.",
        ),
        (
            "応募書類は〒530-0001 大阪府大阪市北区梅田2-4-9へ郵送し、到着確認は06-1234-5678へご連絡ください。",
            "応募書類は郵便番号530-0001,大阪府大阪市北区梅田2の4の9へ郵送し,到着確認は06-1234-5678へご連絡ください.",
        ),
        (
            "当日の受付は〒460-0008 愛知県名古屋市中区栄3-5-12 栄スクエア8Fで、緊急連絡先は052-111-2222です。",
            "当日の受付は郵便番号460-0008,愛知県名古屋市中区栄3の5の12,栄スクエア8階で,緊急連絡先は052-111-2222です.",
        ),
        (
            "ご来場前に、〒980-0014 宮城県仙台市青葉区本町2-10-30の会場案内と022-300-4000の受付番号をご確認ください。",
            "ご来場前に,郵便番号980-0014,宮城県仙台市青葉区本町2の10の30の会場案内と022-300-4000の受付番号をご確認ください.",
        ),
        (
            "返送先は〒812-0011 福岡県福岡市博多区博多駅前1-2-3 博多ビル905号、担当直通は092-555-0101です。",
            "返送先は郵便番号812-0011,福岡県福岡市博多区博多駅前1の2の3,博多ビル'九〇五号,担当直通は092-555-0101です.",
        ),
        (
            "イベント本部は〒920-0858 石川県金沢市木ノ新保町1-1 金沢ゲート4Fに設置し、当日窓口は076-222-3333で対応します。",
            "イベント本部は郵便番号920-0858,石川県金沢市木ノ新保町1の1,金沢ゲート4階に設置し,当日窓口は076-222-3333で対応します.",
        ),
        (
            "リハーサル会場は〒604-8006 京都府京都市中京区下丸屋町403 7F、出演者連絡先は075-444-5555です。",
            "リハーサル会場は郵便番号604-8006,京都府京都市中京区下丸屋町403'7階,出演者連絡先は075-444-5555です.",
        ),
        (
            "オンライン参加が難しい方は、〒730-0031 広島県広島市中区紙屋町1-2-3へお越しいただくか、082-123-4567へお電話ください。",
            "オンライン参加が難しい方は,郵便番号730-0031,広島県広島市中区紙屋町1の2の3へお越しいただくか,082-123-4567へお電話ください.",
        ),
        (
            "配送センターは〒330-0854 埼玉県さいたま市大宮区桜木町1-9-6 大宮センタービル3Fで、集荷依頼は048-600-7000まで。",
            "配送センターは郵便番号330-0854,埼玉県さいたま市大宮区桜木町1の9の6,大宮センタービル3階で,集荷依頼は048-600-7000まで.",
        ),
        (
            "会員証の再発行は〒221-0056 神奈川県横浜市神奈川区金港町1-4 横浜イーストビル1102号室で受け付け、電話は045-222-8899です。",
            "会員証の再発行は郵便番号221-0056,神奈川県横浜市神奈川区金港町1の4,横浜イーストビル'1102号室で受け付け,電話は045-222-8899です.",
        ),
        (
            "展示会場の最寄り窓口は〒650-0021 兵庫県神戸市中央区三宮町1-5-26 三宮センター6F、代表番号は078-333-4444です。",
            "展示会場の最寄り窓口は郵便番号650-0021,兵庫県神戸市中央区三宮町1の5の26,三宮センター6階,代表番号は078-333-4444です.",
        ),
        (
            "来場予約後の変更は、〒700-0901 岡山県岡山市北区本町6-36 岡山第一セントラルビル2号館5Fまたは086-234-5678で承ります。",
            "来場予約後の変更は,郵便番号700-0901,岡山県岡山市北区本町6の36,岡山第一セントラルビル2号館5階または086-234-5678で承ります.",
        ),
        (
            "応募者説明会は〒420-0852 静岡県静岡市葵区紺屋町17-1 葵タワー10Fで開催し、欠席連絡は054-205-6000へお願いします。",
            "応募者説明会は郵便番号420-0852,静岡県静岡市葵区紺屋町17の1,葵タワー10階で開催し,欠席連絡は054-205-6000へお願いします.",
        ),
        (
            "特設ストアは〒900-0015 沖縄県那覇市久茂地1-1-1 パレットくもじ2Fにオープンし、問い合わせは098-860-1234です。",
            "特設ストアは郵便番号900-0015,沖縄県那覇市久茂地1の1の1,パレットくもじ2階にオープンし,問い合わせは098-860-1234です.",
        ),
        (
            "製品交換の送り先は〒263-0023 千葉県千葉市稲毛区緑町1-16-12、受付時間は9:30-17:00、電話は043-245-6789です。",
            "製品交換の送り先は郵便番号263-0023,千葉県千葉市稲毛区緑町1の16の12,受付時間は九時30分-十七時,電話は043-245-6789です.",
        ),
        (
            "イベント当日は〒371-0024 群馬県前橋市表町2-30-8 AQERU前橋6Fに集合し、遅刻時は027-220-5500へご連絡ください。",
            "イベント当日は郵便番号371-0024,群馬県前橋市表町2の30の8,AQERU前橋6階に集合し,遅刻時は027-220-5500へご連絡ください.",
        ),
        (
            "セミナー受付は〒310-0015 茨城県水戸市宮町1-7-33 水戸オーパ9Fで、資料請求は029-303-4040でも可能です。",
            "セミナー受付は郵便番号310-0015,茨城県水戸市宮町1の7の33,水戸オーパ9階で,資料請求は029-303-4040でも可能です.",
        ),
        (
            "落とし物のお問い合わせは〒950-0087 新潟県新潟市中央区東大通1-1-1 新潟第一ビル8F、または025-250-6006まで。",
            "落とし物のお問い合わせは郵便番号950-0087,新潟県新潟市中央区東大通1の1の1,新潟第一ビル8階,または025-250-6006まで.",
        ),
        (
            "本社移転先は〒380-0823 長野県長野市南千歳1-22-6 MIDORI長野4Fで、代表電話は026-217-1188となります。",
            "本社移転先は郵便番号380-0823,長野県長野市南千歳1の22の6,ミドリ長野4階で,代表電話は026-217-1188となります.",
        ),
        (
            "試食会の会場は〒860-0807 熊本県熊本市中央区下通1-3-8 下通NSビル5F、予約確認は096-327-7001です。",
            "試食会の会場は郵便番号860-0807,熊本県熊本市中央区下通1の3の8,下通エヌエスビル5階,予約確認は096-327-7001です.",
        ),
        (
            "記者発表は〒790-0001 愛媛県松山市一番町3-2-1 ANAクラウンプラザホテル松山4Fで行い、広報直通は089-915-5505です。",
            "記者発表は郵便番号790-0001,愛媛県松山市一番町3の2の1,エーエヌエークラウンプラザホテル松山4階で行い,広報直通は089-915-5505です.",
        ),
        (
            "ユーザー会の受付住所は〒760-0023 香川県高松市寿町2-4-20 高松センタービル7F、当日連絡先は087-811-2233です。",
            "ユーザー会の受付住所は郵便番号760-0023,香川県高松市寿町2の4の20,高松センタービル7階,当日連絡先は087-811-2233です.",
        ),
        (
            "会場周辺が混雑した場合は、〒600-8216 京都府京都市下京区東塩小路町902 京都駅前ビルB1Fの臨時受付か075-708-9000をご利用ください。",
            "会場周辺が混雑した場合は,郵便番号600-8216,京都府京都市下京区東塩小路町902京都駅前ビル地下1階の臨時受付か075-708-9000をご利用ください.",
        ),
        (
            "返金手続きは〒980-6125 宮城県仙台市青葉区中央1-3-1 AER25Fの窓口、または022-715-8811の専用ダイヤルで受け付けます。",
            "返金手続きは郵便番号980-6125,宮城県仙台市青葉区中央1の3の1,エアー25階の窓口,または022-715-8811の専用ダイヤルで受け付けます.",
        ),
        (
            "メディア受付は〒100-0005 東京都千代田区丸の内1-9-1 グラントウキョウノースタワー18Fで、当日朝は03-3211-2200へご一報ください。",
            "メディア受付は郵便番号100-0005,東京都千代田区丸の内1の9の1,グラントウキョウノースタワー18階で,当日朝は03-3211-2200へご一報ください.",
        ),
        (
            "会場案内は〒151-0051 東京都渋谷区千駄ヶ谷5-24-2 4Fで配布し、問い合わせは03-3350-1234です。",
            "会場案内は郵便番号151-0051,東京都渋谷区千駄ヶ谷5の24の2'4階で配布し,問い合わせは03-3350-1234です.",
        ),
        (
            "発送元は〒540-0001 大阪府大阪市中央区城見2-1-61 10F、配送確認は06-6940-5678で承ります。",
            "発送元は郵便番号540-0001,大阪府大阪市中央区城見2の1の61'10階,配送確認は06-6940-5678で承ります.",
        ),
        (
            "臨時窓口は〒980-8484 宮城県仙台市青葉区中央1-1-1 6Fに設置し、専用番号は022-222-0101です。",
            "臨時窓口は郵便番号980-8484,宮城県仙台市青葉区中央1の1の1'6階に設置し,専用番号は022-222-0101です.",
        ),
        (
            "予約変更は〒460-8430 愛知県名古屋市中区栄3-16-1 9Fの受付か052-264-2200へご連絡ください。",
            "予約変更は郵便番号460-8430,愛知県名古屋市中区栄3の16の1'9階の受付か052-264-2200へご連絡ください.",
        ),
    ],
)
def test_normalize_text_complex_marketing_showcase(
    input_text: str,
    expected_text: str,
) -> None:
    """
    実際のユースケースに即した宣伝文や告知文において、複数の正規化条件が複合的に適用されて自然な読みへ正規化されることを確認する。
    """

    assert normalize_text(input_text) == expected_text
