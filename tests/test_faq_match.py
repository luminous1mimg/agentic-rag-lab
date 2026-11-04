from pathlib import Path

from agentic_rag.faq_match import load_faq, match_faq, text_match

PACKAGE_TSV = Path(__file__).resolve().parents[1] / "src" / "agentic_rag" / "faq_data" / "faq.tsv"


def _write_tmp_tsv(tmp_path):
    tsv = tmp_path / "faq.tsv"
    tsv.write_text(
        "如何生成基础网格\t怎么生成网格|网格生成命令\t使用 blockMesh 命令生成网格。\n"
        "如何检查网格质量\t怎么检查网格\t执行 checkMesh 命令检查网格质量。\n"
        "格式错误的行，没有制表符分隔\n",
        encoding="utf-8",
    )
    return tsv


def test_load_skips_malformed_lines(tmp_path):
    entries = load_faq(_write_tmp_tsv(tmp_path))
    assert len(entries) == 2
    assert entries[0].standard == "如何生成基础网格"
    assert entries[0].aliases == ("怎么生成网格", "网格生成命令")


def test_exact_and_alias_match(tmp_path):
    tsv = _write_tmp_tsv(tmp_path)
    assert match_faq("如何生成基础网格", tsv=tsv).answer.startswith("使用 blockMesh")
    # Alias must resolve to its own standard question, not a substring hit.
    match = match_faq("怎么生成网格", tsv=tsv)
    assert match.standard == "如何生成基础网格"
    assert match.score == 1.0


def test_no_match(tmp_path):
    assert match_faq("今天天气怎么样", tsv=_write_tmp_tsv(tmp_path)) is None
    assert text_match("今天天气怎么样", tsv=_write_tmp_tsv(tmp_path)) == (None, "抱歉，无匹配")


def test_merged_openfoam_data():
    entries = load_faq(PACKAGE_TSV)
    assert len(entries) == 50
    match = match_faq("怎么生成网格", tsv=PACKAGE_TSV)
    assert match is not None
    assert "blockMesh" in match.answer
