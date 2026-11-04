"""Offline evaluation for the merged OpenFOAM FAQ matcher.

This is a deterministic, no-LLM probe, not a real-user query log:
- 50 standard questions + 100 aliases from the shipped TSV (exact hits).
- 150 perturbations: strip the leading "如何", prefix "请问", or add "？".
- 20 unrelated negative queries that must not match anything.

Full threshold sweep (0.5/0.6/0.7/0.8/0.9) lives in
``reports/faq_eval.md``; this test pins the default-threshold behaviour
that the sweep justified: 0.7 keeps perturbations at 100% while removing
the one false positive seen at 0.6 ("怎么学吉他" -> "如何设置声学模拟").
"""

from pathlib import Path

from agentic_rag.faq_match import load_faq, match_faq

PACKAGE_TSV = Path(__file__).resolve().parents[1] / "src" / "agentic_rag" / "faq_data" / "faq.tsv"

NEGATIVE_QUERIES = [
    "今天天气怎么样",
    "怎么做红烧肉",
    "Python怎么安装",
    "如何学好英语",
    "北京到上海的高铁几点",
    "手机屏幕碎了怎么办",
    "怎么减肥最快",
    "股票今天涨了吗",
    "怎么给猫洗澡",
    "word怎么转pdf",
    "如何办理护照",
    "感冒了吃什么药",
    "怎么提高跑步速度",
    "电影票怎么买",
    "如何写简历",
    "怎么查询快递",
    "空调不制冷怎么办",
    "怎么学吉他",
    "如何戒烟",
    "孩子发烧怎么办",
]


def _perturbations(entries):
    queries = []
    for entry in entries:
        standard = entry.standard
        queries.append((standard.replace("如何", "", 1), standard))
        queries.append(("请问" + standard, standard))
        queries.append((standard + "？", standard))
    return queries


def test_exact_standard_and_alias_are_all_correct():
    entries = load_faq(PACKAGE_TSV)
    assert len(entries) == 50
    queries = [(e.standard, e.standard) for e in entries]
    queries += [(alias, e.standard) for e in entries for alias in e.aliases]
    assert len(queries) == 150
    for query, expected in queries:
        match = match_faq(query, tsv=PACKAGE_TSV)
        assert match is not None, query
        assert match.standard == expected, query


def test_perturbations_are_all_correct_at_default_threshold():
    entries = load_faq(PACKAGE_TSV)
    queries = _perturbations(entries)
    assert len(queries) == 150
    for query, expected in queries:
        match = match_faq(query, tsv=PACKAGE_TSV)
        assert match is not None, query
        assert match.standard == expected, query


def test_negative_queries_do_not_match_at_default_threshold():
    for query in NEGATIVE_QUERIES:
        assert match_faq(query, tsv=PACKAGE_TSV) is None, query
