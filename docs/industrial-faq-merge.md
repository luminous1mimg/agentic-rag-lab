# IndustrialFAQ-RAG 合库说明

来源：`IndustrialFAQ-RAG`（私有，OpenFOAM FAQ demo，2025-09），合入 `agentic-rag-lab`。

## 迁了什么

| 旧库 | 新位置 | 说明 |
| --- | --- | --- |
| `src/faq_cfd_software/faq_data/faq.tsv`（50 条） | `src/agentic_rag/faq_data/faq.tsv` | 原样保留，确定性匹配的唯一数据源 |
| `src/faq_cfd_software/match.py` | `src/agentic_rag/faq_match.py` | 改写，不是原样复制（见下） |
| `faq.tsv` 内容 | `examples/kb/openfoam/faq.md` | 转为 Markdown，供通用 RAG（`--mode rag/graph`）检索 |
| — | `tests/test_faq_match.py` | 覆盖精确匹配、别名解析、无匹配、脏数据行 |

没有迁：`assets/faq_demo.png`（演示截图，非代码）、旧 README 里不存在的
`src/indrag/cli.py` 入口（已由 `agentic_rag.main` 替代）。

## 改写点

旧 `match.py` 的反查用子串判断（`matched_sent in std`），相似问可能误命中
别的标准问。新实现把每个标准问和每个相似问都精确映射到自己的条目，
再用 `difflib` 选最佳候选；解析跳过格式错误的行；结果缓存，不再每次
查询重读文件。`text_match()` 的返回形状保持兼容。

## 用法

```bash
# 确定性 FAQ 命中：离线、快、答案稳定；没命中则回落到 RAG
python -m agentic_rag.main "怎么生成网格" --mode faq --kb examples/kb/openfoam

# 通用 RAG：直接查 OpenFOAM Markdown 知识库
python -m agentic_rag.main "怎么生成网格" --mode rag --kb examples/kb/openfoam --reindex
```

## 旧库处理建议

合库验证通过后，可将 `IndustrialFAQ-RAG` 归档，并在其 README 中指向
本仓库的 `examples/kb/openfoam/faq.md` 和 `src/agentic_rag/faq_match.py`。
本 PR 不动 `main` 分支，也不删除旧库。
