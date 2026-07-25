# Example: per-chapter metadata for batch slide generation.
# 8 standard pages per chapter:
# 01_cover, 02_glance, 03_what, 04_why,
# 05_mechanism, 06_frameworks, 07_rule, 08_summary

META = [
    {
        "num": "01", "slug": "ch01_intro",
        "zh": "提示链", "en": "Prompt Chaining",
        "one_liner": "拆线性流水线",
        "problem": "单个超长提示词难以维护，输出质量不稳定",
        "solution": "把任务拆成有序步骤，每步一个聚焦提示词",
        "key_points": ["步骤间通过结构化输出衔接", "每步可独立调试"],
        "frameworks": [("LangChain", "LCEL", "pipe"), ("LlamaIndex", "Workflow", "event")],
        "scenarios": ["文档摘要", "多轮改写"],
        "pitfalls": ["步骤过细导致上下文丢失"],
        "quote": "Divide and conquer.",
    },
    # ... more chapters
]
