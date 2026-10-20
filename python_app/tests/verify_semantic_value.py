"""生成明确标注为模拟数据的双HTML验收产物；绝不调用真实模型。"""
from pathlib import Path
import json
import time
import tracemalloc

from test_semantic_value import SemanticValueTests


def main() -> None:
    output = Path(__file__).resolve().parents[2] / "output/semantic_value_offline_validation"
    output.mkdir(parents=True, exist_ok=True)
    case = SemanticValueTests("test_dual_export_categories_and_database_read_only")
    tracemalloc.start()
    started = time.perf_counter()
    try:
        case.setUp()
        case.test_dual_export_categories_and_database_read_only()
        for name in ("index.html", "index_report.html"):
            html = (case.h.out / name).read_text(encoding="utf-8")
            html = html.replace("<body>", '<body><aside style="padding:12px;background:#fff1cc;text-align:center">离线模拟验收数据，不是用户真实重跑结果。</aside>', 1)
            (output / name).write_text(html, encoding="utf-8", newline="\n")
        summary = dict(mode="offline_mock_not_model_quality", status="passed", sessions=4,
                       examples=["喜欢恐龙", "我要牛奶", "去了水上乐园", "In binder."],
                       real_model_calls=0, mock_calls=len(case.h.fake.calls),
                       elapsed_seconds=round(time.perf_counter() - started, 3),
                       python_peak_bytes=tracemalloc.get_traced_memory()[1],
                       parameters_flops="未运行模型，不适用", model_tokens="未调用真实服务，不估算")
        (output / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        case.doCleanups()
        tracemalloc.stop()


if __name__ == "__main__":
    main()
