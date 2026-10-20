"""仅运行Mock链路并导出新结构双HTML，不调用模型或覆盖真实结果。"""
from pathlib import Path
import json
import time
import tracemalloc
from test_memory_management import ManagementTests
from export_memory_html import export_user


def main() -> None:
    output = Path(__file__).resolve().parents[2] / "output/memory_management_offline_validation"
    output.mkdir(parents=True, exist_ok=True)
    case = ManagementTests()
    started = time.perf_counter()
    tracemalloc.start()
    try:
        case.setUp()
        case.test_learning_structure_atomic_store_and_both_html()
        previous = json.loads((case.h.out / "report.json").read_text(encoding="utf-8"))["results"]
        case.write_report(previous + case.topic_sessions())
        export_user(case.h.out, case.h.out / "index.html")
        for name in ("index.html", "index_report.html"):
            html = (case.h.out / name).read_text(encoding="utf-8")
            html = html.replace("<body>", '<body><aside style="padding:12px;background:#fff1cc;text-align:center">离线模拟验收，不是8EAF真实重跑结果。</aside>', 1)
            (output / name).write_text(html, encoding="utf-8", newline="\n")
        summary = dict(status="passed", mode="offline_mock", real_model_calls=0, sessions=3,
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
