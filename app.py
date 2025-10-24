from __future__ import annotations

from pathlib import Path

import gradio as gr


def _strip_front_matter(text: str) -> str:
    """Remove leading YAML block delimited by --- if present."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip()
    return text


def _read_text(path: Path, default: str = "内容暂不可用。") -> str:
    if not path.exists():
        return default
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return _strip_front_matter(text)


with gr.Blocks(title="MA-LfL 项目复现空间") as demo:
    gr.Markdown(
        "# MA-LfL 复现工作台\n"
        "在左侧标签页查看项目简介、复现报告与关键配置。如需了解详细工程约定，请参考仓库中的 `docs/` 文档。"
    )
    with gr.Tab("项目概览"):
        gr.Markdown(_read_text(Path("README.md")))
    with gr.Tab("复现报告"):
        gr.Markdown(_read_text(Path("project_report.md"), default="`project_report.md` 尚未生成。"))
    with gr.Tab("配置清单"):
        config_path = Path("config.yaml")
        config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else "# config.yaml 未找到"
        gr.Code(value=config_text, language="yaml", label="config.yaml")


if __name__ == "__main__":
    demo.launch()
