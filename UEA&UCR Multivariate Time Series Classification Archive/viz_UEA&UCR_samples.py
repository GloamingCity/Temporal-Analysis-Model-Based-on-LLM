# Usage
# cd "D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\UEA&UCR Multivariate Time Series Classification Archive"
# python "viz_UEA&UCR_samples.py" --jsonl_path BinaryHeartbeat_BinaryHeartbeat_TRAIN_descriptions.jsonl --sample_index 0
# 参数说明：
#   --jsonl_path   : 生成的 JSONL 样本文件路径（必须）
#   --sample_index : 样本行号（从0开始），与 --sample_id 二选一，如果两者都给则优先 sample_id。
#   --sample_id    : 样本 id，例如 BinaryHeartbeat_BinaryHeartbeat_TRAIN_idx0_L512_start0
#   --no_events    : 不在图中高亮 z-score 检测到的异常事件区域
#   --dpi          : 输出图片的分辨率（dpi），默认300
# 命令行可通过 "--feature" 指定特征名称，但 UEA/UCR 样本本质上是单变量序列，通常不需要指定。

import json
from pathlib import Path
import argparse
import re
import pandas as pd
import matplotlib
# 使用无显示后端以便在没有显示服务器的环境中也能保存 PNG
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path as _Path

# 字体配置与其它可视化脚本保持一致
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
]
FONT_PATH = None
for _p in FONT_CANDIDATES:
    if _Path(_p).exists():
        FONT_PATH = _p
        break
cn_font = None
if FONT_PATH:
    try:
        cn_font = font_manager.FontProperties(fname=FONT_PATH)
        matplotlib.rcParams["font.sans-serif"] = [cn_font.get_name()]
    except Exception:
        cn_font = None
matplotlib.rcParams["axes.unicode_minus"] = False


def _token_visual_width(token: str) -> float:
    """粗略估计 token 的显示宽度：中文字符按 1.0，ASCII 按 0.56。"""
    w = 0.0
    for ch in token:
        if "\u4e00" <= ch <= "\u9fff":
            w += 1.0
        elif "\uff00" <= ch <= "\uffef":
            w += 1.0
        else:
            w += 0.56
    return w


def _wrap_single_line_mixed_text(line: str, max_width_units: float) -> list[str]:
    """
    对单行文本做中英混排折行：
    - 中文可按字切分
    - 英文/数字/路径等连续片段尽量不拆
    - 尽量避免标点落在新行行首
    """
    if not line:
        return [""]

    token_pattern = re.compile(r"[A-Za-z0-9]+(?:[._:/\\-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|\s+|.")
    tokens = token_pattern.findall(line)

    lines: list[str] = []
    cur_tokens: list[str] = []
    cur_w = 0.0

    def flush_current():
        nonlocal cur_tokens, cur_w
        lines.append("".join(cur_tokens).rstrip())
        cur_tokens = []
        cur_w = 0.0

    for tok in tokens:
        if not cur_tokens and tok.isspace():
            continue

        tok_w = _token_visual_width(tok)
        if cur_tokens and (cur_w + tok_w > max_width_units):
            if re.fullmatch(r"[，。；：、！？）】》,.;:!?)]", tok):
                cur_tokens.append(tok)
                cur_w += tok_w
                continue
            flush_current()
            if tok.isspace():
                continue

        cur_tokens.append(tok)
        cur_w += tok_w

    if cur_tokens or not lines:
        flush_current()

    return lines


def format_description_for_plot(description: str, wrap_width: int) -> str:
    """
    保留描述文本中的显式换行，再对每一行做智能折行。
    """
    logical_lines = description.split("\n")
    wrapped_lines: list[str] = []
    for ln in logical_lines:
        if ln.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_single_line_mixed_text(ln, max_width_units=float(wrap_width)))
    return "\n".join(wrapped_lines)


def load_sample_from_jsonl(jsonl_path: str, sample_index: int | None = None, sample_id: str | None = None):
    """
    从 JSONL 文件里读取一个样本，返回 (样本对象, 行索引)。
    如果给出 sample_id，则按 id 精确匹配并返回其所在行编号。
    否则使用 sample_index（默认0）。
    """
    jsonl_path = Path(jsonl_path)
    assert jsonl_path.exists(), f"{jsonl_path} 不存在"
    if sample_id is not None:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                if obj.get("id") == sample_id:
                    return obj, idx
        raise ValueError(f"在 {jsonl_path} 中未找到 id={sample_id} 的样本")
    if sample_index is None:
        sample_index = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == sample_index:
                return json.loads(line), i
    raise IndexError(f"sample_index={sample_index} 超出范围")


def sample_to_dataframe(sample: dict) -> pd.DataFrame:
    """
    将样本的 values 列表转换为 DataFrame，索引为整数序列。
    列名使用 sample 中的 feature 姓或默认 "value"。
    """
    vals = sample.get("values", [])
    colname = sample.get("feature_names") or sample.get("target_col") or "value"
    df = pd.DataFrame(vals, columns=[colname])
    return df


def plot_sample_with_description(
    sample: dict,
    feature: str | None = None,
    wrap_width: int = 32,
    show_events: bool = True,
    output_path: str | None = None,
    dpi: int = 300,
):
    """
    画出指定 feature 的时间序列以及对应的中文描述。
    """
    df = sample_to_dataframe(sample)
    if feature is None:
        feature = df.columns[0] if len(df.columns) > 0 else None
    if feature not in df.columns:
        raise ValueError(f"feature '{feature}' 不在数据列中：{df.columns.tolist()}")

    desc_list = sample.get("descriptions", [])
    description = desc_list[0] if desc_list else "(样本中没有 descriptions 字段)"
    features_struct = sample.get("features", {})

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.6])

    # 左侧时间序列
    ax_ts = fig.add_subplot(gs[:, :2])
    ax_ts.plot(df.index, df[feature], linewidth=1.2)
    ax_ts.set_title(f"{sample.get('id','')}  |  {feature}", fontsize=12, fontweight='bold')
    ax_ts.set_xlabel("时间点")
    ax_ts.set_ylabel(feature)
    ax_ts.grid(True, alpha=0.3)

    if show_events and "events" in features_struct:
        events = features_struct.get("events", [])
        if events:
            times = df.index.to_list()
            for ev in events[:5]:
                s = ev.get("start", 0)
                e = ev.get("end", 0)
                s = max(0, min(s, len(times) - 1))
                e = max(0, min(e, len(times) - 1))
                ax_ts.axvspan(times[s], times[e], alpha=0.15, color='red')

    # 右侧文本描述
    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")

    wrapped_text = format_description_for_plot(description, wrap_width=wrap_width)

    text_kwargs = dict(ha="left", va="top", fontsize=10, wrap=True)
    if cn_font:
        text_kwargs["fontproperties"] = cn_font
    ax_txt.text(0.0, 1.0, wrapped_text, **text_kwargs)
    ax_txt.set_title("自动生成的中文描述", loc="left", fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = Path(output_path) if output_path is not None else Path("output_viz_UEA&UCR.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 UEA&UCR 描述样本：画出时间序列 + 中文描述，方便肉眼检查描述是否准确。"
    )
    parser.add_argument("--jsonl_path", type=str, required=True, help="生成的 JSONL 样本文件路径")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="要可视化的样本行号（从 0 开始）。与 --sample_id 二选一，如果两者都给则优先 sample_id。",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="要可视化的样本 id，例如 BinaryHeartbeat_BinaryHeartbeat_TRAIN_idx0_L512_start0",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="要绘制的特征名，默认使用样本中的第一列",
    )
    parser.add_argument(
        "--no_events",
        action="store_true",
        help="不在图中高亮 z-score 检测到的异常事件区域",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图片的分辨率（dpi），默认300，值越高越清晰但文件越大",
    )

    args = parser.parse_args()

    sample, sample_idx = load_sample_from_jsonl(
        jsonl_path=args.jsonl_path,
        sample_index=None if args.sample_id is not None else args.sample_index,
        sample_id=args.sample_id,
    )

    # 构造输出文件名
    dataset = sample.get("dataset", "unknown_dataset")
    inst = sample.get("instance_index", "?")
    label = sample.get("class_label", "")
    window_length = sample.get("window_length", "L?")
    start_idx = sample.get("start_index", sample.get("start", "0"))

    invalid_chars = set('\\/:*?"<>|')
    def sanitize(s: str) -> str:
        return ''.join('_' if c in invalid_chars else c for c in str(s)).strip()

    safe_dataset = sanitize(dataset)
    safe_label = sanitize(label)
    safe_inst = sanitize(inst)
    safe_window = sanitize(window_length)
    safe_start = sanitize(start_idx)

    output_filename = f"{sample_idx:04d}_{safe_dataset}_idx{safe_inst}_cls{safe_label}_L{safe_window}_start{safe_start}.png"
    output_path = Path(args.jsonl_path).parent / output_filename

    plot_sample_with_description(
        sample,
        feature=args.feature,
        show_events=not args.no_events,
        output_path=str(output_path),
        dpi=args.dpi,
    )

    print(f"✅ 已生成可视化图片：{output_path}")


if __name__ == "__main__":
    main()
