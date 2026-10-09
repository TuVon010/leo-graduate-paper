"""Algorithm-independent bilingual diagrams for the physical system model.

All topology, time, capacity, and workload values are illustrative.
Outputs portable PNG and outlined SVG; no experimental data are consumed.
"""
import argparse
from math import hypot
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED = "#203247", "#60758A"
BLUE, GREEN, ORANGE, RED = "#2476AC", "#23856D", "#BC7D24", "#B54A57"
GRAY, LIGHT, PURPLE = "#BAC7D3", "#EEF3F7", "#7865A5"


def configure(language):
    if language == "zh":
        available = {font.name for font in font_manager.fontManager.ttflist}
        fonts = [font for font in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "SimSun")
                 if font in available]
        if not fonts:
            raise RuntimeError("Chinese figures require Microsoft YaHei or Noto Sans CJK SC.")
        plt.rcParams["font.family"] = [fonts[0], "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams.update({"svg.fonttype": "path", "axes.unicode_minus": False,
                         "font.size": 11, "text.color": INK})


def canvas(width, height, language):
    configure(language)
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def label(ax, x, y, zh, en, language, size=11, color=INK, **kwargs):
    return ax.text(x, y, zh if language == "zh" else en, fontsize=size, color=color,
                   va="center", linespacing=1.45, zorder=10, **kwargs)


def box(ax, x, y, width, height, color=GRAY, fill="white", zorder=1):
    patch = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0,rounding_size=0.09",
                           edgecolor=color, facecolor=fill, linewidth=1, zorder=zorder)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=BLUE, width=1.8, directed=True, style="-"):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>" if directed else "-",
                           mutation_scale=15, color=color, linewidth=width,
                           linestyle=style, zorder=3)
    ax.add_patch(patch)
    return patch


def satellite(ax, x, y, color=BLUE, scale=1):
    ax.add_patch(Circle((x, y), .40 * scale, facecolor="white", edgecolor="none", zorder=4))
    for sign in (-1, 1):
        ax.add_patch(Rectangle((x + sign * .30 * scale - .13 * scale, y - .12 * scale),
                               .26 * scale, .24 * scale, facecolor=color,
                               edgecolor="white", linewidth=.8, zorder=5))
    ax.add_patch(Rectangle((x - .12 * scale, y - .17 * scale), .24 * scale, .34 * scale,
                           facecolor="#F6E5AE", edgecolor=INK, linewidth=1, zorder=6))


def save(fig, output, stem, language):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output / f"{stem}_{language}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path.resolve())
    plt.close(fig)


def physical(language, output):
    fig, ax = canvas(14, 8, language)
    label(ax, .55, 7.60, "LEO 卫星边缘计算系统：物理场景与任务服务",
          "LEO satellite edge computing: physical system and task service",
          language, 19, fontweight="bold")
    label(ax, .55, 7.12, "整任务执行 · 逐跳存储转发 · 共享通信与计算资源",
          "Whole-task execution  |  Store-and-forward input  |  Shared link and CPU budgets",
          language, 11, color=MUTED)
    panel = box(ax, .4, 1.72, 13.2, 5.0, fill="#FCFDFE")
    for angle, height in ((0, 4.3), (20, 3.8), (-17, 3.7)):
        orbit = Ellipse((7, 3.9), 12.1, height, angle=angle, fill=False,
                        edgecolor="#DAE3EB", linestyle=(0, (5, 4)), linewidth=1.1)
        orbit.set_clip_path(panel)
        ax.add_patch(orbit)
    earth = Circle((7, 3.0), 1.1, facecolor="#E7F1F5", edgecolor="#AFC8D0", linewidth=1)
    ax.add_patch(earth)
    for width in (.65, 1.4):
        ax.add_patch(Ellipse((7, 3), width, 2.2, fill=False, edgecolor="#C1D7DE"))
    label(ax, 7, 3.0, "地球", "Earth", language, 13, color=MUTED, ha="center")
    positions = [(2.2, 5.25), (6.8, 5.85), (11.6, 5.15), (2.5, 2.8), (11.1, 2.6)]
    for first, second in ((0, 3), (1, 4), (2, 4)):
        arrow(ax, positions[first], positions[second], GRAY, 1.25, directed=False)
    for first, second in ((0, 1), (1, 2)):
        a, b = positions[first], positions[second]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = hypot(dx, dy)
        offset = .47
        arrow(ax, (a[0] + offset * dx / length, a[1] + offset * dy / length),
              (b[0] - offset * dx / length, b[1] - offset * dy / length), BLUE, 2.7)
    for index, point in enumerate(positions):
        satellite(ax, *point, GREEN if index == 2 else BLUE if index < 2 else GRAY)
    for index, zh, en, symbol in (
        (0, "源卫星", "Source satellite", r"$s_u$"),
        (1, "中继卫星", "Relay satellite", r"$v_1$"),
        (2, "计算卫星", "Computing satellite", r"$s_u^*$"),
    ):
        x, y = positions[index]
        label(ax, x, y + .58, zh, en, language, 12, ha="center", fontweight="bold")
        ax.text(x, y - .44, symbol, ha="center", va="center", fontsize=13, zorder=10)
    label(ax, 4.4, 5.23, "完整输入 D", "Full input D", language, 10, color=BLUE, ha="center")
    label(ax, 9.2, 5.27, "完整输入 D", "Full input D", language, 10, color=BLUE, ha="center")
    box(ax, .95, 6.10, 2.6, .58, color=BLUE, fill="#F0F6FA")
    label(ax, 2.25, 6.39, r"任务 $(s_u,D_u,C_u,\tau_u,t_u)$",
          r"Task $(s_u,D_u,C_u,\tau_u,t_u)$", language, 10, ha="center")
    arrow(ax, (3.55, 6.25), (2.58, 5.48), BLUE, 1.2)
    box(ax, .75, 3.65, 3.05, .78, color=GREEN, fill="#F1F8F5", zorder=4)
    label(ax, 2.27, 4.15, "本地执行同样可行", "Local execution is available",
          language, 10.5, color=GREEN, ha="center")
    label(ax, 2.27, 3.88, r"$s_u^*=s_u,\quad h_u=0$", r"$s_u^*=s_u,\quad h_u=0$",
          language, 11, ha="center")
    arrow(ax, (2.2, 4.77), (2.2, 4.45), GREEN, 1.2)
    box(ax, 9.6, 3.60, 3.5, .86, color=GREEN, fill="#F1F8F5", zorder=4)
    label(ax, 11.35, 4.17, "输入全部到达后计算", "Compute after complete input arrival",
          language, 10.5, color=GREEN, ha="center")
    label(ax, 11.35, 3.86, r"$\sum_{u\in\mathcal{K}_s(t)} f_{s,u}(t)\leq F_s$",
          r"$\sum_{u\in\mathcal{K}_s(t)} f_{s,u}(t)\leq F_s$", language, 11, ha="center")
    arrow(ax, (11.6, 4.76), (11.6, 4.48), GREEN, 1.2)
    label(ax, 7, 1.99, "轨道与连接为示意；地面接入和结果回传不计入本模型",
          "Schematic orbits and links; ground access and result return are outside the model",
          language, 9.7, color=MUTED, ha="center")
    for x, zh, en in ((.8, "完整输入发送", "Full-input transmission"),
                      (5.0, "传播与下一跳", "Propagation and next hop"),
                      (9.2, "共享 CPU 处理", "Shared CPU processing")):
        box(ax, x, .75, 3.45, .60, fill=LIGHT)
        label(ax, x + 1.725, 1.05, zh, en, language, 11, ha="center")
    arrow(ax, (4.25, 1.05), (4.95, 1.05), MUTED, 1.3)
    arrow(ax, (8.45, 1.05), (9.15, 1.05), MUTED, 1.3)
    label(ax, 7, .35, "服务完成点：星上计算结束；未完成工作量跨时隙保留",
          "Service ends at onboard computation; residual work persists across slots",
          language, 10.5, color=MUTED, ha="center")
    save(fig, output, "physical_system", language)


def dynamic_graph(language, output):
    fig, ax = canvas(14, 9, language)
    label(ax, .55, 8.60, "图 E-1  动态图：连接变化与资源变化",
          "Fig. E-1  Dynamic graph: connectivity and resource changes",
          language, 19, fontweight="bold")
    label(ax, .55, 8.14, "节点集合保持不变；边集合和剩余工作量随时间变化",
          "The satellite set is fixed; edges and residual workloads evolve over time",
          language, 11, color=MUTED)
    active_sets = ({(0, 1), (1, 2), (2, 3)}, {(1, 2)}, {(0, 1), (2, 3)})
    q_values = ((4, 2, 0, 3), (1, 4, 2, 0), (0, 2, 5, 1))
    i_values = ((0, 1, 2, 0), (0, 0, 2, 0), (0, 1, 0, 0))
    for frame, time in enumerate((1.2, 3.2, 5.2)):
        x = .4 + frame * 4.6
        box(ax, x, 5.0, 4.0, 2.7, fill="#FCFDFE")
        label(ax, x + 2, 7.27, f"时刻 t = {time:.1f} s", f"Instant t = {time:.1f} s",
              language, 12, ha="center", fontweight="bold")
        points = [(x + .50 + node, 6.60) for node in range(4)]
        for edge in ((0, 1), (1, 2), (2, 3)):
            a, b = points[edge[0]], points[edge[1]]
            present = edge in active_sets[frame]
            arrow(ax, (a[0] + .16, a[1]), (b[0] - .16, b[1]),
                  BLUE if present else RED, 2 if present else 1.2,
                  directed=False, style="-" if present else (0, (3, 3)))
        for node, (px, py) in enumerate(points):
            ax.add_patch(Circle((px, py), .17, facecolor="white", edgecolor=BLUE, linewidth=1.5, zorder=5))
            label(ax, px, 6.17, f"$S_{node + 1}$", f"$S_{node + 1}$",
                  language, 11, ha="center")
            ax.text(px, 5.67, f"Q={q_values[frame][node]}\nI={i_values[frame][node]}",
                    ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.65)
        label(ax, x + 2, 5.12, "工作量单位：Gcycles", "Workloads: Gcycles",
              language, 9, ha="center", color=MUTED)
    arrow(ax, (4.42, 6.47), (4.87, 6.47), MUTED, 1.2)
    arrow(ax, (9.02, 6.47), (9.47, 6.47), MUTED, 1.2)
    arrow(ax, (.65, 4.52), (1.12, 4.52), BLUE, 2, directed=False)
    label(ax, 1.27, 4.52, "活动边", "Active edge", language, 10)
    arrow(ax, (3.55, 4.52), (4.02, 4.52), RED, 1.2, directed=False, style=(0, (3, 3)))
    label(ax, 4.16, 4.52, "不可用边（不属于当前边集合）", "Unavailable edge (outside the active set)",
          language, 10)
    label(ax, .6, 3.90, "共享服务的独立示例：有竞争任务时，单个任务获得的速率下降",
          "Independent sharing example: contention reduces an individual task's service rate",
          language, 11, fontweight="bold")
    for left, start, end, unit, zh, en, color in (
        (.085, 1.0, 3.0, "Mbit/s", "链路分配速率", "Allocated link rate", BLUE),
        (.57, 1.5, 3.5, "Gcycles/s", "CPU 分配速率", "Allocated CPU rate", GREEN),
    ):
        chart = fig.add_axes([left, .145, .355, .245])
        chart.axvspan(start, end, color="#F4E8D1", alpha=.8)
        chart.axhline(20, color=GRAY, linestyle="--", linewidth=1)
        chart.step([0, start, end, 4], [20, 10, 20, 20], where="post", color=color, linewidth=2)
        chart.set(xlim=(0, 4), ylim=(0, 23), xticks=[0, 1, 2, 3, 4], yticks=[0, 10, 20])
        chart.set_ylabel(unit, color=MUTED, fontsize=10)
        chart.set_xlabel("示例时间 (s)" if language == "zh" else "Illustrative time (s)", fontsize=10)
        chart.set_title(zh if language == "zh" else en, fontsize=11, loc="left", color=INK)
        chart.text((start + end) / 2, 15, "两个任务共享" if language == "zh" else "Two active tasks",
                   fontsize=9.5, ha="center", color=ORANGE)
        for side in ("top", "right"):
            chart.spines[side].set_visible(False)
        chart.spines["left"].set_color(GRAY)
        chart.spines["bottom"].set_color(GRAY)
        chart.tick_params(labelsize=9, colors=MUTED)
    label(ax, 7, .36, "连接、工作量与服务曲线均为示意，不是实际轨道或实验记录",
          "Connectivity, workloads, and service curves are schematic; no measured data are shown",
          language, 10, color=MUTED, ha="center")
    save(fig, output, "dynamic_graph", language)


def contacts(language, output):
    fig, ax = canvas(14, 9.3, language)
    # Plot time along a dedicated affine map, leaving room for row labels.
    x0, scale = 2.10, 1.28
    xt = lambda t: x0 + scale * t
    label(ax, .55, 8.90, "图 E-2  快照可用性、接触窗口与逐跳服务",
          "Fig. E-2  Snapshot availability, contact windows, and sequential service",
          language, 18.5, fontweight="bold")
    label(ax, .55, 8.43, "同一输入依次经历发送、传播和下一跳；窗口容量由速率积分得到",
          "Input follows sending, propagation, and the next hop; window capacity is a rate integral",
          language, 10.5, color=MUTED)
    label(ax, .65, 7.92, "(a) 逐时隙连接状态", "(a) Slot-by-slot link availability",
          language, 12, fontweight="bold")
    label(ax, 11.2, 7.92, r"示例 $\delta t=1$ s", r"Example: $\delta t=1$ s",
          language, 10, color=MUTED)
    for slot in range(8):
        label(ax, xt(slot + .5), 7.53, f"$n={slot}$", f"$n={slot}$",
              language, 10, ha="center", color=MUTED)
    rows = ((r"$e_{12}$", [1, 1, 1, 0, 0, 1, 1, 1]),
            (r"$e_{23}$", [0, 1, 1, 1, 1, 0, 0, 0]),
            (r"$e_{34}$", [1, 1, 0, 0, 1, 1, 1, 1]))
    for row, (name, states) in enumerate(rows):
        y = 7.02 - .49 * row
        label(ax, 1.7, y, name, name, language, 12, ha="right")
        for slot, present in enumerate(states):
            ax.add_patch(Rectangle((xt(slot) + .025, y - .15), scale - .05, .30,
                                   facecolor=BLUE if present else LIGHT,
                                   edgecolor=GRAY, linewidth=.6))
    label(ax, .65, 5.53, "(b) 连续接触窗口", "(b) Continuous contact windows",
          language, 12, fontweight="bold")
    windows = (((0, 3), (5, 8)), ((1, 5),), ((0, 2), (4, 8)))
    for row, intervals in enumerate(windows):
        y = 4.97 - .54 * row
        label(ax, 1.7, y, rows[row][0], rows[row][0], language, 12, ha="right")
        ax.plot([xt(0), xt(8)], [y, y], color=GRAY, linewidth=1)
        for start, end in intervals:
            ax.add_patch(Rectangle((xt(start), y - .16), scale * (end - start), .32,
                                   facecolor="#CFE5DE", edgecolor=GREEN, linewidth=.8))
            label(ax, (xt(start) + xt(end)) / 2, y,
                  f"[{start}, {end})", f"[{start}, {end})", language, 10, ha="center", color=GREEN)
    # Unknown beyond the observation endpoint is neither guaranteed contact nor closure.
    ax.add_patch(Rectangle((xt(8), 3.54), .45, 4.19, facecolor="#F1F1F1",
                           edgecolor=GRAY, hatch="///", linewidth=.5))
    ax.plot([xt(8), xt(8)], [3.50, 7.76], color=MUTED, linewidth=1.2, linestyle="--")
    label(ax, xt(8) + .21, 5.54, "信息截止\n之后未知", "Unknown\nbeyond view",
          language, 9, color=MUTED, rotation=90, ha="center")
    label(ax, 3.75, 3.45, r"示例：$R_e=20$ Mbit/s，$K_{12,[0,3)}=60$ Mbit",
          r"Example: $R_e=20$ Mbit/s, $K_{12,[0,3)}=60$ Mbit",
          language, 10.5, color=GREEN)
    label(ax, .65, 2.86, "(c) 10 Mbit 输入的服务过程", "(c) Service of a 10 Mbit input",
          language, 12, fontweight="bold")

    def service(start, end, y, color, caption=None, dashed=False):
        ax.add_patch(Rectangle((xt(start), y - .14), scale * (end - start), .28,
                               facecolor="white" if dashed else color, edgecolor=color,
                               linewidth=1.3, linestyle="--" if dashed else "-"))
        if caption:
            label(ax, xt((start + end) / 2), y + .35, caption, caption,
                  language, 10, color=color, ha="center")

    label(ax, 1.7, 2.05, "任务 u", "Task u", language, 11, ha="right")
    service(.90, 1.40, 2.05, BLUE, r"$e_{12}$")
    service(1.40, 1.42, 2.05, PURPLE)
    service(1.42, 1.92, 2.05, GREEN, r"$e_{23}$")
    service(1.92, 1.94, 2.05, PURPLE)
    ax.annotate("传播 0.02 s" if language == "zh" else "Propagation: 0.02 s",
                xy=(xt(1.41), 2.05), xytext=(xt(3.4), 2.24),
                color=PURPLE, fontsize=9.5,
                arrowprops={"arrowstyle": "->", "color": PURPLE, "linewidth": .9})
    label(ax, 1.7, 1.33, r"CPU ($S_3$)", r"CPU ($S_3$)", language, 11, ha="right")
    service(1.94, 2.94, 1.33, ORANGE)
    label(ax, xt(4.0), 1.33, "完整输入到达后才计算", "CPU starts after complete input arrival",
          language, 10, color=ORANGE)
    label(ax, 1.7, .61, "任务 v", "Task v", language, 11, ha="right")
    service(2.7, 3.0, .61, RED)
    service(3.0, 3.2, .61, RED, dashed=True)
    ax.plot([xt(3), xt(3)], [.32, 2.48], color=RED, linewidth=1, linestyle="--")
    label(ax, xt(4.0), .61, "接触 t=3 s 结束，输入未发完", "Contact closes at t=3 s; input is incomplete",
          language, 10, color=RED)
    label(ax, 7, .15, "时间、连接与资源数值均为示意；虚线框表示未能获得的服务",
          "Times and resources are illustrative; the dashed box marks unavailable service",
          language, 9.5, color=MUTED, ha="center")
    save(fig, output, "contact_windows", language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_figures")
    args = parser.parse_args()
    for language in ("zh", "en") if args.language == "both" else (args.language,):
        physical(language, args.output)
        dynamic_graph(language, args.output)
        contacts(language, args.output)
