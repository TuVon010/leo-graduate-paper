"""Original bilingual publication-style system diagram, aligned with schema 2.

Illustrative topology and contact windows; this is not an orbital data plot.
Only matplotlib/numpy are required. Prefer an installed CJK font for Chinese.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = "#233346"
MUTED = "#61758A"
BLUE = "#2676AE"
GREEN = "#24826F"
ORANGE = "#BD7C27"
RED = "#B94D55"
PURPLE = "#7662A4"
GRAY = "#ABB9C6"


def draw(language, output):
    zh = language == "zh"
    if zh:
        available = {f.name for f in font_manager.fontManager.ttflist}
        preferred = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "SimSun"]
        fonts = [font for font in preferred if font in available]
        if not fonts:
            raise RuntimeError("Install a CJK font (e.g. Noto Sans CJK SC) to render Chinese labels")
        plt.rcParams["font.family"] = [fonts[0], "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16.5, 10.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 16.5), ylim=(0, 10.8), aspect="equal")
    ax.axis("off")

    def t(x, y, chinese, english=None, size=12, color=INK, weight="normal", ha="left", va="center", **kwargs):
        return ax.text(x, y, chinese if zh else (english if english is not None else chinese),
                       fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.45, zorder=10, **kwargs)

    def box(x, y, w, h, fill="white", edge=GRAY, lw=1.0, radius=0.12, z=2):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%s" % radius,
                               facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
        ax.add_patch(patch)
        return patch

    def arrow(start, end, color=INK, lw=1.4, style="-", curve=0, scale=14, z=3, ends="-|>"):
        patch = FancyArrowPatch(start, end, arrowstyle=ends, mutation_scale=scale, linewidth=lw,
                                linestyle=style, color=color, connectionstyle="arc3,rad=%s" % curve,
                                shrinkA=0, shrinkB=0, zorder=z)
        ax.add_patch(patch)
        return patch

    def link(a, b, color=GRAY, lw=1.1, style="-", directed=False):
        a, b = np.array(a), np.array(b)
        unit = (b - a) / np.linalg.norm(b - a)
        return arrow(a + unit * 0.48, b - unit * 0.48, color, lw, style,
                     scale=15, z=3, ends="-|>" if directed else "-")

    def sat(x, y, color=BLUE, scale=1.0):
        s = scale
        ax.add_patch(Circle((x, y), .47*s, color="white", alpha=.94, zorder=4))
        for sign in (-1, 1):
            ax.add_patch(Rectangle((x + sign*.39*s - .14*s, y - .135*s), .28*s, .27*s,
                                   facecolor=color, edgecolor="white", linewidth=.7, zorder=5))
            for part in (1, 2):
                ax.plot([x + sign*.39*s - .14*s + part*.28*s/3]*2,
                        [y-.13*s, y+.13*s], color="white", lw=.5, zorder=6)
        ax.add_patch(Rectangle((x - .145*s, y-.19*s), .29*s, .38*s,
                               facecolor="#F7E7B5", edgecolor=INK, lw=.9, zorder=6))
        ax.add_patch(Circle((x, y), .065*s, facecolor=color, edgecolor=INK, lw=.6, zorder=7))
        ax.plot([x, x+.16*s], [y+.19*s, y+.33*s], color=INK, lw=.8, zorder=6)

    def work_state(x, y, count, color, width=1.08):
        box(x, y, width, .30, fill="white", edge=color, radius=.035, z=7)
        for i in range(4):
            ax.add_patch(Rectangle((x+.06+i*(width-.10)/4, y+.055), (width-.17)/4, .19,
                                   facecolor=color if i<count else "#E9EFF4", edgecolor="none", zorder=8))

    t(.55, 10.24, "接触窗口感知的 LEO 星上计算与路由系统", "Contact-aware LEO computing and routing system", size=21, weight="bold")
    t(.57, 9.80, "任务已在源卫星 · 联合选择计算目的节点与多跳路径 · 动态共享通信和计算资源",
      "Tasks at source satellites  |  Joint compute-node/path selection  |  Shared communication and computation", size=11.5, color=MUTED)

    # (a) physical scene: orbital lines provide context, selected data edges are explicit.
    physical_panel = box(.5, 3.95, 8.0, 5.44, fill="#FCFDFE", edge="#D7E0E8", radius=.16, z=0)
    t(.78, 9.02, "(a) 星座、任务与执行选择", "(a) Constellation, tasks and execution choices", size=14, weight="bold")
    for angle, width, height in ((-20, 7.3, 4.35), (25, 7.6, 3.7), (67, 5.6, 3.4)):
        orbit = Ellipse((4.25, 6.30), width, height, angle=angle, fill=False,
                        edgecolor="#DDE5ED", lw=1.0, linestyle=(0, (4, 4)), zorder=1)
        orbit.set_clip_path(physical_panel)
        ax.add_patch(orbit)
    earth = Circle((4.15, 5.52), .94, facecolor="#E8F2F5", edgecolor="#AEC6CE", lw=1.0, zorder=2)
    ax.add_patch(earth)
    for latitude in (-.50, 0, .50):
        ellipse = Ellipse((4.15, 5.52+latitude), 2*np.sqrt(.94**2-latitude**2), .20,
                          fill=False, edgecolor="#C1D8DF", lw=.7, zorder=2)
        ellipse.set_clip_path(earth)
        ax.add_patch(ellipse)
    for w in (.55, 1.35):
        ellipse = Ellipse((4.15, 5.52), w, 1.88, fill=False, edgecolor="#C1D8DF", lw=.7, zorder=2)
        ax.add_patch(ellipse)
    t(4.15, 5.52, "地球", "Earth", size=12, color="#6D909C", ha="center")
    t(4.15, 4.05, "Walker 轨道示意，位置与连线非实际快照", "Walker schematic; not an orbital snapshot", size=9.5, color=MUTED, ha="center")

    source, relay, compute = (1.55, 7.10), (4.10, 8.03), (6.85, 7.65)
    busy, n1, n2 = (6.95, 5.87), (1.50, 5.02), (7.85, 6.85)
    for a,b in ((source,n1),(n1,relay),(compute,busy),(busy,n2)):
        link(a,b)
    link(source, busy, GRAY, 1.3, (0, (4, 3)))
    link(relay, busy, RED, 1.6, (0, (4, 3)))
    link(source, relay, BLUE, 2.8, directed=True)
    link(relay, compute, BLUE, 2.8, directed=True)
    for location, color in ((source,BLUE),(relay,BLUE),(compute,GREEN),(busy,ORANGE),(n1,GRAY),(n2,GRAY)):
        sat(*location, color)

    t(1.10, 7.62, "源卫星", "Source satellite", size=12, color=BLUE, weight="bold")
    t(1.55, 6.63, r"$s_u$", size=13, ha="center")
    t(4.10, 8.53, "转发卫星", "Relay satellite", size=11.5, ha="center")
    t(4.10, 7.60, r"$v_1$", size=12, ha="center")
    t(6.85, 8.18, "选定的计算卫星", "Selected compute satellite", size=11.5, color=GREEN, weight="bold", ha="center")
    t(6.85, 7.20, r"$s_u^\star$", size=13, ha="center")
    work_state(6.3, 6.68, 1, GREEN)
    t(6.86, 6.45, "CPU / 剩余工作量", "CPU / residual work", size=9.5, color=GREEN, ha="center")
    work_state(6.42, 5.20, 4, ORANGE)
    t(6.97, 4.99, "其他任务共同竞争", "Competing tasks", size=9.5, color=ORANGE, ha="center")
    t(5.30, 7.48, "完整输入逐跳转发", "Full-input transfer", size=10, color=BLUE, ha="center",
      bbox={"facecolor":"white", "edgecolor":"none", "pad":2})
    t(5.38, 7.04, "接触可能提前结束", "Contact may close", size=9.5, color=RED, ha="center", rotation=-38,
      bbox={"facecolor":"white", "edgecolor":"none", "pad":1})

    box(.78, 8.00, 2.03, .66, fill="#EEF5FA", edge="#C4DCEB", z=5)
    t(1.79, 8.42, "新到达任务", "New task", size=10.5, color=BLUE, weight="bold", ha="center")
    t(1.79, 8.16, r"$u=(s_u,D_u,C_u,\tau_u)$", size=10.5, ha="center")
    arrow((1.40,8.0),(1.50,7.56),BLUE,1.0)
    box(.80, 5.85, 2.00, .61, fill="#F0F7F3", edge="#B9D8CE", z=5)
    t(1.80, 6.24, "可选择在源卫星本地计算", "Local execution is an option", size=10, color=GREEN, ha="center")
    t(1.80, 5.99, r"$s_u^\star=s_u,\ \pi_u=(s_u)$", size=10.5, ha="center")
    arrow((1.32,6.71),(1.32,6.46),GREEN,1.0)
    arrow((3.66,8.68),(4.72,8.60),MUTED,1.1,curve=-.2)
    t(4.26, 8.87, "轨道运动", "Orbital motion", size=9.5, color=MUTED, ha="center")

    # Legend is inside the physical panel, above the contact inset.
    for x,color,style,ch,en in ((.93,BLUE,"-","选定任务路径","Selected route"),
                               (3.50,GRAY,"-","其他当前 ISL","Other active ISLs"),
                               (6.20,RED,(0,(4,3)),"预测接触风险","Contact risk")):
        ax.plot([x,x+.5],[4.50,4.50],color=color,lw=2,ls=style,zorder=6)
        t(x+.62,4.50,ch,en,size=9.5,color=MUTED)

    # (b) Predictive window: a failed service interval is visibly rejected.
    box(.5, 1.72, 8.0, 1.96, fill="#FAFCFE", edge="#D7E0E8", radius=.16, z=0)
    t(.78, 3.37, "(b) 接触窗口：当前连通不等于能传完", "(b) Contact window: available now does not imply completion", size=12.5, weight="bold")
    x0, x1 = 2.52, 7.95
    ax.plot([x0,x1],[2.10,2.10],color=GRAY,lw=.9)
    arrow((x1,2.1),(x1+.12,2.1),MUTED,.9)
    t(x0,1.90,r"$t_n$",size=10.5,ha="center")
    t(x1,1.90,r"$t_n+H\delta t$",size=10.5,ha="center")
    t(.83,2.91,"可用链路","Available link",size=10.5)
    t(.83,2.46,"风险候选","Risky candidate",size=10.5)
    ax.add_patch(Rectangle((x0,2.76),4.70,.26,facecolor="#CCE4DB",edgecolor="none",zorder=2))
    ax.add_patch(Rectangle((x0+.30,2.81),1.52,.16,facecolor=BLUE,edgecolor="none",zorder=3))
    t(x0+2.07,2.90,"发送在窗口内完成","Transmission fits",size=9.5,color=GREEN)
    ax.add_patch(Rectangle((x0,2.31),2.24,.26,facecolor="#CCE4DB",edgecolor="none",zorder=2))
    ax.add_patch(Rectangle((x0+1.4,2.36),2.01,.16,facecolor=RED,alpha=.84,edgecolor="none",zorder=3))
    ax.plot([x0+2.24]*2,[2.24,2.65],color=RED,lw=1.3,ls=(0,(3,2)),zorder=4)
    t(x0+3.62,2.46,"超过窗口，预测筛除","Overruns window: reject",size=9.5,color=RED)
    t(.83,1.91,"窗口 / 预计发送","Window / service",size=8.5,color=MUTED)

    # (c) logical controller. No physical ground/GEO controller is assumed.
    box(8.92, 1.72, 7.05, 7.67, fill="#FCFBFE", edge="#DBD5E8", radius=.16, z=0)
    t(9.21,9.02,"(c) 逻辑控制器：状态 → 决策 → 执行", "(c) Logical controller: observe, decide, execute",size=14,weight="bold")
    t(9.24,8.59,"当前任务、拓扑、CPU 与在途工作量 + 短期轨道快照",
      "Current tasks, graph and workloads + short orbit lookahead",size=10.5,color=MUTED)
    arrow((8.28,8.18),(9.38,8.18),PURPLE,1.4,(0,(4,3)))
    t(8.61,8.41,"状态摘要","State",size=9.5,color=PURPLE,ha="center")
    stages = [
        (7.58,"1  接触计划与窗口建模","1  Contact-plan / window modeling",
         "未来 ISL 可用区间、参考容量","ISL availability intervals and reference capacity",BLUE),
        (6.40,"2  接触窗口感知候选生成","2  Contact-aware candidate generation",
         "计算目的节点 + 当前图中的无环路径","Compute node + loop-free path in the current graph",BLUE),
        (5.22,"3  预测可行性 Shield","3  Predictive feasibility shield",
         "接触 / 期限 / 覆盖 + 链路与 CPU 预约日历","Contact / deadline / coverage + service calendars",PURPLE),
        (4.04,"4  任务感知 GAT-PPO","4  Task-aware GAT-PPO",
         "变长候选评分，选择 $(s_u^\star,\pi_u)$","Variable-candidate scoring: select $(s_u^\star,\pi_u)$",PURPLE),
        (2.86,"5  局部 KKT 资源分配","5  Local KKT resource allocation",
         "共享 ISL 速率 + 星上 CPU 分配","Shared ISL rates and onboard CPU allocation",GREEN),
    ]
    for y,ch,en,sub_ch,sub_en,color in stages:
        box(9.39,y,5.39,.86,fill="white",edge=color,lw=1.15,radius=.08,z=3)
        t(9.66,y+.58,ch,en,size=12.2,weight="bold",color=color)
        t(9.66,y+.25,sub_ch,sub_en,size=10)
    for y in (7.58,6.40,5.22,4.04):
        arrow((12.08,y),(12.08,y-.32),MUTED,1.2)
    # An intra-batch loop is predictive only, before physical atomic submission.
    for a,b in (((14.78,4.47),(15.28,4.47)),((15.28,4.47),(15.28,5.65))):
        arrow(a,b,PURPLE,1.1,ends="-")
    arrow((15.28,5.65),(14.78,5.65),PURPLE,1.1)
    t(15.56,5.04,"同批选择后\n重算预约", "Rebook each\nbatch choice",size=9,color=PURPLE,ha="center",rotation=90)
    t(9.44,2.38,"预约用于预测筛选；实际服务仍按动态共享执行",
      "Calendars predict feasibility; execution uses dynamic sharing",size=10,color=MUTED)
    t(9.44,2.05,"策略不绑定固定卫星 ID；可测试不同星座规模",
      "No fixed satellite-ID output layer; evaluate unseen scales",size=10,color=MUTED)
    arrow((9.39,3.29),(8.28,6.23),GREEN,1.25,(0,(4,3)))
    t(8.72,4.79,"动作 / 分配","Actions / rates",size=9,color=GREEN,ha="center",rotation=70,
      bbox={"facecolor":"white", "edgecolor":"none", "pad":2})

    # Physical execution: store-and-forward, followed by shared CPU. No Q/F double counting.
    box(.5,.29,15.47,1.13,fill="#F3F7F9",edge="#D7E0E8",radius=.13,z=0)
    t(.78,1.10,"任务实际执行", "Physical execution",size=11,weight="bold")
    labels=[(3.00,"完整输入逐跳发送","Full-input forwarding"),
            (6.15,"传播后进入下一跳","Propagation to next hop"),
            (9.46,"输入到达后共享 CPU","Shared CPU after arrival"),
            (13.00,"记录完成 / 违约 / 失败","Completion / violations / failures")]
    for index,(x,ch,en) in enumerate(labels):
        t(x,.78,ch,en,size=11.0,ha="center")
        if index<len(labels)-1:
            arrow((x+1.29,.78),(labels[index+1][0]-1.40,.78),GREEN,1.4)
    t(.79,.49,"未完成任务跨时隙保留剩余工作量；源卫星本地执行直接进入共享 CPU",
      "Residual work persists across slots; local execution enters the source CPU directly",size=9.5,color=MUTED)

    output.mkdir(parents=True, exist_ok=True)
    stem = "system_model_" + language
    for extension in ("png","svg","pdf"):
        path = output / (stem + "." + extension)
        fig.savefig(path,dpi=220,facecolor="white",metadata={"Title":"Contact-aware LEO computing and routing system"} if extension=="pdf" else None)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT / "paper_figures")
    parser.add_argument("--language",choices=["zh","en","both"],default="both")
    args = parser.parse_args()
    for language in (["zh","en"] if args.language=="both" else [args.language]):
        draw(language,args.output)
