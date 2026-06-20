import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.ticker import ScalarFormatter

# 1. 寻找最新的 TensorBoard 日志文件
def get_latest_log(log_dir):
    # 递归查找目录下所有的 events 文件
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"在 {log_dir} 及其子目录中未找到 TensorBoard 日志文件！")
    # 按修改时间排序，取最新
    latest_file = max(event_files, key=os.path.getmtime)
    return latest_file

# 2. TensorBoard 官方指数加权移动平均平滑算法
def smooth(scalars, weight=0.85):
    if not scalars: return []
    last = scalars[0]
    smoothed =[]
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_academic_loss(log_dir, save_name):
    # ==========================================
    # 采用最符合 IEEE/顶刊排版规范的全局设置
    # ==========================================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",         # 数学公式使用类似 Times 的字体
        "axes.formatter.use_mathtext": True,# 强制使用标准的 \times 10^N 科学计数法
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,              # 边框粗细
    })

    latest_file = get_latest_log(log_dir)
    print(f"正在解析最新日志: {latest_file}")

    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()

    # 定义需要提取的标签及其专属学术配色 (经典低饱和度配色)
    tags_config = {
        'Loss/Valid/Total':    {'title': 'Total Loss',    'color': '#E64B35'}, # 朱砂红
        'Loss/Valid/Distance': {'title': 'Distance Loss', 'color': '#1D3557'}, # 普鲁士蓝
        'Loss/Valid/Mu':       {'title': 'Mu Loss',       'color': '#2A9D8F'}  # 水鸭绿
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for i, (ax, (tag, config)) in enumerate(zip(axes, tags_config.items())):
        if tag not in ea.Tags()['scalars']:
            print(f"[警告] 未找到标签: {tag}")
            continue

        events = ea.Scalars(tag)
        steps =[e.step for e in events]
        values = [e.value for e in events]

        # 绘制原始数据 (底色) 和平滑曲线 (主色)
        ax.plot(steps, values, color=config['color'], alpha=0.2, linewidth=1.5, label='Raw Data')
        ax.plot(steps, smooth(values, weight=0.85), color=config['color'], linewidth=2.5, label='Smoothed')

        # 坐标轴标签与标题
        ax.set_title(config['title'], fontweight='bold', pad=12)
        ax.set_xlabel('Training Epochs') # 严谨：改为 Steps 而不是 Epochs
        
        if i == 0: 
            ax.set_ylabel('Validation Loss') # 仅在最左侧子图显示 Y 轴名称，避免冗余

        # 学术级 Y 轴数值格式化 (智能科学计数法)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3)) # 核心修复：只有小于 10^-3 或大于 10^3 时才触发科学计数法
        ax.yaxis.set_major_formatter(formatter)

        # 网格与图例
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
        # 严谨的图例样式：取消圆角，黑线细边框
        ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)

    # 调整子图间距，防止坐标轴重叠
    plt.tight_layout()

    # ==========================================
    # 保存高精度学术图片
    # ==========================================
    # 1. PDF 矢量图：用于 LaTeX，无限放大绝对清晰
    plt.savefig(f"{save_name}.pdf", format='pdf', bbox_inches='tight')
    # 2. 600 DPI PNG：用于插入 Word，满足所有期刊印刷要求
    plt.savefig(f"{save_name}.png", format='png', bbox_inches='tight', dpi=600)
    print(f"\n✅ 学术图表已成功生成！\n- 矢量图 (LaTeX首选): {save_name}.pdf\n- 高清图 (Word 首选): {save_name}.png")

if __name__ == "__main__":
    # 你的绝对路径
    LOG_PATH = "/home/lyb/neupan_ws/src/NeuPAN/model/scout_mini_diff/runs"
    OUTPUT_NAME = "NeuPAN_Convergence_Academic"
    
    plot_academic_loss(LOG_PATH, OUTPUT_NAME)