import sys
from collections import defaultdict

# 报告中的基准成功率（占位，已填示例，可自行调整）
BASELINE_SR = {
    # 短周期任务示例
    "Stack-Can": {"Seen": 77.78, "Unseen": 62.12},
    "Push-Box": {"Seen": 70.37, "Unseen": 75.76},
    "Open-Drawer": {"Seen": 59.26, "Unseen": 50.00},
    "Close-Drawer": {"Seen": 100.00, "Unseen": 98.48},
    "Flip-Mug": {"Seen": 59.26, "Unseen": 30.77},
    "Pour-Balls": {"Seen": 77.78, "Unseen": 83.33},
    "Open-Laptop": {"Seen": 100.00, "Unseen": 83.33},
    # 长周期任务示例
    "Insert-And-Unload-Cans": {"Seen": 44.44, "Unseen": 31.82},
    "Stack-Can-Into-Drawer": {"Seen": 40.74, "Unseen": 28.79},
    "Sort-Cans": {"Seen": 55.56, "Unseen": 18.18},
    "Unload-Cans": {"Seen": 66.67, "Unseen": 50.00},
    "Insert-Cans": {"Seen": 22.22, "Unseen": 15.15},
}

def parse_metrics_line(line: str):
    # 将 "success: 0 reach_success: 1.0 lift_success: 0.0 ..." 拆成字典
    metrics = {}
    tokens = line.strip().split()
    # 形如 key: value 的成对出现
    for i in range(0, len(tokens) - 1):
        if tokens[i].endswith(":"):
            key = tokens[i][:-1]
            val_str = tokens[i + 1]
            try:
                metrics[key] = float(val_str)
            except ValueError:
                continue
    return metrics

def compute_stats(path: str):
    stats = defaultdict(lambda: {
        "total": 0,
        "success_count": 0,
        "progress_done": 0.0,
        "progress_total": 0.0
    })

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.startswith("Task:"):
            # 提取任务名（Task: XXX, 逗号前）
            task_part = line[len("Task:"):].strip()
            task_name = task_part.split(",")[0].strip()

            # 下一行应为指标行
            if i + 1 < len(lines):
                metrics_line = lines[i + 1]
                m = parse_metrics_line(metrics_line)

                # 累加总次数
                s = stats[task_name]
                s["total"] += 1

                # 成功率：success 为 1 记成功
                succ_val = m.get("success", 0.0)
                if succ_val >= 1.0:
                    s["success_count"] += 1

                # 进度率：统计该次试验的所有 *_success 字段（包括 reach_success、lift_success、insert_success 等）
                sub_done = 0.0
                sub_total = 0.0
                for k, v in m.items():
                    if k.endswith("_success"):
                        sub_total += 1.0
                        sub_done += float(v)

                s["progress_done"] += sub_done
                s["progress_total"] += sub_total

                i += 2
            else:
                # 没有指标行，跳过
                i += 1
        else:
            i += 1

    return stats

def main():
    if len(sys.argv) < 2:
        print("用法: python log_stats.py /path/to/log.txt")
        sys.exit(1)
    path = sys.argv[1]

    # 先计算统计结果，以确定任务顺序
    stats = compute_stats(path)
    task_order = list(stats.keys())  # 保持字典插入顺序（Python 3.7+）

    def _fmt_percent(v):
        if v is None:
            return "N/A"
        try:
            return f"{(v*100.0 if v <= 1.0 else v):.2f}%"
        except Exception:
            return str(v)

    # 按统计任务顺序输出 baseline
    print("Baseline Success Rate:")
    for task in task_order:
        group = BASELINE_SR.get(task, {})
        seen = group.get("Seen")
        unseen = group.get("Unseen")
        print(f"- {task}: Seen={_fmt_percent(seen)}, Unseen={_fmt_percent(unseen)}")

    # 再输出我们的统计结果，顺序一致
    print("\nOur Seen Success Rate：")
    for task in task_order:
        s = stats[task]
        total = s["total"]
        succ = s["success_count"]
        pdone = s["progress_done"]
        ptotal = s["progress_total"]
        success_rate = (succ / total) if total > 0 else 0.0
        progress_rate = (pdone / ptotal) if ptotal > 0 else 0.0
        print(f"- {task}: Success Rate={success_rate:.3f} ({succ}/{total}), Progress Rate={progress_rate:.3f} ({pdone:.1f}/{ptotal:.1f})")

if __name__ == "__main__":
    main()