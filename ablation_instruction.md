# ALOS 参数组合实验
> 目标：在 **Clamp=ON**、**条件积分=ON** 前提下，比较 **reset 机制（hard/soft/no）× leakage** 的组合，找出最优组合。  
> 路径：**仅 Zigzag**。  
> 控制器与船模：controllers

---

## 1) 输出
1. 可执行的demo（做多进程）
2. 结果汇总 CSV：`results_summary.csv`（每行一个 run），该项应该由demo生成
3. 报告 Markdown：`report.md`（详细介绍实验配置，可以给表，给出最优组合 + Top-3），该项由agent写
4. 每个参数配置的表现热力图（每张单独输出png），名称要体现配置（什么reset机制+什么leakage），该项由demo生成

上述内容在工作目录下的exp_data文件中，新建文件夹（名字自拟）放置

---

## 2) 固定配置
- Clamp：ON，默认值（或30度）
- 条件积分：ON，默认值或与之前ablation demo一致
- PID设定/目标速度设定等其它项均参考之前的demo（比如ablation demo）
- 初始状态：初始heading 0, 速度0

> 备注：本实验只需要把 “ALOS 内部参数 + 环境干扰” 做成 sweep，其他一律不动。

---

## 3) 变量：ALOS 参数网格（必须跑全量 18 组）
### 3.1 Reset 机制（3 种）
- `"no_reset"` 
- `"soft_reset"`（soft reset 系数：0.5）
- `"hard_reset"` 

### 3.2 Leakage（6 个值，按固定列表）
- `leakage ∈ [0, 0.1, 0.01, 0.005, 0.003, 0.001]`

总组合：3 × 6 = **18**。

---

## 4) 环境干扰（current + force，同方向）
### 4.1 强度档位（5 档）
- `current_speed ∈ [0.0, 0.1, 0.2, 0.3, 0.4]  m/s`
- `force_magnitude = (current_speed / 0.1) * 5  N`
  - 对应：0, 5, 10, 15, 20 N
- current 与 force **方向相同**。

### 4.2 干扰方向集合（6 档）
- `direction_deg ∈ [0, 30, 90, 150, 180, 270]`度

总环境场景：5 × 6 = **30**。

---

## 5) 参考路径（仅 Zigzag）
使用单一路径，固定为：
- `zigzag_waypoints = [(0,0), (100,100), (0,200), (100,300), (0,400)]`

---

## 6) 实验矩阵规模
- runs_total = 18（参数组合） × 30（环境场景） × 1（路径） = **540 runs**

---

## 7) 指标
每个 run 产出如下指标（写入 `results_summary.csv`）：
1. `cte_rmse`：CTE 的 RMSE
2. `cte_p95`：|CTE| 的 95% 分位

---

## 8) 最优组合判定
定义综合评分（越小越好）：
- 对每个场景（30 个）内，对 18 组做 min-max 归一化：
  - `n(x) = (x - min) / (max - min + eps)`
- 每个 run 的 score：
  - `score = 0.7*n(cte_rmse) + 0.3*n(cte_p95)`
- 对每个参数组合（18 组），在全部 30 个场景上取：
  - `mean_score`（主排序）
  - `std_score`（稳健性参考）

输出：
- Top-1（最优）+ Top-3
- 并在 `report.md` 里写清楚：Top-1 的 `reset_mode`、`soft_reset_coeff`（若有）、`leakage`

---

## 9) 结果文件字段规范
### 9.1 `results_summary.csv` 列
至少包含：
- `run_id`
- `reset_mode`
- `soft_reset_coeff`（0, 0.5, 1）
- `leakage`
- `current_speed`
- `force_magnitude`
- `direction_deg`
- `cte_rmse`
- `cte_p95`
- `score`（按第 8 节算）
- `is_failed`（发散/数值异常/提前退出则 True）
- `fail_reason`（字符串，便于排查）

---

## 10) 本次实验的结论输出格式（report.md 最简模板）
- 实验设置
- 最优组合（Top-1）参数
- Top-3 表格（reset, leakage, mean_score, std_score）
- 在 30 个场景上的 win-rate（Top-1 在多少场景拿第一）
