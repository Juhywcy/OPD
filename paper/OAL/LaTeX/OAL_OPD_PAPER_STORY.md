# OAL-OPD AAAI 2027 论文故事

**暂定标题**：Outcome-Aligned On-Policy Distillation for Long-Horizon Mathematical Reasoning

**目标会议**：AAAI 2027

**代码入口**：`/Users/juhy/Library/CloudStorage/OneDrive-个人/drpaper/OPD/OPD/scripts/train/run_outcome_aligned_logit_opd.sh`

**当前经验事实**：你已经观察到，使用 outcome-aligned 的 token candidate 去训练，比原始 OPD 更好。论文要做的事情，是把这个经验发现转化成一个清晰、可验证、能被 reviewer 接受的技术故事：最终 outcome 可以校准 dense teacher supervision，但不需要把粗粒度 outcome reward 广播到每个 token。

## 一句话故事

On-policy distillation 提供了细粒度 token-level teacher supervision，但 dense 不等于 reliable；OAL-OPD 说明，最终二值 outcome 可以作为 teacher-student token preference 的方向校验信号，在保留 OPD 细粒度监督的同时，过滤掉与最终结果方向不一致的 token-candidate 更新。

## Reviewer 应该读到的主线

论文开头应该从长程推理训练里的一个矛盾切入。Outcome reward 比较可信，因为它直接告诉我们最终答案是否正确；但它太稀疏，无法解释几千个中间 token 哪些有用、哪些有害。OPD 类方法正好相反：teacher 在 on-policy trajectory 上给出 dense token-level candidate preference，监督非常细；但它默认 teacher 的局部偏好处处可信。

在长程数学推理里，这个默认假设并不稳。一条 response 最终可能正确，但中间局部 token 上 teacher 的偏好不一定总是强；一条 response 最终可能错误，但其中很多局部 token 也可能看起来很合理。也就是说，teacher dense reward 有粒度，但未必总和最终 correctness 对齐。

OAL-OPD 的核心观察是：最终 outcome 虽然粗，但能提供一个有价值的方向约束。对于最终正确的 response，如果 teacher 比 student 更支持某个 candidate，那么这个更新方向和成功轨迹是一致的；对于最终错误的 response，如果 teacher 比 student 更不支持某个 candidate，那么这个更新方向和抑制失败轨迹是一致的。反过来的更新就可疑，因为它们要求 student 朝着与最终 outcome 相冲突的方向移动。

因此，OAL-OPD 应该被写成 OPD 上的一个 calibration layer，而不是新的 RL 算法，也不是 outcome reward mixing。它保留原始 OPD 的 dense reward，只用最终 outcome 去判断每个 token-candidate 级别的 teacher-student log-probability gap 是否方向正确。最重要的区别是：outcome 不是 token reward；outcome 只是 dense teacher update 的方向门控。

## 方法核心

对 prompt `x`，从 student 采样 on-policy responses `a_i`。最终 verifier 或 reward function 给出：

```text
y_i in {0, 1}
```

其中 `1` 表示最终答案正确，`0` 表示最终答案错误。

对第 `i` 条 response、第 `t` 个 token position、第 `k` 个 top-k candidate，定义 teacher-student log-probability gap：

```text
Delta_{i,t,k} =
  log p_T(c_{i,t,k} | x, a_{i,<t})
  -
  log p_S(c_{i,t,k} | x, a_{i,<t})
```

实现里这个量叫 `logit_delta_scores`，但论文里要注意说清楚：它实际是 log probability gap，不是 raw logits。

原始 OPD 的 candidate reward 可以写成：

```text
r^OPD_{i,t,k} = w_{i,t,k} Delta_{i,t,k}
```

其中 `w` 是 candidate weight。当前脚本默认设置是：

```bash
LOG_PROB_TOP_K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
OPD_TOPK_RENORMALIZE=True
```

OAL 增加一个 outcome-alignment mask：

```text
M_{i,t,k} = 1[Delta_{i,t,k} >  margin]   if y_i = 1
M_{i,t,k} = 1[Delta_{i,t,k} < -margin]   if y_i = 0
```

最终训练 advantage：

```text
A_{i,t,k} = M_{i,t,k} r^OPD_{i,t,k}
```

当前脚本默认：

```bash
ADV_ESTIMATOR=outcome_aligned_logit_opd
OAL_MARGIN=0.0
N_RESPONSES=4
MAX_RESP_LENGTH=8192
MAX_VAL_RESP_LENGTH=31744
```

这给论文一个非常简单的技术叙事：方法只改变哪些 OPD dense updates 被信任；它不改变 rollout 生成方式，也不改变最终 outcome 的打分方式。

## Claims 和证据

| Claim | 需要证明什么 | 最低证据 |
|---|---|---|
| OPD 存在 outcome-misaligned token-candidate updates。 | 正确轨迹和错误轨迹里的 `teacher_logp - student_logp` 符号分布不同。 | 按 final correctness 分组的正/负 candidate ratio 直方图或柱状图。 |
| OAL-OPD 优于原始 OPD。 | OAL 在数学推理验证集上超过 OPD。 | AIME24、AIME25、AMC23，最好加 MATH-500；报告 pass@1 和 pass@16。 |
| OAL 优于简单 outcome broadcasting。 | OAL 保留 dense teacher 信息，同时避免把粗粒度 outcome 分配到所有 token。 | 和 GRPO-only、outcome-mix、hard outcome-mix、OPD 对比。 |
| 收益不是因为梯度更少。 | 方向过滤比单纯降低更新规模更重要。 | random mask 或 magnitude-matched OPD ablation；同时报告 kept-candidate ratio。 |
| OAL 额外开销低。 | OAL 复用 OPD 已经计算出的 tensor。 | runtime/memory 表，或实现分析：`logit_delta_scores`、`true_reward_score`、`oal_keep_mask`。 |

## 论文结构

### Abstract

Abstract 不要从泛泛的 LLM 背景开始。第一句就点出 sparse outcome reward 和 dense teacher reward 的矛盾。然后介绍 OAL-OPD：用 outcome 对 teacher-student token preference 做方向过滤。最后一句填入最终主结果。

建议结构：

```text
On-policy distillation provides dense token-level supervision for reasoning models, but its teacher preferences can be locally misleading on long trajectories. We propose OAL-OPD, which uses the final outcome of a sampled response only as a direction check for dense teacher-student token preferences. For correct responses, OAL keeps teacher-supported candidate updates; for wrong responses, it keeps updates that suppress student-favored candidates. This preserves OPD's dense signal without broadcasting coarse outcome rewards across all tokens. Experiments on [benchmarks] show that OAL-OPD improves [metric] over OPD and [baselines], with diagnostics confirming that the method filters outcome-inconsistent updates rather than simply shrinking gradients.
```

### 1. Introduction

Introduction 第一页结束前要回答三个问题。

**What**：OPD 的 dense teacher reward 在长程推理中可能和最终正确性不一致。

**Why hard**：Outcome reward 太稀疏，难以给几千个 token 分配信用；teacher dense reward 又太局部，不能处处相信。

**So what**：OAL 用 final outcome 作为 sign-level calibration，让 dense teacher reward 只在方向合理时进入训练。

推荐 contribution bullets：

1. 我们识别出 OPD 在长程数学推理中的一个 failure mode：token-candidate 级别的 teacher update 可能与最终 outcome 不一致。
2. 我们提出 OAL-OPD，一个简单的 outcome-aligned mask，根据 teacher-student log-probability gap 的符号过滤 dense OPD updates。
3. 我们在数学推理 benchmark 上证明 OAL-OPD 优于 OPD 和 outcome-based baselines。
4. 我们通过 diagnostics 和 ablations 展示 filter 何时激活、保留多少 candidate，以及收益不是单纯来自减小梯度规模。

### 2. Related Work

Related Work 按问题组织，不要按论文流水账组织。

- On-policy distillation 和 teacher-guided RL。
- Outcome supervision、GRPO、trajectory-level reinforcement learning。
- Process supervision 和 token-level credit assignment。
- Teacher signal reliability 与 uncertainty-aware distillation。

核心定位句：

```text
Unlike outcome-reward methods, OAL does not convert final correctness into a token reward. Unlike OPD, it does not trust every dense teacher preference. It combines the two by using final correctness only to decide whether a dense teacher update has the right sign.
```

### 3. Method

Method 要短而精确。

1. 定义 OPD：student policy、teacher policy、top-k candidate、teacher-student log-probability gap。
2. 解释 gap 的符号为什么在正确和错误 trajectory 上含义不同。
3. 定义 OAL mask 和 final advantage。
4. 给出实现细节：`only_stu`、top-k 16、student-probability weighting、top-k renormalization、margin。
5. 说明复杂度：相对 OPD 不需要额外 teacher forward；mask 只使用已有 log probabilities 和 outcome scores。

注意不要过度声称 OAL 找到了真正 causal token。更准确的说法是：OAL 过滤掉违反 trajectory-level sign constraint 的 dense updates。

### 4. Experiments

实验部分应该用一个主结果表和一张训练曲线撑起 empirical spine。

主结果表：

| Method | AIME24 pass@1 | AIME24 pass@16 | AIME25 pass@1 | AIME25 pass@16 | AMC23 pass@1 | AMC23 pass@16 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OPD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Outcome-mix | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| OAL-OPD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

训练曲线：

- x-axis：training step。
- y-axis：average pass@16 或 validation accuracy。
- curves：OPD vs OAL-OPD。
- 标注 OAL 的收益是 final quality、更快收敛，还是训练稳定性。

Ablation 表：

- `OAL_MARGIN`: `0.0`, `0.03`, `0.05`, `0.1`。
- `LOG_PROB_TOP_K`: `8`, `16`, `32`。
- `OPD_TOPK_RENORMALIZE`: `True` vs `False`。
- `TOP_K_STRATEGY`: `only_stu`, `intersection`, `union`。
- `REWARD_WEIGHT_MODE`: `student_p`, `teacher_p`, `none`。

### 5. Analysis

Analysis 的目标是证明机制，而不是只堆更多结果。

Diagnostic 1：sign alignment。

- 对正确 response，`Delta > 0` 的 candidate 比例是多少？
- 对错误 response，`Delta < 0` 的 candidate 比例是多少？
- 这个比例随训练如何变化？

Diagnostic 2：filtering behavior。

- 画 `oal/kept_candidate_ratio`。
- 画 `oal/token_keep_ratio`。
- 画 `oal/correct_response_ratio`。
- 证明 OAL 不是删掉几乎所有 token，也不是等价于常数 scale-down。

Diagnostic 3：哪里更有效。

- 按 benchmark 难度拆：AMC23 vs AIME24/AIME25。
- 按 response length bucket 拆。
- 如果收益集中在长输出上，就强调 long-horizon credit assignment。
- 如果收益集中在更短或更简单输出上，就强调稳定 high-confidence local improvement。

### 6. Limitations

这部分要主动写，reviewer 反而更容易接受。

- OAL 依赖 outcome correctness；verifier 有噪声时可能过滤掉有用更新。
- 二值 outcome 不能定位真正 causal token。
- 当前实现依赖 top-k candidate coverage。
- 当前实验主要集中在数学推理；对 tool use 或通用 instruction following 的泛化还需要验证。

### 7. Conclusion

结尾收束到一个窄而稳的 claim：

```text
Final outcomes can calibrate dense distillation at the token-candidate level. OAL-OPD keeps OPD's fine-grained teacher signal while removing updates whose direction contradicts trajectory-level correctness, leading to stronger and more interpretable on-policy distillation for mathematical reasoning.
```

## 图表计划

| ID | 作用 | 内容 | 优先级 |
|---|---|---|---|
| Fig. 1 | Hero method figure | 三栏对比：OPD 保留所有 dense updates；outcome-mix 广播 scalar outcome；OAL 只保留 outcome-aligned token-candidate updates。 | High |
| Table 1 | 主结果 | OPD、OAL-OPD、GRPO、outcome-mix 在 AIME24/AIME25/AMC23/MATH-500 上的结果。 | High |
| Fig. 2 | 训练动态 | OPD vs OAL 的 validation pass@16 over steps。 | High |
| Fig. 3 | Mask diagnostics | kept-candidate ratio、token-keep ratio、correct-response ratio。 | High |
| Table 2 | Ablations | margin、top-k size、renormalization、candidate strategy。 | High |
| Fig. 4 | 机制分析 | 按 final correctness 拆分 `Delta` 的符号分布。 | High |
| Fig. 5 | 长度分析 | 按 response-length bucket 比较 OPD 和 OAL。 | Medium |

## 推荐写作顺序

1. 先锁主结果表。Introduction 应该围绕最稳定、最强的指标来写。
2. 先画 Fig. 1 再写 Method。图清楚，Method 就更容易写短。
3. 先写 Method 再写 Related Work。Related Work 的定位依赖你对 filtering 和 broadcasting 的精确定义。
4. 先写 Analysis 再写 Abstract。Abstract 里的 diagnostic claim 要等分析图确认后再写。

## 完整初稿前最低实验清单

- OPD vs OAL-OPD：至少 AIME24、AIME25、AMC23。
- 至少一个 sparse outcome baseline：GRPO 或 outcome-mix。
- `OAL_MARGIN` ablation。
- `LOG_PROB_TOP_K` ablation，或者解释为什么固定为 `16`。
- 确认并报告 `OPD_TOPK_RENORMALIZE=True`，因为旧笔记里这个默认值曾经不同。
- Mask diagnostics：
  - `oal/kept_candidate_ratio`
  - `oal/positive_candidate_ratio`
  - `oal/negative_candidate_ratio`
  - `oal/token_keep_ratio`
  - `oal/correct_response_ratio`
  - `oal/outcome_score_mean`

## 风险和应对

**风险：reviewer 觉得 OAL 太简单。**

应对：论文重点不要放在实现复杂度，而要放在 diagnostic insight 和 filtering principle。必须有 sign-alignment 图和 magnitude-matched mask ablation。

**风险：提升只出现在一个 benchmark。**

应对：诚实报告 benchmark-wise results，把主 claim 放在稳定 aggregate 上。如果收益不均匀，就用 difficulty 和 length analysis 解释。

**风险：OAL 被误解成 reward mixing。**

应对：在 Introduction、Method、Fig. 1 caption 里反复强调：outcome 是 direction gate，不是 token reward。

**风险：OPD baseline 不够强。**

应对：除了 estimator，其余训练设置保持一致：training steps、model、data、top-k、response count、validation sampling 都要 match。

## 具体下一步

先整理一个结果表，每个 run 一行：

```text
run_name, estimator, dataset, actor, teacher, top_k, margin, seed,
AIME24_pass1, AIME24_pass16,
AIME25_pass1, AIME25_pass16,
AMC23_pass1, AMC23_pass16,
kept_candidate_ratio, token_keep_ratio, final_step
```

这个表填好以后，论文的 title、abstract、contribution bullets 就可以从 placeholder 变成精确表述。
