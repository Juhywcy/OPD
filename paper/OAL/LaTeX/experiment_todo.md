# POV-OPD Experiment TODO

本文档根据论文当前的 `experiments.tex`、`results_table.tex` 和
`gradient_experiments.tex` 整理。论文关注的是 **OPD 梯度可靠性修正**，不包含
长度压缩、LAC 或 accuracy--efficiency 实验。

## 0. 统一实验约定

- [ ] 训练集统一使用 DAPO-Math-17K。
- [ ] 同一 Student/Teacher 配置下，GRPO、Raw OPD 和 POV-OPD 使用相同的学生初始化、训练数据、rollout budget 和优化超参数。
- [ ] 所有模型统一评估 AMC23、AIME24、AIME25、MATH-500、GPQA 和 Minerva。
- [ ] 每个问题独立采样 8 个回答。
- [ ] 报告 Avg@8：全部 8 个回答的平均正确率。
- [ ] 将论文中的 Best@8 改为 Pass@8：每题 8 个回答中至少一个正确的比例。
- [ ] 固定并记录评估的 prompt template、temperature、top-p、最大生成长度和答案判定器。
- [ ] 对模型差异报告 paired、problem-level bootstrap 95% CI；AIME 等小数据集尤其需要不确定性。

## 1. 主结果：GRPO vs. Raw OPD vs. POV-OPD

### 1.1 训练任务

| Student | Teacher | GRPO | Raw OPD | POV-OPD |
|---|---|---:|---:|---:|
| Qwen3-1.7B | Qwen3-4B Thinking | [ ] | [ ] | [ ] |
| DS-R1-Distill-Qwen-1.5B | JustRL-1.5B | [ ] | [ ] | [ ] |
| DS-R1-Distill-Qwen-1.5B | Skywork-OR1-Math-7B | 复用上一行 GRPO | [ ] | [ ] |
| DS-R1-Distill-Qwen-7B | Skywork-OR1-Math-7B | [ ] | [ ] | [ ] |

共 11 个不同训练任务，而不是 12 个。GRPO 不查询教师，因此两个
DS-R1-Distill-Qwen-1.5B 配置必须复用同一个 GRPO checkpoint 和结果。

### 1.2 每个 checkpoint 的评估

- [ ] AMC23：Avg@8、Pass@8
- [ ] AIME24：Avg@8、Pass@8
- [ ] AIME25：Avg@8、Pass@8
- [ ] MATH-500：Avg@8、Pass@8
- [ ] GPQA：Avg@8、Pass@8
- [ ] Minerva：Avg@8、Pass@8

主表共有：

$$
4\times 3\times 6\times 2=144
$$

个结果单元格。还应保存逐题、逐 sample 的原始预测和 correctness，便于重新计算
Pass@8、置信区间和显著性。

### 1.3 主结果判定

- [ ] POV-OPD 的 Avg@8 是否稳定高于 Raw OPD。
- [ ] POV-OPD 的 Pass@8 是否不降低成功轨迹的探索能力。
- [ ] POV-OPD 是否优于 GRPO，从而支持 dense teacher guidance 的价值。
- [ ] 检查是否存在单个 benchmark 明显退化，不能只报告 macro average。

## 2. Outcome / Prefix / Support 组件消融

本组只使用：

- Student：DeepSeek-R1-Distill-Qwen-1.5B
- Teacher：JustRL-1.5B

| Variant | Outcome | Prefix | Support | Training | Evaluation |
|---|---:|---:|---:|---:|---:|
| Raw OPD | off | off | off | [ ] | [ ] |
| OPD + Outcome | on | off | off | [ ] | [ ] |
| OPD + Prefix | off | on | off | [ ] | [ ] |
| OPD + Outcome + Prefix | on | on | off | [ ] | [ ] |
| Full POV-OPD | on | on | on | [ ] | [ ] |

Raw OPD 和 Full POV-OPD 可以复用主实验。因此，若这两个 checkpoint 已完成，只需
额外训练 Outcome-only、Prefix-only 和 Outcome+Prefix 三个变体。

### 2.1 OPD + Prefix 配置

其余训练超参数完全继承原始训练脚本，只改变组件开关：

```bash
ADV_ESTIMATOR=prefix_trust_oal_opd \
OAL_ENABLED=True \
OAL_SPLIT_MODE=all \
OAL_WEIGHT_MODE=hard \
PT_OAL_PREFIX_TRUST_ENABLED=True \
PT_OAL_ALIGNED_BOOST_ALPHA=0.0 \
RUN_NAME=ablation_opd_prefix_ds15b_justrl15b \
bash scripts/train/run_prefix_trust_oal_opd.sh
```

该配置对应：

$$
A_{i,t}^{\mathrm{OPD+Prefix}}
=A_{i,t}^{\mathrm{OPD}}w_{i,t}^{\mathrm{pre}}.
$$

### 2.2 消融结果

每个变体都在六个主数据集上计算 Avg@8 和 Pass@8，再对六个数据集等权平均：

- [ ] Raw OPD：Macro Avg@8、Macro Pass@8
- [ ] Outcome-only：Macro Avg@8、Macro Pass@8
- [ ] Prefix-only：Macro Avg@8、Macro Pass@8
- [ ] Outcome+Prefix：Macro Avg@8、Macro Pass@8
- [ ] Full POV-OPD：Macro Avg@8、Macro Pass@8
- [ ] 对各项增益给出评估不确定性，避免把采样噪声解释为组件收益。

## 3. Independent Outcome-Reference Gradient Audit

本实验 **不训练模型**。冻结 DS-R1-Distill-Qwen-1.5B / JustRL-1.5B 的最终
POV-OPD checkpoint，只进行 rollout、forward 和 backward，不执行
`optimizer.step()`。

### 3.1 协议

- [ ] 预先固定 held-out audit prompts。
- [ ] 对相同 prompts 独立采样 rollout group $A_j$ 和 $B_j$。
- [ ] 在 $A_j$ 上分别计算 Outcome-PG、Raw OPD 和 POV-OPD gradient。
- [ ] 在 $B_j$ 上计算独立 outcome-gradient reference $g^{\mathrm{out}}_{j,B}$。
- [ ] 所有 advantage 和 validation weight 均 detach。
- [ ] 三个候选梯度使用完全相同的参数集合、mask 和 loss normalization。
- [ ] 预先固定 reference norm threshold 和 candidate near-zero threshold。
- [ ] 仅使用 reference-valid audit batch 构造所有方法共享的集合 $\mathcal J$。

### 3.2 报告指标

对 Outcome-PG、Raw OPD 和 POV-OPD 分别报告：

- [ ] Mean cosine 及 uncertainty
- [ ] Negative-cosine batch rate
- [ ] Near-zero candidate-gradient rate
- [ ] Relative gradient norm
- [ ] 共同的 reference-valid audit batch 数 $|\mathcal J|$

需要确认：

- [ ] POV-OPD 的 mean cosine 高于 Raw OPD。
- [ ] POV-OPD 的 negative-cosine rate 低于 Raw OPD。
- [ ] 改进不能仅来自把梯度缩小到接近零，因此必须同时报告 near-zero rate 和 relative norm。

若只审计最终 checkpoint，应将论文中的 “at each pre-specified frozen student
checkpoint” 改为 “at the final frozen student checkpoint”。若只计算 `lm_head` 等
参数子空间，论文中必须称为 gradient-subspace audit。

## 4. $b^\star$ 前后的 Prefix Horizon Audit

本实验同样 **不需要训练**，可以复用第 3 节的冻结 checkpoint、held-out prompts
以及独立 outcome reference。

### 4.1 Raw OPD window gradient

对每个未施加 prefix weighting 的 window 计算：

$$
g^{\mathrm{OPD}}_{i,b}
=\frac{1}{|W_{i,b}|}\sum_{t\in W_{i,b}}
D_{i,t}\nabla_\theta\log\pi_\theta(y_{i,t}\mid h_{i,t}).
$$

定义：

$$
\mathrm{Conflict}_{i,b}
=\mathbf{1}\!\left[
\left\langle g^{\mathrm{OPD}}_{i,b},
g^{\mathrm{out}}_{j,B}\right\rangle<0
\right].
$$

trigger window 和其后的 window 都归入 post-$b^\star$。

### 4.2 Position-matched control

每条 triggered trajectory 匹配一条 control：

- [ ] 相同 prompt。
- [ ] 相同 outcome class。
- [ ] 响应长度相似；运行前固定 length caliper。
- [ ] control 未触发 CUSUM。
- [ ] 一对一匹配，建议不放回。
- [ ] control 使用 triggered trajectory 的相同 normalized position 作为伪边界。

### 4.3 必须报告

- [ ] 审计 trajectory 总数
- [ ] CUSUM trigger rate
- [ ] Triggered trajectory 数量
- [ ] 成功 matched pair 数量和匹配率
- [ ] Triggered conflict rate：pre、post、post-pre
- [ ] Control conflict rate：pre、post、post-pre
- [ ] Triggered mean cosine：pre、post
- [ ] Control mean cosine：pre、post
- [ ] Conflict DID 及 cluster-bootstrap 95% CI
- [ ] Cosine DID 及 cluster-bootstrap 95% CI

$$
\mathrm{DID}
=
(\mathrm{post}-\mathrm{pre})_{\mathrm{triggered}}
-(\mathrm{post}-\mathrm{pre})_{\mathrm{control}}.
$$

有说服力的结果应满足：

- [ ] Conflict DID 为正。
- [ ] Conflict DID 的 95% CI 不包含 0。
- [ ] Triggered 组 post-$b^\star$ cosine 的下降大于 control。
- [ ] Cosine DID 的 CI 支持上述结论。

CI 应按 prompt 或 audit batch 做 cluster bootstrap，不能把同一 trajectory 的多个
window 当成相互独立的样本。

### 4.4 实现入口

- 第 3 节与 window gradient：`verl/recipe/repro/pov_gradient_audit.py`
- 第 4 节匹配、DID 与 cluster bootstrap：`verl/recipe/repro/prefix_horizon_audit.py`
- 统一启动脚本：`scripts/audit/run_pov_audits.sh`
- 参数、输出和 smoke test：`verl/recipe/repro/POV_AUDIT.md`

实现为独立的离线审计流程，不修改训练 trainer/actor，不创建 optimizer，也不会更新
checkpoint。默认 `GRADIENT_PARAMETER_REGEX=lm_head`，因此论文中应称为
gradient-subspace audit。

## 5. 当前论文表格需要补充或修正

- [ ] 将所有 Best@8 统一改为 Pass@8，除非实际计算的是八次完整评估中的最大准确率。
- [ ] 在 Prefix audit 中增加 Triggered trajectory 数量。
- [ ] 在 Prefix audit 中增加 matched rate。
- [ ] 在 Prefix audit 表格中直接报告 Conflict DID 及 95% CI。
- [ ] 增加 Cosine DID 及 95% CI，或删除相应显著性主张。
- [ ] 明确 audit checkpoint、A/B response 数、audit 数据、gradient parameter scope 和数值阈值。
- [ ] 明确 length-matching caliper、是否不放回匹配，以及 bootstrap 单位。
- [ ] 评估初始 Student 和 Teacher，验证 Teacher 在所用任务上确实更强。

## 6. 推荐执行顺序

1. [ ] 完成四组配置的主训练任务。
2. [ ] 对所有主模型统一完成六数据集 Avg@8/Pass@8 评估。
3. [ ] 在 DS-R1-Qwen-1.5B / JustRL-1.5B 上完成三个额外消融训练。
4. [ ] 用最终 POV checkpoint 完成 independent-reference gradient audit。
5. [ ] 复用 audit rollout 完成 $b^\star$ matched-control DID。
6. [ ] 计算置信区间、填入 `sections/experiment_values.tex` 和主结果表。
7. [ ] 根据实测结果修改实验分析、摘要和结论，避免提前写入未被数据支持的结论。
