# ODW-OPD: Outcome-Discriminative Window OPD

## 背景问题

原始 OPD 使用 teacher 在 on-policy 轨迹上的 token-level dense reward 来监督 student。这个信号的优点是细粒度，能够告诉模型每个 token 附近 teacher 更偏好哪些候选 token；但它的问题是，teacher dense reward 不一定总是和最终 outcome 一致。

尤其在长推理任务里，teacher 可能在局部 token 上给出看似合理的高分，但整条轨迹最后是错的；也可能一条最终正确的轨迹在中间某些局部 token 上被 teacher 低估。也就是说，teacher dense reward 有时会把正确轨迹和错误轨迹排反。

之前直接把 outcome reward 广播到长 suffix 的做法不稳定，因为 0/1 outcome reward 太粗，会污染大量中间 token。ODW-OPD 的目标不是用 outcome reward 替代 teacher dense reward，而是用 outcome reward 判断 teacher dense reward 在局部窗口内是否有正确的排序能力。

## 核心思想

对同一个 prompt 采样多条 on-policy response。根据最终答案，把这些 response 分成正确轨迹和错误轨迹。

然后按固定窗口切分 response，在每个窗口里比较：

- 正确轨迹在该窗口的平均 teacher dense reward
- 错误轨迹在该窗口的平均 teacher dense reward

如果 teacher 在这个窗口里给正确轨迹的 reward 高于错误轨迹，说明这个窗口的 teacher dense reward 与 outcome 一致，可以保留较强监督。

如果 teacher 在这个窗口里给错误轨迹的 reward 高于正确轨迹，说明这个窗口的 teacher dense reward 排序不可靠，应当降低该窗口的监督权重。

## Prompt Group 筛选

对每个 prompt group，假设采样了 `N` 条 response。

设第 `i` 条 response 的 outcome 为：

$$
y_i \in \{0, 1\}
$$

其中 `1` 表示最终答案正确，`0` 表示最终答案错误。

只保留 mixed-outcome group：

$$
0 < \sum_i y_i < N
$$

也就是说：

- 全对 group 不训练
- 全错 group 不训练
- 有对有错的 group 才训练

这样做的原因是：ODW-OPD 需要在同一个 prompt 内比较正确轨迹和错误轨迹。如果一个 group 全对或全错，就没有局部排序对比信号。

## 窗口级 Margin

把 response 划分成长度为 `W` 的窗口。第 `j` 个窗口为：

$$
\mathcal{W}_j = [jW, (j+1)W)
$$

对第 `i` 条 response，在窗口 `j` 上计算 teacher dense reward 的局部平均：

$$
s_{i,j}
=
\frac{
    \sum_{t \in \mathcal{W}_j} m_{i,t} r^{teacher}_{i,t}
}{
    \sum_{t \in \mathcal{W}_j} m_{i,t} + \epsilon
}
$$

其中：

- $r^{teacher}_{i,t}$ 是 OPD 给出的 token-level teacher dense reward
- $m_{i,t}$ 是 response mask
- $\epsilon$ 是防止除零的小常数

窗口 `j` 的 outcome-discriminative margin 定义为：

$$
\Delta_j
=
\mathbb{E}_{i:y_i=1}[s_{i,j}]
-
\mathbb{E}_{i:y_i=0}[s_{i,j}]
$$

直观含义：

- $\Delta_j > 0$：teacher 在这个窗口更偏好最终正确轨迹，局部 reward 排序可信
- $\Delta_j < 0$：teacher 在这个窗口更偏好最终错误轨迹，局部 reward 排序不可信
- $\Delta_j \approx 0$：teacher 在这个窗口无法区分正确和错误轨迹

## 窗口权重

把 margin 转成窗口监督权重：

$$
g_j
=
\sigma\left(
    \frac{\Delta_j - \delta}{\tau}
\right)
$$

其中：

- $\delta$ 是 margin threshold，默认 `0.0`
- $\tau$ 是 temperature，控制 sigmoid 的平滑程度

如果 $\Delta_j$ 明显大于 $\delta$，则 $g_j$ 接近 1。

如果 $\Delta_j$ 小于 $\delta$，则 $g_j$ 接近 0。

## 最终 Advantage

ODW-OPD 不引入额外 outcome advantage，也不把 outcome reward 广播到 token 上。它只用 outcome 来校准 teacher dense reward 的局部可信度。

最终 token advantage 为：

$$
A_{i,t}
=
g_{w(t)} \cdot r^{teacher}_{i,t}
$$

其中 $w(t)$ 表示 token `t` 所属的窗口。

如果 OPD reward 是 top-k 维度：

$$
r^{teacher}_{i,t,k}
$$

则同一个窗口权重作用到该 token 的所有 top-k candidate：

$$
A_{i,t,k}
=
g_{w(t)} \cdot r^{teacher}_{i,t,k}
$$

## 为什么不使用 w_min

这里不设置最小权重 `w_min`。

原因是：如果某个窗口 teacher dense reward 明显把正确和错误轨迹排反，那么该窗口的监督应当被充分压低。保留一个固定下限会让错误 teacher signal 继续进入训练，削弱 outcome 校准的意义。

为了避免全 batch 没有有效 token，代码里只在 actor loss 层面做了保护：如果当前 micro-batch 的 ODW 有效 mask 全为 0，则回退到原始 response mask，防止 loss denominator 为 0。

## 与之前方法的区别

### 相比原始 OPD

原始 OPD 默认所有 teacher dense reward 都可信：

$$
A_{i,t} = r^{teacher}_{i,t}
$$

ODW-OPD 会检查 teacher dense reward 是否真的把正确轨迹排在错误轨迹前面：

$$
A_{i,t} = g_{w(t)} r^{teacher}_{i,t}
$$

### 相比 reliability-based AH-OPD

AH-OPD 估计每个 token 的 teacher reliability，主要依赖 top-k overlap、entropy gap、prefix overlap 等模型内部信号。

ODW-OPD 不再显式计算 token reliability，而是使用 outcome 分组，在窗口层面检验 teacher dense reward 是否与最终正确性一致。

### 相比 outcome-mix

outcome-mix 会把 outcome reward 加到 horizon 后的 token 上，容易把粗粒度 0/1 信号扩散到大量中间 token。

ODW-OPD 不广播 outcome reward。Outcome 只用于判断 teacher dense reward 的局部排序是否可靠，最终训练信号仍然来自 teacher dense reward。

## 当前实现位置

入口脚本：

```bash
run_odw_opd_qwen3_0p6b.sh
```

核心 estimator：

```python
@register_adv_est("outcome_discriminative_window_opd")
def compute_outcome_discriminative_window_opd_advantage(...)
```

实现文件：

```text
verl/verl/trainer/ppo/core_algos_odw_opd.py
```

训练入口：

```bash
python3 -m verl.trainer.main_ppo_odw_opd
```

Actor：

```text
verl/verl/workers/actor/dp_actor_odw_opd.py
```

Trainer：

```text
verl/verl/trainer/ppo/ray_trainer_odw_opd.py
```

## 关键超参数

### `ODW_WINDOW_SIZE`

窗口长度。

默认：

```bash
ODW_WINDOW_SIZE=512
```

窗口越小，定位越细，但 margin 估计更噪。

窗口越大，估计更稳，但会损失 token-level 信用分配的细粒度。

### `ODW_MARGIN_DELTA`

margin threshold。

默认：

```bash
ODW_MARGIN_DELTA=0.0
```

只有当正确轨迹的窗口 reward 高于错误轨迹时，窗口权重才会超过 0.5。

如果设置为正数，要求 teacher 明显更偏好正确轨迹才保留强监督。

### `ODW_TEMPERATURE`

sigmoid temperature。

默认：

```bash
ODW_TEMPERATURE=0.1
```

越小，窗口权重越接近 hard gate。

越大，窗口权重变化越平滑。

### `ODW_FILTER_MIXED`

是否过滤全对和全错 group。

默认：

```bash
ODW_FILTER_MIXED=True
```

建议保持为 True，因为该方法依赖同 prompt 内正确/错误轨迹的对比。

## 日志指标

当前实现会记录：

```text
odw/window_weight_mean
odw/window_weight_min
odw/window_weight_max
odw/window_delta_mean
odw/kept_sample_ratio
odw/mixed_group_ratio
odw/all_correct_group_ratio
odw/all_wrong_group_ratio
odw/outcome_score_mean
odw/positive_sample_ratio
odw/active_token_ratio
```

重点看：

- `odw/mixed_group_ratio`：有多少样本来自 mixed-outcome group
- `odw/active_token_ratio`：真正参与 loss 的 token 比例
- `odw/window_weight_mean`：teacher dense reward 平均保留强度
- `odw/window_delta_mean`：teacher 是否整体更偏好正确轨迹
- `odw/all_correct_group_ratio` 和 `odw/all_wrong_group_ratio`：被过滤掉的 group 占比

## 预期现象

如果 teacher dense reward 与 outcome 排序一致，应该看到：

- `odw/window_delta_mean > 0`
- `odw/window_weight_mean` 较高
- 训练接近原始 OPD，但能压制部分错误窗口

如果 teacher dense reward 经常把正确/错误轨迹排反，应该看到：

- `odw/window_delta_mean <= 0`
- `odw/window_weight_mean` 降低
- ODW 会自动减弱这些窗口的 teacher 监督

如果 `odw/mixed_group_ratio` 很低，说明当前采样 group 大量全对或全错，该方法有效训练样本不足。此时可以考虑：

- 增大 `N_RESPONSES`
- 调整 sampling temperature
- 换更合适难度的数据
- 使用更难或更简单的 prompt 子集做 ablation

## 当前默认运行方式

```bash
bash run_odw_opd_qwen3_0p6b.sh
```

默认使用：

```bash
ADV_ESTIMATOR=outcome_discriminative_window_opd
TOP_K_STRATEGY=union
REWARD_WEIGHT_MODE=student_p
OPD_TOPK_RENORMALIZE=False
ODW_WINDOW_SIZE=512
ODW_MARGIN_DELTA=0.0
ODW_TEMPERATURE=0.1
ODW_FILTER_MIXED=True
```

# OAL-OPD: Outcome-Aligned Logit OPD

## 背景问题

原始 OPD 的 token-level dense reward 来自 teacher 和 student 在候选 token 上的 log probability 差异。它能够提供细粒度监督，但这个监督方向不一定总是和最终 outcome 一致。

如果一条 response 最终是正确的，那么更合理的监督是：强化 teacher 相比 student 更支持的 token。

如果一条 response 最终是错误的，那么更合理的监督是：只保留 teacher 相比 student 更不支持的 token，使错误轨迹上的有害 token 被压低。

OAL-OPD 的目标是让每个 token candidate 的 OPD 监督方向与最终 outcome 对齐。

## 核心思想

对每个 sampled response，先用 reward function 得到最终 outcome：

$$
y_i \in \{0, 1\}
$$

其中 $y_i=1$ 表示答案正确，$y_i=0$ 表示答案错误。

对每个 token position 和 top-k candidate，计算 teacher 与 student 的 log probability 差：

$$
\Delta_{i,t,k}
=
\log p_T(a_{i,t,k} \mid x, a_{i,<t})
-
\log p_S(a_{i,t,k} \mid x, a_{i,<t})
$$

注意：代码里使用的是 log probability 差，不是未归一化的 raw logits 差。

原始 OPD reward 可以写成：

$$
r^{teacher}_{i,t,k}
=
w_{i,t,k} \cdot \Delta_{i,t,k}
$$

其中 $w_{i,t,k}$ 是 top-k candidate 的权重。当前脚本默认：

```bash
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
OPD_TOPK_RENORMALIZE=False
```

也就是说候选 token 只取 student top-k，权重使用 student probability，并且不在 top-k 内重新归一化。

## Outcome-Aligned Mask

OAL-OPD 不改变原始 OPD reward 的数值形式，而是加一个 outcome-aligned mask。

如果 response 正确：

$$
M_{i,t,k}
=
\mathbb{1}
\left[
\Delta_{i,t,k} > \delta
\right]
$$

如果 response 错误：

$$
M_{i,t,k}
=
\mathbb{1}
\left[
\Delta_{i,t,k} < -\delta
\right]
$$

其中 $\delta$ 是 margin，默认是 `0.0`。

直观理解：

- 正确 response：只保留 teacher 比 student 更看好的 candidate
- 错误 response：只保留 teacher 比 student 更不看好的 candidate
- 与 outcome 方向不一致的 token candidate 不参与更新

## 最终 Advantage

最终 advantage 为：

$$
A_{i,t,k}
=
M_{i,t,k}
\cdot
r^{teacher}_{i,t,k}
$$

也就是：

$$
A_{i,t,k}
=
M_{i,t,k}
\cdot
w_{i,t,k}
\cdot
\Delta_{i,t,k}
$$

这里不把 outcome reward 广播到 token 上。Outcome 只用于决定哪些 token candidate 的 teacher dense reward 与最终结果方向一致。

## 与原始 OPD 的区别

原始 OPD：

$$
A_{i,t,k}
=
r^{teacher}_{i,t,k}
$$

OAL-OPD：

$$
A_{i,t,k}
=
M_{i,t,k}
\cdot
r^{teacher}_{i,t,k}
$$

区别在于：OAL-OPD 会过滤掉和最终 outcome 不一致的 teacher dense reward。

## 当前实现位置

入口脚本：

```bash
scripts/train/run_outcome_aligned_logit_opd.sh
```

核心 estimator：

```python
@register_adv_est("outcome_aligned_logit_opd")
def compute_outcome_aligned_logit_opd_advantage(...)
```

实现文件：

```text
verl/verl/trainer/ppo/core_algos_oal_opd.py
```

Actor：

```text
verl/verl/workers/actor/dp_actor_oal_opd.py
```

Trainer：

```text
verl/verl/trainer/ppo/ray_trainer_oal_opd.py
```

## 关键超参数

### `OAL_MARGIN`

控制保留 token candidate 所需的最小 log probability 差。

默认：

```bash
OAL_MARGIN=0.0
```

如果设大一点，只有 teacher 和 student 差异更明显的 candidate 会被保留。

## 日志指标

当前实现会记录：

```text
oal/kept_candidate_ratio
oal/positive_candidate_ratio
oal/negative_candidate_ratio
oal/token_keep_ratio
oal/correct_response_ratio
oal/outcome_score_mean
```

重点看：

- `oal/kept_candidate_ratio`：被保留的 top-k candidate 比例
- `oal/token_keep_ratio`：至少有一个 candidate 被保留的 token 比例
- `oal/correct_response_ratio`：当前 batch 的正确 response 比例
- `oal/outcome_score_mean`：最终 outcome reward 平均值

## 当前默认运行方式

```bash
bash scripts/train/run_outcome_aligned_logit_opd.sh
```

默认使用：

```bash
ADV_ESTIMATOR=outcome_aligned_logit_opd
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
OPD_TOPK_RENORMALIZE=False
OAL_MARGIN=0.0
```

# BOAL-OPD: Block Outcome-Aligned OPD

## 背景问题

OAL-OPD 在每个 token candidate 上独立判断 teacher dense reward 是否与 outcome 对齐。这种做法粒度很细，但也可能过于局部：一个 token 的 log probability 差值很噪，不能稳定反映一段推理是否可靠。

BOAL-OPD 使用固定长度 block 聚合局部信号。它不再逐 token 独立做 hard filter，而是判断每个 block 与 outcome 是否一致，并用前面 block 的错误累积来调节当前和后续 block 的监督强度。

## 核心思想

把每条 response 按固定长度切成 block：

$$
\mathcal{B}_j = [jW, (j+1)W)
$$

其中 $W$ 是 block size。

对每个 block，累加该 block 内的 teacher-student log probability 差：

$$
S_{i,j}
=
\sum_{t \in \mathcal{B}_j}
\sum_k
\Delta_{i,t,k}
$$

其中：

$$
\Delta_{i,t,k}
=
\log p_T(a_{i,t,k})
-
\log p_S(a_{i,t,k})
$$

直观理解：

- $S_{i,j} > 0$：teacher 整体比 student 更支持这个 block 中的候选 token
- $S_{i,j} < 0$：teacher 整体比 student 更不支持这个 block 中的候选 token

## Block 与 Outcome 的对齐判断

如果 response 正确：

$$
\text{aligned}_{i,j}
=
\mathbb{1}
\left[
S_{i,j} > \delta
\right]
$$

如果 response 错误：

$$
\text{aligned}_{i,j}
=
\mathbb{1}
\left[
S_{i,j} < -\delta
\right]
$$

其中 $\delta$ 是 block-level margin。

也就是说：

- 正确 response 中，teacher 应该整体支持该 block
- 错误 response 中，teacher 应该整体反对该 block

如果不满足这个关系，就认为该 block 与 outcome 不对齐。

## 累积 Bad Block 衰减

BOAL-OPD 的关键不是只看当前 block，而是看到目前为止有多少 block 已经与 outcome 不对齐。

定义：

$$
C_{i,j}
=
\sum_{l=0}^{j}
\mathbb{1}
\left[
\text{aligned}_{i,l}=0
\right]
$$

注意这里包含当前 block。如果当前 block 本身不对齐，那么它自己的权重也会立即降低。

block 权重为：

$$
g_{i,j}
=
\exp
\left(
-\lambda C_{i,j}
\right)
$$

其中 $\lambda$ 是衰减系数。

直观理解：

- 前面一路都对齐：$C_{i,j}=0$，权重为 1
- 当前 block 第一次不对齐：$C_{i,j}=1$，权重为 $\exp(-\lambda)$
- 不对齐 block 越多，后续 teacher dense reward 越不可信

## 最终 Advantage

BOAL-OPD 仍然保留 teacher dense reward，只是用 block 权重调节强度：

$$
A_{i,t,k}
=
g_{i,b(t)}
\cdot
r^{teacher}_{i,t,k}
$$

其中 $b(t)$ 表示 token `t` 所属的 block。

如果原始 OPD reward 为：

$$
r^{teacher}_{i,t,k}
=
w_{i,t,k}
\cdot
\Delta_{i,t,k}
$$

则 BOAL-OPD 为：

$$
A_{i,t,k}
=
g_{i,b(t)}
\cdot
w_{i,t,k}
\cdot
\Delta_{i,t,k}
$$

## 与 OAL-OPD 的区别

OAL-OPD 是 token candidate 级 hard filter：

$$
A_{i,t,k}
=
M_{i,t,k}
r^{teacher}_{i,t,k}
$$

BOAL-OPD 是 block 级 soft decay：

$$
A_{i,t,k}
=
g_{i,b(t)}
r^{teacher}_{i,t,k}
$$

OAL 更激进，直接把不符合方向的 candidate 置零。

BOAL 更平滑，用 block-level 的历史不对齐程度逐渐降低 teacher 监督。

## 当前实现位置

入口脚本：

```bash
scripts/train/run_block_outcome_aligned_opd.sh
```

核心 estimator：

```python
@register_adv_est("block_outcome_aligned_opd")
def compute_block_outcome_aligned_opd_advantage(...)
```

实现文件：

```text
verl/verl/trainer/ppo/core_algos_boal_opd.py
```

Actor：

```text
verl/verl/workers/actor/dp_actor_boal_opd.py
```

Trainer：

```text
verl/verl/trainer/ppo/ray_trainer_boal_opd.py
```

## 关键超参数

### `BOAL_BLOCK_SIZE`

block 长度。

默认：

```bash
BOAL_BLOCK_SIZE=512
```

block 越小，定位越细，但 block score 更噪。

block 越大，判断更稳，但信用分配更粗。

### `BOAL_ALIGN_MARGIN`

block 对齐判断的 margin。

默认：

```bash
BOAL_ALIGN_MARGIN=0.0
```

设为正数后，需要 block score 更明显地与 outcome 同向，才认为该 block 对齐。

### `BOAL_DECAY_LAMBDA`

bad block 累积衰减系数。

默认：

```bash
BOAL_DECAY_LAMBDA=0.25
```

权重公式：

$$
g = \exp(-\lambda C)
$$

如果 $\lambda$ 越大，遇到不对齐 block 后 teacher 监督下降越快。

## 日志指标

当前实现会记录：

```text
boal/block_weight_mean
boal/block_weight_min
boal/block_weight_max
boal/block_score_mean
boal/block_score_abs_mean
boal/aligned_token_ratio
boal/cumulative_bad_mean
boal/correct_response_ratio
boal/outcome_score_mean
```

重点看：

- `boal/block_weight_mean`：平均 teacher 监督保留强度
- `boal/aligned_token_ratio`：处在 outcome-aligned block 里的 token 比例
- `boal/cumulative_bad_mean`：平均累计 bad block 数
- `boal/block_score_mean`：block-level teacher-student 差值方向
- `boal/block_score_abs_mean`：block-level 信号强度

## 当前默认运行方式

```bash
bash scripts/train/run_block_outcome_aligned_opd.sh
```

默认使用：

```bash
ADV_ESTIMATOR=block_outcome_aligned_opd
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
OPD_TOPK_RENORMALIZE=False
BOAL_BLOCK_SIZE=512
BOAL_ALIGN_MARGIN=0.0
BOAL_DECAY_LAMBDA=0.25
```
