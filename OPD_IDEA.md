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
