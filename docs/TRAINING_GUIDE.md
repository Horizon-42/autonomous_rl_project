# Duckietown + SB3 入门：从跑通训练到加「完成一圈/Checkpoint」奖励

这份文档以 [train_jetracer_centerline.py](train_jetracer_centerline.py) 为主线，带你从「能跑起来」→「能看曲线」→「能改奖励」→「能自己加 checkpoint / 完成一圈奖励」。

## 0. 你现在这份脚本在做什么

脚本整体流程：

1. 创建 Duckietown 环境（摄像头 RGB 图像观测）。
2. 叠加 3 个 wrapper：
   - `JetRacerWrapper`：把策略输出的动作 `[throttle, steering]` 映射成 DuckietownEnv 的 `(vel, angle)`。
   - `JetRacerRaceRewardWrapper`：把环境原始 reward 替换成「更像比赛」的 shaped reward（更鼓励跑得快 / 往前推进，同时不出界）。
   - `ResizeNormalizeObs`：把图像 resize 成 (3, 84, 84) 且归一化到 [0,1]，给 CNN 使用。
3. 用 SB3 的 PPO（`CnnPolicy`）开始训练。
4. 记录日志：
   - `VecMonitor` 写 `monitor.csv`（用于 episode 回报曲线）
   - SB3 logger 写 TensorBoard event（用于 TensorBoard 曲线）

你可以把它理解成：

> 「环境 + 奖励」决定你想让车学什么；「算法/网络」决定它怎么学。

---

## 1. 动作空间：JetRacer → Duckietown

### 1.1 JetRacer 动作（策略输出）

脚本里策略学的是：

- `throttle ∈ [0, 1]`：油门（只允许向前，这会明显加快早期收敛）
- `steering ∈ [-1, 1]`：转向

这由 `JetRacerWrapper.action_space` 定义。

### 1.2 DuckietownEnv 动作（环境真正执行）

DuckietownEnv 期望的动作是 `(vel, angle)`，范围也是 [-1, 1]。

脚本用下面映射：

- `vel = clip(throttle * v_scale, -1, 1)`
- `angle = clip(-steering * omega_scale, -1, 1)`

其中 `v_scale=0.4`、`omega_scale=1.2` 是你当前脚本的默认值。

> 注意：`omega_scale=1.2` 可能让 `angle` 超过 1，所以脚本里做了 clip。这样不会把环境 action 弄出界。

---

## 2. 观测：图像怎么喂给 PPO

`ResizeNormalizeObs` 做两件事：

- resize：从原始 HWC 的 RGB 图变成 84×84（更省算力）
- 变换维度：HWC → CHW（SB3 的 CNN 更习惯 CHW）
- 归一化：uint8 [0,255] → float32 [0,1]

同时在创建 PPO 时设置了：

- `policy_kwargs={"normalize_images": False}`

因为我们已经手动归一化了。

---

## 3. 奖励：为什么说它更像「赢比赛」

你当前用的是 `JetRacerRaceRewardWrapper`。

### 3.1 从环境拿到哪些“赛道信息”

Duckietown 在每一步的 `info` 里提供了 simulator 信息（脚本通过 `info["Simulator"]` 访问）。常用字段包括：

- `lane_position`：车道坐标系下的估计
  - `dist`：离中心线的横向距离（米）
  - `angle_rad`：车身朝向与车道切线的夹角（弧度）
  - `dot_dir`：大致可理解为“朝前程度”（1 最好，0 横着，<0 倒着）
- `robot_speed`：速度
- `proximity_penalty`：靠近障碍物/墙的惩罚信号（通常 ≤ 0）
- `msg`：done/异常的文本提示（例如包含 “invalid pose”）

### 3.2 目前 reward 的结构（直觉版）

奖励主要由这些项组成：

- **推进/速度（想赢比赛就得跑起来）**
  - `progress = max(0, dot_dir) * speed`
  - reward += `w_progress * progress` + `w_speed * speed`

- **跑偏/朝向（不走歪，速度才能上去）**
  - penalty += `w_center * abs(dist)`
  - penalty += `w_heading * abs(angle_rad)`

- **安全（离墙太近就扣）**
  - reward += `w_proximity * proximity_penalty`

- **抑制蛇形（让轨迹更稳）**
  - penalty += `w_steer * steering^2`
  - penalty += `w_steer_rate * (steering - prev_steering)^2`

- **出界/翻车（比赛直接失败）**
  - 如果 done 且是 invalid-pose：额外 -`offroad_penalty`

这类 reward 的核心是：

> 不只是“贴着中心线”，而是“在赛道上尽快向前推进”。

---

## 4. 如何跑训练

在环境里运行（例）：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate duckie_clean

python train_jetracer_centerline.py \
  --map-name loop_empty \
  --total-timesteps 200000 \
  --n-envs 1
```

如果你想看到渲染窗口（更慢）：

```bash
python train_jetracer_centerline.py --map-name loop_empty --total-timesteps 200000 --render --n-envs 1
```

---

## 5. 如何看 reward 曲线（最重要的入门反馈）

### 5.1 先确认日志文件有产出

默认日志目录是：

- `runs/centerline/`

你至少应该能看到：

- `monitor.csv`（episode 回报/长度）
- `events.out.tfevents...`（TensorBoard）

### 5.2 TensorBoard

启动：

```bash
python -m tensorboard.main --logdir runs/centerline --port 6007 --bind_all
```

浏览器打开：

- `http://<server-ip>:6007/`

重点看：

- `rollout/ep_rew_mean`

> 如果你遇到 `MessageToJson()` 的 protobuf 报错，按 [README.md](README.md) 里的方法 pin `protobuf==3.20.3`。

---

## 6. 如何把已训练的模型“重新跑起来”（enjoy / evaluation）

训练会保存一个 SB3 的模型文件（默认是 `models/centerline_ppo.zip`）。要在环境里重新跑起来（评估/看效果），你需要：

1. 重新创建和训练时一致的 env + wrappers
2. `PPO.load()` 加载 `.zip`
3. 循环 `predict()` → `env.step()`（可选 `render()`）

我已经提供了一个最小脚本：

- `enjoy_jetracer_centerline.py`

### 6.1 Headless 跑（服务器推荐）

无窗口运行、只打印每个 episode 的长度/回报：

```bash
python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip --episodes 3
```

### 6.2 带窗口渲染跑（需要显示器/GL）

```bash
python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip --render --episodes 3
```

### 6.3 加键盘控制（需要 `--render`）

开启键盘控制：

```bash
python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip --render --keyboard
```

按键说明（尽量做得最小但够用）：

- `M`：切换 `model` / `manual`
- 手动模式 `manual` 下：方向键控制油门/转向（增量式）
- `SPACE`：油门/转向归零
- `R`：reset 当前 episode
- `Q` 或 `ESC`：退出

如果在服务器上没法弹窗口，可以用 headless 模式，或者用你本地机器跑（同一个模型文件复制过去即可）。

### 6.4 Deterministic vs 非 deterministic

- 默认（非 deterministic）会带一点随机性，更像训练时的策略。
- 想更稳定地评估当前策略：

```bash
python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip --deterministic --episodes 3
```

---

## 7. SB3 训练日志怎么读：`ep_len_mean` / `ep_rew_mean`

你在训练输出里看到的这段：

```
| rollout/        |          |
|    ep_len_mean  |  ...     |
|    ep_rew_mean  |  ...     |
```

是 SB3（Stable-Baselines3）对**最近一段时间内采样到的 episode**做的统计。

### 7.1 `ep_len_mean` 是什么？

- 全称：episode length mean
- 含义：最近若干个 episode 的**平均长度**（平均多少 step 结束）
- 单位：**steps（环境步数）**

在 Duckietown 里，episode 常见结束原因：

- **invalid pose / off-road**：跑出可行驶区域（最常见）
- **max_steps reached**：走满环境的 `max_steps`（你脚本里环境设置是 `max_steps=1000`）

所以一般来说：

- `ep_len_mean` 变大：车更不容易出界/撞墙，能“活更久”
- 但注意：也可能是车学会了“慢慢挪着不死”，所以要结合 `ep_rew_mean` 看

### 7.2 `ep_rew_mean` 是什么？

- 全称：episode reward mean（也叫 episode return mean）
- 含义：最近若干个 episode 的**平均总回报**

每个 episode 的总回报是把每一步的 reward 累加：

$$R_{episode} = \sum_{t=0}^{T-1} r_t$$

在你的脚本里，`r_t` 是 `JetRacerRaceRewardWrapper` 计算出来的 race reward（不是 Duckietown 默认 reward）。

### 7.3 这两个数怎么结合起来判断“有没有进步”

常见几种组合：

- `ep_len_mean ↑` 且 `ep_rew_mean ↑`：最理想，车更稳也更符合目标（跑得更快/推进更多）
- `ep_len_mean ↑` 但 `ep_rew_mean ↓`（或不涨）：可能在“苟活”——速度很慢但不出界
- `ep_rew_mean ↑` 但 `ep_len_mean ↓`：可能变得更激进更快，但更容易冲出界；通常需要加大稳定性/出界惩罚

### 7.4 为什么 `ep_rew_mean` 一开始可能是负的？

这不一定是 bug，原因通常是：

- 初期策略随机，`abs(dist)`、`abs(angle_rad)`、以及 steering penalty 会很大
- 经常 invalid-pose，会吃到 `offroad_penalty`

随着策略变好：

- 车更能保持在路上（`ep_len_mean` 上升）
- `dot_dir * speed` 的“推进/速度项”开始占主导（`ep_rew_mean` 上升）

### 7.5 这些指标数据从哪来？

你脚本里使用了：

- `VecMonitor(..., filename=runs/centerline/monitor.csv)`

它负责记录每个 episode 的 return 和 length，并让 SB3 能计算并输出 `ep_len_mean` / `ep_rew_mean`。

如果你想更细地分析，也可以直接查看：

- `runs/centerline/monitor.csv`

### 7.6 `time/*` 这些字段什么意思？

你看到的：

- `time/fps`
- `time/iterations`
- `time/time_elapsed`
- `time/total_timesteps`

主要是**训练进度与速度**，不直接代表“学得好不好”，但能帮你判断训练是否正常在跑。

- `time/fps`
  - 含义：每秒钟完成多少个环境 step（steps/s）。
  - 解读：越高训练越快；Duckietown 渲染/CPU 负载会显著影响它。
  - 常见现象：打开 `render()`、提高分辨率、或 n_envs 增大，fps 往往下降。

- `time/iterations`
  - 含义：SB3 的训练迭代次数（对 PPO 来说，一般每次迭代会采样 `n_steps * n_envs` 个 step，然后做若干轮优化）。
  - 解读：只是计数器，用来对应日志频率。

- `time/time_elapsed`
  - 含义：从训练开始到现在经过的秒数（wall-clock）。
  - 解读：配合 `fps`、`total_timesteps` 看训练吞吐是否正常。

- `time/total_timesteps`
  - 含义：到目前为止累计与环境交互的 step 总数。
  - 解读：这才是“训练预算”的核心尺度（比如 1e6 steps）。

### 7.7 `train/*` 这些字段什么意思？（PPO 常见诊断指标）

你看到的：

- `train/approx_kl`
- `train/clip_fraction`
- `train/clip_range`
- `train/entropy_loss`
- `train/explained_variance`
- `train/learning_rate`
- `train/loss`
- `train/n_updates`
- `train/policy_gradient_loss`
- `train/std`
- `train/value_loss`

它们大多是 PPO 的优化过程统计，用来判断更新是否“太猛/太弱”、价值函数是否在拟合，以及策略是否过早变得确定。

#### 7.7.1 `approx_kl`（近似 KL 散度）

- 含义：新旧策略分布的差异大小（PPO 用它监控每次更新是否偏离太多）。
- 解读：
  - 太大：更新步子太大，可能不稳定（PPO 可能会 early stop 某些 epoch，或你会看到性能抖动）。
  - 太小：更新很保守，学习可能慢。
- 经验上：常见在 0.01 左右波动是比较正常的区间之一，但和任务、网络、归一化、奖励尺度都强相关。

#### 7.7.2 `clip_fraction` / `clip_range`

- `clip_range`
  - 含义：PPO 的 clip 超参数（比如 0.2），控制概率比率 $r_t(\theta)$ 被裁剪的范围。
- `clip_fraction`
  - 含义：本次更新中，有多少比例的样本触发了 clip（被裁剪了）。
- 解读：
  - `clip_fraction` 很高：说明更新压力很大，很多样本都在被裁剪；可能需要更小 learning rate、或奖励尺度/优势过大。
  - `clip_fraction` 接近 0：更新很温和；可能学习偏慢。

#### 7.7.3 `entropy_loss`

- 含义：策略熵项的 loss（在 SB3 里通常是负数）。它对应“探索度”的正则项。
- 解读：
  - 熵更高（更随机）通常意味着探索更多；熵逐渐降低表示策略更确定。
  - 如果熵很早就降得很低，策略可能过早收敛到某个坏的习惯动作（比如一直慢速直走/一直打方向）。

#### 7.7.4 `explained_variance`（价值函数解释方差）

- 含义：价值网络对回报的拟合程度指标，范围一般在 $(-\infty, 1]$，越接近 1 越好。
- 解读：
  - 接近 1：价值函数拟合得不错（不代表策略一定好，但训练通常更稳定）。
  - 接近 0 或负：价值函数几乎没学到或在发散；可能 reward 太噪/尺度不合适/学习率太大。

#### 7.7.5 `policy_gradient_loss`

- 含义：策略梯度项（actor）对应的 loss。
- 解读：绝对值不太好直接横向比较，更适合看是否突然爆炸、或长期几乎不变（卡住）。

#### 7.7.6 `value_loss`

- 含义：价值函数（critic）的回归误差。
- 解读：
  - 偏大：可能回报尺度太大/太噪，或者 critic 学不过来。
  - 偏小：critic 拟合不错；但仍要看 `explained_variance` 是否高。

#### 7.7.7 `loss`

- 含义：训练总 loss（把 policy loss、value loss、entropy loss 等组合起来）。
- 解读：它不是直接的“性能指标”；更重要的还是 `rollout/ep_rew_mean`、成功率、是否完成一圈等任务指标。

#### 7.7.8 `std`

- 含义：策略分布的标准差（对连续动作策略很常见）。
- 解读：
  - `std` 大：动作更随机（探索更多）。
  - `std` 小：动作更确定（探索更少）。
  - 如果 `std` 过早变得很小，可能探索不够；可以考虑提高熵系数（如果你改了默认值）或调整奖励让探索更有梯度。

#### 7.7.9 `learning_rate` / `n_updates`

- `learning_rate`
  - 含义：当前优化学习率（如果使用 schedule 可能会变化；你现在多半是常数）。
- `n_updates`
  - 含义：到目前为止做过多少次梯度更新（计数）。

---

小提醒：这些 `train/*` 指标更像“发动机仪表盘”。判断训练好不好，优先看：

1. `rollout/ep_rew_mean` 是否上升
2. `rollout/ep_len_mean` 是否上升（不再频繁 invalid pose）
3. 你定义的比赛目标：是否更快、更稳定、更少出界、最终能否完成一圈

---

## 8. 你如何开始“自己改奖励”（实践路线）

建议按这个顺序（每一步只改一点，容易定位问题）：

1. **先让车跑起来**：提高 `w_speed` 或 `w_progress`
2. **再约束不出界**：提高 `offroad_penalty`，或增大 `w_center/w_heading`
3. **最后减少蛇形**：提高 `w_steer` / `w_steer_rate`

训练里你会看到一个典型现象：

- `ep_rew_mean` 一开始很负（到处撞/出界）
- 能走直后会逐渐变好
- 如果速度起来后又开始乱甩，说明需要更强的稳定性惩罚

---

## 9. 进阶：加 checkpoint 奖励（推荐先做这个）

“完成一圈”通常需要你能可靠判断是否绕回起点；对新手来说 **checkpoint 更容易**。

### 9.1 你需要什么信息

你可以从 `info["Simulator"]` 里拿到：

- `tile_coords`: 当前车所在的 tile 坐标（整数格）
- `cur_pos`: 连续坐标（米）

`tile_coords` 很适合作为 checkpoint 的离散状态。

### 9.2 最小实现思路

在 `JetRacerRaceRewardWrapper` 里加两个状态：

- `self._next_cp_idx`
- `self._checkpoints: list[tuple[int,int]]`

每一步检查：

- 当前 tile 是否等于 `checkpoints[next_cp_idx]`
- 如果是：给奖励 `+cp_reward`，然后 `next_cp_idx += 1`

### 9.3 建议的“第一次实现”（伪代码）

把下面逻辑放在 [train_jetracer_centerline.py](train_jetracer_centerline.py) 的 `JetRacerRaceRewardWrapper.step()` 里（在构造 `info["race_reward"]` 之前）：

```python
# 读取 tile
sim = info.get("Simulator", {})
tile = sim.get("tile_coords")  # 形如 [i, j]

# 判定 checkpoint
if isinstance(tile, (list, tuple)) and len(tile) == 2:
    cur_tile = (int(tile[0]), int(tile[1]))
    if cur_tile == self._checkpoints[self._next_cp_idx]:
        shaped_reward += self._cp_reward
        self._next_cp_idx = (self._next_cp_idx + 1) % len(self._checkpoints)
```

### 9.4 怎么选 checkpoint

最简单：

1. 先用当前脚本跑起来（哪怕策略很烂）
2. 观察打印/调试（你也可以临时 `print(info["Simulator"]["tile_coords"])`）
3. 手动挑 4~8 个均匀分布的 tile 作为 checkpoint

---

## 10. 再进阶：完成一圈奖励（在 checkpoint 之后做）

完成一圈最稳的做法：

- 定义一组 checkpoint 构成“路线”
- 当你按顺序通过所有 checkpoint，再回到起点 checkpoint，给 “lap 完成奖励”

### 10.1 最小实现思路

状态机：

- `next_cp_idx` 从 0 开始
- 依次命中 checkpoint 0,1,2,...,N-1
- 当 `next_cp_idx` 从 N-1 → 0 的那一刻，视为“完成一圈”，给 `lap_reward`

伪代码（接在 checkpoint 逻辑里）：

```python
if hit_checkpoint:
    if self._next_cp_idx == len(self._checkpoints) - 1:
        shaped_reward += self._lap_reward
        self._lap_count += 1
    self._next_cp_idx = (self._next_cp_idx + 1) % len(self._checkpoints)
```

### 10.2 常见坑

- **抖动重复触发**：车在 checkpoint tile 上停留多步，会反复给奖励。
  - 简单修复：加一个 `self._last_tile`，只在 tile 变化时允许触发。
- **逆行/跳关**：如果车倒着走也能碰到 checkpoint。
  - 简单修复：只在 `dot_dir > 0` 或 `speed > 某阈值` 时才算命中。

---

## 11. 建议你接下来怎么做（循序渐进）

1. 先跑 200k~1M steps，确保能在 `loop_empty` 上稳定不出界
2. 加 4~8 个 checkpoint 奖励，让策略更像“跑圈”而不是“原地抖动”
3. checkpoint 稳了再加 lap 完成奖励
4. 再考虑更强的“比赛目标”：例如超车/避障/计时（这会需要更明确的任务定义）

如果你愿意，我也可以按你选定的 map（例如 `loop_empty`）帮你挑一组 checkpoint tile 坐标，并把 checkpoint/lap 逻辑直接落到脚本里（保持改动最小）。
