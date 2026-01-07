# Duckietown RL 环境 Bug 报告：`undefined symbol: iJIT_NotifyEvent` 及训练链路修复

日期：2026-01-07  
环境：Linux + conda 环境 `duckie_clean`（Python 3.8）  
项目：Duckietown `gym-duckietown` + Stable-Baselines3 PPO

---

## 1. 问题概述（Bug 现象）

在 `duckie_clean` 环境中运行训练脚本时（以及任何 `import torch` 的场景），出现以下典型问题：

1) **PyTorch 无法导入**

- 报错形态：
  - `ImportError: .../torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`
  - 同时还存在其他缺失符号：`iJIT_IsProfilingActive`、`iJIT_GetNewMethodID`

2) **训练脚本运行时依赖缺失/不兼容（次要但会阻塞运行）**

- `networkx` 缺失核心类型：`ImportError: cannot import name 'MultiDiGraph' from 'networkx'`
- `PyYAML` 缺失：`ModuleNotFoundError: No module named 'yaml'`
- SB3 日志系统强依赖 TensorBoard：`AssertionError: tensorboard is not installed...`
- SB3 `CnnPolicy` 对图像观测空间断言失败：
  - `AssertionError: You should use NatureCNN only with images not with Box(... float32)`

以上问题中，**最核心阻塞点**是第 1 条（`import torch` 直接失败），导致 SB3/训练脚本无法启动。

---

## 2. 根因分析：为什么会出现 `iJIT_NotifyEvent` 未定义符号

### 2.1 这是“动态链接”层面的错误，而不是 Python 代码逻辑错误

- `torch` 的 Python 包会加载其 C++/CPU 核心共享库（例如 `libtorch_cpu.so`）。
- 当执行 `import torch` 时，操作系统的动态链接器会尝试解析该 `.so` 依赖的**外部符号**。
- 一旦发现某个符号在当前进程可见的动态库集合中**不存在**，加载会立刻失败，导致 `ImportError`。

### 2.2 `libtorch_cpu.so` 依赖了 Intel iJIT/ITT profiling 符号

通过对 `libtorch_cpu.so` 的符号表检查，可看到它有以下**未定义引用**（`U`）：

- `iJIT_GetNewMethodID`
- `iJIT_IsProfilingActive`
- `iJIT_NotifyEvent`

这类 `iJIT_*` 符号通常来自 Intel 的 JIT/VTune/ITT profiling 运行时库（常见名字例如 `libittnotify.so`）。

### 2.3 为什么环境里找不到提供这些符号的库

- 在该 conda 环境中，即使安装了某些 `ittapi` 类包，也可能只提供**静态库**（例如 `libittnotify.a`），而**不提供可被运行时动态加载的共享库**（例如 `libittnotify.so`）。
- 动态链接器只会在运行时加载的共享库中找符号；静态库无法在 `import torch` 时被“自动”链接进来。

因此结果是：

> `libtorch_cpu.so` 在加载时需要解析 `iJIT_*`，但当前环境与系统中没有任何 `.so` 导出这些符号，于是 `import torch` 失败。

---

## 3. 修复策略与实施：我做了什么

修复目标：

- 让 `import torch` 在 `duckie_clean` 中稳定成功
- 保持 Duckietown 依赖约束（尤其是 `numpy<=1.20` / `gym==0.21`）不被破坏
- 让训练脚本能至少完成一次短跑（smoke test）并保存模型

下面按问题分组说明修复动作。

---

### 3.1 修复 PyTorch 导入失败（核心）

#### 3.1.1 诊断确认（定位到缺失符号）

我做的检查要点：

- 明确失败库：`.../site-packages/torch/lib/libtorch_cpu.so`
- 检查其未定义符号：确认确实存在 `U iJIT_NotifyEvent` 等
- 搜索 conda 环境的 `$CONDA_PREFIX/lib`：确认没有任何共享库导出 `iJIT_*`

结论：这是缺失运行时符号导致的动态链接失败。

#### 3.1.2 尝试安装 `ittapi` 的结果

- 找到了 conda-forge 上的 `ittapi` 包。
- 安装后发现其主要产物是 `libittnotify.a`（静态库），而不是能让动态链接器直接解析符号的 `.so`。

因此，这条路线无法直接修复 `import torch`。

#### 3.1.3 最终修复方案：提供一个 stub 共享库 + 自动 `LD_PRELOAD`

核心思路：

- 对 PyTorch 而言，这些 `iJIT_*` 更像是“可选 profiling 功能”的接口。
- 我们只需要让动态链接避免失败，即提供同名符号即可；实现可以是 no-op。

具体动作：

1) 用 C 写一个极小的动态库（stub），导出以下符号：

- `iJIT_GetNewMethodID()`：返回递增 id
- `iJIT_IsProfilingActive()`：返回 0（不开启 profiling）
- `iJIT_NotifyEvent(int, void*)`：no-op，返回 0

2) 编译为共享库并放到 conda 环境：

- 生成 `libjitprofiling_stub.so` 并安装到：`$CONDA_PREFIX/lib/libjitprofiling_stub.so`

3) 使用 conda 激活脚本自动设置 `LD_PRELOAD`：

- 写入：`$CONDA_PREFIX/etc/conda/activate.d/jitprofiling_stub.sh`
- 写入：`$CONDA_PREFIX/etc/conda/deactivate.d/jitprofiling_stub.sh`

效果：

- 每次 `conda activate duckie_clean` 时，都会自动将 stub 加入 `LD_PRELOAD`，从而在加载 `libtorch_cpu.so` 前提供所需符号。
- 验证：`import torch` 成功。

> 这属于“运行时补丁”方式：不改 torch 二进制、不改系统库，仅在当前 conda env 内注入缺失符号。

---

### 3.2 修复训练链路的其他阻塞点（让脚本可以跑通）

#### 3.2.1 `networkx` 异常：导入到的是损坏的 namespace package

现象：

- `import networkx` 得到的是 `<module 'networkx' (namespace)>`，`__file__`/`__version__` 为空
- 导致 `MultiDiGraph` 等核心类不存在

处理：

- 安装一个正常发行版 networkx（固定到 `2.6.3`，并在 setup 脚本层面约束为 `networkx<3`）
- 验证 `hasattr(nx, 'MultiDiGraph') == True`

#### 3.2.2 `yaml` 缺失：Duckietown-world 依赖 PyYAML

处理：

- 安装 `pyyaml`
- 验证 `import yaml` 成功

#### 3.2.3 TensorBoard 不是必需项：让脚本不因缺 tensorboard 直接退出

原问题：

- SB3 logger 配置了 `format_strings=["stdout", "tensorboard"]`
- SB3 会在未安装 TensorBoard 时触发断言失败

处理（修改训练脚本）：

- 默认只启用 stdout
- 若 `from torch.utils.tensorboard import SummaryWriter` 可导入，则额外启用 tensorboard
- 否则打印提示并继续

#### 3.2.4 `CnnPolicy` 图像空间断言失败：观测为 CHW float32 已归一化

原问题：

- 观测 wrapper 输出 `float32` 且范围 `[0,1]`（CHW）
- SB3 的 `NatureCNN`/`is_image_space` 检查默认假设图像像素可能是 uint8 或需要 SB3 归一化

处理（修改训练脚本）：

- 在 PPO 初始化中加入：`policy_kwargs={"normalize_images": False}`
- 明确告诉 SB3：图像已归一化，不要再做默认归一化假设/处理

---

## 4. 代码与脚本层面的落地变更

### 4.1 环境脚本

- 文件：`env_setup.sh`
- 变更内容（关键点）：
  - 安装 Duckietown-world 运行时依赖：`pyyaml`、`networkx<3`
  - 安装 torch 后，构建并安装 `libjitprofiling_stub.so`
  - 通过 conda `activate.d`/`deactivate.d` 自动设置/恢复 `LD_PRELOAD`

### 4.2 训练脚本

- 文件：`train_jetracer_centerline.py`
- 变更内容（关键点）：
  - TensorBoard 变为可选：没有 tensorboard 也能训练（stdout logging）
  - 增加 `policy_kwargs={"normalize_images": False}` 以适配 CHW float32 图像输入

---

## 5. 验证结果（修复是否生效）

验证项：

1) `duckie_clean` 内：

- `import torch` 成功
- `import stable_baselines3` 成功
- `import yaml` 成功
- `import networkx` 且 `MultiDiGraph` 存在
- `pip check` 无 broken requirements

2) 训练脚本 smoke test：

- 以内置地图 `loop_empty` 运行短训练
- 能启动 PPO、完成一次迭代并保存模型（例如 `models/centerline_ppo.zip`）

---

## 6. 风险与注意事项

1) **关于 `LD_PRELOAD` 的副作用**

- 本方案只在 `duckie_clean` 环境激活时生效；退出环境会恢复原值。
- stub 仅提供 `iJIT_*`，通常不会影响 torch 的数值计算路径。

2) **避免再次安装 `stable-baselines3[extra] shimmy`**

- 该组合会引入 `gymnasium/shimmy` 路线，极容易把 `numpy/gym` 升级，破坏 Duckietown 对 `numpy<=1.20`、`gym==0.21` 的硬约束。

3) **地图名（map-name）资源问题**

- 例如 `loop_only_duckies` 并不是 Duckietown-world 内置地图资源，因此会报找不到 `loop_only_duckies.yaml`。
- 建议使用内置地图（如 `loop_empty`、`small_loop` 等）或把自定义地图资源正确安装到 Duckietown-world 的资源路径中。

---

## 7. 结论

- `iJIT_NotifyEvent` bug 的本质是：PyTorch 共享库在加载时需要解析 `iJIT_*` 符号，但环境内缺少提供这些符号的共享库，导致动态链接失败。
- 修复采用了最小侵入方案：在 conda env 内注入 stub `.so` 并自动 `LD_PRELOAD`，从而让 `import torch` 恢复正常。
- 同时补齐训练链路所需依赖（PyYAML、有效的 networkx），并增强训练脚本对 TensorBoard 和图像归一化假设的兼容性，最终使训练脚本可以跑通。
