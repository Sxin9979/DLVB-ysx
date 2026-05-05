# HANDOFF

## 1. 本轮任务目标与当前进展

### 总体目标
本轮的核心目标是：为 `E3VB` 项目搭建“**多数据集统一训练**”的第一版基础设施，服务于后续将 `checked_outfile` 和 `vboutfile` 等不同分布的数据集放到**同一个模型**里训练，而不是各训各的。

用户当前的建模目标可以概括为：

- 前 `98%` 权重占比的结构要预测准；
- 这些高权重结构之间的排序要对；
- 不要把太多尾部近零结构误判为重点结构；
- 推理时给若干结构，模型能输出这些结构的相对权重。

前期分析结论是：

- `checked` 和 `vb` 的数据分布、每个 molecule 的结构数、尾部强度不同；
- 不能简单把所有 `.xmo` 丢进一个目录然后全局随机 split；
- 正确方向是：
  - 各 dataset 先独立 split；
  - 再把同名 split 合并；
  - 训练时做 dataset-aware 的采样；
  - 验证时做 per-dataset 统计与 macro checkpoint；
  - 后续再做 molecule-local loss。

### 当前进展
本轮**已经完成**的内容：

1. 支持多数据集配置入口：
   - 旧配置仍支持 `data.xmo_dir`
   - 新配置支持 `data.datasets: [...]`

2. 支持“每个 dataset 各自 split，再合并为统一 cache”：
   - `ProcessedDatasetCache`
   - `PackedDatasetCache`
   - split metadata
   - structure counts

3. `dataset_id` 已贯穿到缓存、chunk、batch、metrics：
   - molecule / chunk / packed chunk / packed ref
   - `UnifiedBatch.dataset_index_per_molecule`

4. 训练时支持 dataset-balanced molecule batching：
   - 在 molecule-group 级别按 dataset 交错出 batch

5. 增加 `focus_plus_mixed_tail` 采样：
   - 支持 hardest tail + random tail 混合

6. 验证支持 per-dataset metrics 和 macro metrics：
   - pooled summary 仍保留
   - 可以生成 `macro_*`
   - `focusPriorityScore()` 已加入 `precision` 惩罚

7. 新增 unified config 模板：
   - `config_e2e_checked_vb_unified_topmass.yaml`

### 当前尚未完成
本轮**没有继续实现**以下内容：

- `utils/losses.py` 中真正的 **molecule-local loss 平均** 改造
  - 即：
    - `focus_regression_loss`
    - `focus_rank_loss`
    - `focus_tail_rank_loss`
    - 先按 molecule 算，再对 molecule 平均
- 真实 preprocess / train runtime smoke test
  - 当前 shell Python 环境缺少 `numpy` / `yaml`，无法完成运行时验证
- 子集归一化辅助 loss（subset normalized KL/CE）
- `active_only_low_risk` 的结构消融实验


## 2. 已修改 / 新增 / 检查过的文件

### 已修改

- `/pool1/home/ysxin/E3nn/E3VB/data/schema.py`
- `/pool1/home/ysxin/E3nn/E3VB/data/__init__.py`
- `/pool1/home/ysxin/E3nn/E3VB/data/processor.py`
- `/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py`
- `/pool1/home/ysxin/E3nn/E3VB/utils/config.py`
- `/pool1/home/ysxin/E3nn/E3VB/utils/metrics.py`
- `/pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py`
- `/pool1/home/ysxin/E3nn/E3VB/main.py`
- `/pool1/home/ysxin/E3nn/E3VB/e3_o3_property_test.py`

### 已新增

- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/HANDOFF.md`

### 已检查但未修改的关键文件

- `/pool1/home/ysxin/E3nn/E3VB/data/xmo_builder.py`
- `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`
- `/pool1/home/ysxin/E3nn/E3VB/predict.py`
- `/pool1/home/ysxin/E3nn/E3VB/model/end_to_end.py`
- `/pool1/home/ysxin/E3nn/E3VB/model/rumer_encoder.py`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_outfile_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass.yaml`


## 3. 每个文件的关键改动与原因

### `data/schema.py`

#### 关键改动

- 新增 `DatasetSourceConfig`
- 在以下对象上增加 `dataset_id`
  - `UnifiedSample`
  - `ProcessedMolecule`
  - `ProcessedMoleculeChunk`
  - `PackedMoleculeChunk`
  - `PackedChunkReference`
- 在 `ProcessedDatasetCache` 中增加：
  - `dataset_ids`
  - `split_ids_by_dataset`
- 在 `PackedDatasetCache` 中增加：
  - `dataset_ids`
  - `split_ids_by_dataset`
  - `split_structure_counts_by_dataset`
- 在 `UnifiedBatch` 中增加：
  - `dataset_index_per_molecule`
- 更新 `UnifiedBatch.tree_flatten()` / `tree_unflatten()`

#### 为什么这样改

统一训练要做：

- dataset-balanced sampling
- per-dataset validation
- macro checkpoint

所以 batch 必须知道每个 molecule 属于哪个 dataset；cache 也必须保留 split 的 dataset 归属信息。


### `data/__init__.py`

#### 关键改动

- 导出 `DatasetSourceConfig`

#### 为什么这样改

保持数据层对外 API 完整，避免其他模块引用时找不到新 schema。


### `data/processor.py`

#### 关键改动

- `UnifiedDataConfig`：
  - `xmo_dir: str | None`
  - 新增 `datasets: Tuple[DatasetSourceConfig, ...] | None`
- 新增：
  - `dataSources()`
  - `usesMultipleDatasets()`
  - `cacheMoleculeId()`
- `discoverXmoFiles()` 改为支持传入具体 `xmo_dir`
- `createMoleculeSplits()` 支持：
  - `split` override
  - `seed_offset`
- `buildProcessedMolecule()` 改为接收 `dataset_id`
  - 多数据集模式下 `molecule_id` 会变成 `"{dataset_id}::{orig_molecule_id}"`
- `buildProcessedDatasetCache()` 重写为：
  - 遍历 `dataSources()`
  - 每个 dataset 单独 parse/build
  - 每个 dataset 单独 split
  - 最终合并为统一 `ProcessedDatasetCache`
- `materializeUnifiedSamples()` 给 `UnifiedSample` 填 `dataset_id`
- `chunkProcessedMolecule()` 给 `ProcessedMoleculeChunk` 填 `dataset_id`
- `buildPackedDatasetCache()`：
  - 补全 `dataset_ids`
  - 计算 `split_structure_counts_by_dataset`
  - 透传 `split_ids_by_dataset`
- `savePackedChunk()`：
  - `PackedChunkReference` 写入 `dataset_id`

#### 为什么这样改

这是整个 unified training 的**根骨架**：

- 旧逻辑是单 `xmo_dir`、全局 split；
- 新逻辑变成多 dataset，各自 split，再统一训练；
- 多数据集时必须避免 molecule id 冲突；
- cache 必须记录每个 split 内各 dataset 的 molecule / structure 归属。


### `data/grain_pipeline.py`

#### 关键改动

- `GraphPackingAdapter`：
  - 新增 `chunkDatasetId()`
  - `subsetPackedChunk()` 保留 `dataset_id`
  - `packChunk()` 保留 `dataset_id`
  - `packBatch()` 支持 `dataset_index_per_molecule`
  - `packPackedChunkBatch()` 支持 `dataset_index_per_molecule`
- `GrainPipeline.__init__()`：
  - 新增 `dataset_ids`
  - 新增 `dataset_index_by_id`
- 新增：
  - `chunkGroupDatasetId()`
  - `buildDatasetBalancedGroupedBatchIndices()`
- `createIterator()`：
  - chunk 级 batch 现在也会写 `dataset_index_per_molecule`
- `createMoleculeIterator()`：
  - 新增参数：
    - `mixed_tail_top_fraction`
    - `dataset_sampling_strategy`
  - 当 `dataset_sampling_strategy == "balanced"` 时，使用 dataset-balanced molecule batching
  - 调 `selectTopMassStructureIndices()` 时传 `mixed_top_fraction`
  - pack batch 时传每个 molecule 的 `dataset_index_per_molecule`

#### 为什么这样改

统一训练真正容易“被大数据集带偏”的地方不是 target scale，而是**batch 构造**：

- `checked` 的 molecule 更多 / 更大 / 更长尾；
- 如果自然顺序或自然频率采样，它会主导梯度；

所以要在 molecule-group 层做 dataset-balanced batching。


### `utils/config.py`

#### 关键改动

- `createDataConfig()`：
  - 支持 `data.datasets`
  - 保留老的 `data.xmo_dir`
- `TrainingConfig` 新增字段：
  - `focus_monitor_precision_weight`
  - `mixed_tail_top_fraction`
  - `dataset_sampling_strategy`
- `validation_monitor` 允许：
  - `macro_mae`
  - `macro_focus_weighted_mae`
  - `macro_focus_priority_score`
- 校验 `dataset_sampling_strategy`：
  - `natural`
  - `balanced`

#### 为什么这样改

配置层必须允许：

- 多数据集入口
- macro checkpoint
- precision-aware monitor
- mixed tail
- dataset-balanced sampling

否则新数据流无法从 YAML 驱动。


### `utils/metrics.py`

#### 关键改动

- 新增 `splitPayloadByDataset()`
  - 按 `num_structures_per_molecule` 和 `dataset_index_per_molecule` 把 pooled split payload 切成 per-dataset payload
- `summarize()` 新增支持：
  - `dataset_index_per_molecule`
  - `dataset_names`
- 当提供 dataset 信息时：
  - 计算 `per_dataset`
  - 计算所有数值型指标的 `macro_*`

#### 为什么这样改

统一训练时如果只看 pooled validation：

- 大数据集分数更容易主导；
- 小数据集退化可能完全被掩盖；

所以需要 per-dataset summary 和 macro summary。


### `utils/top_mass.py`

#### 关键改动

- `selectTopMassStructureIndices()` 新增参数：
  - `mixed_top_fraction`
- 新增策略：
  - `focus_plus_mixed_tail`
- 逻辑：
  - 一部分 tail 取 hardest top-tail
  - 一部分 tail 取 random tail

#### 为什么这样改

之前分析结论是：

- `checked` 需要更多 hardest tail negatives；
- 但统一训练时不能直接全局上 `focus_plus_top_tail`，会更偏 `checked`；

所以第一版统一训练用折中策略：

- `focus_plus_mixed_tail`
- `mixed_tail_top_fraction=0.5`


### `main.py`

#### 关键改动

- 新增辅助函数：
  - `validDatasetIndices()`
- `ExperimentRunner.__init__()` 新增：
  - `dataset_ids`
  - `split_molecule_ids_by_dataset`
  - `split_structure_counts_by_dataset`
- `buildDataset()`：
  - 从 `PackedDatasetCache` 读取 `dataset_ids`
  - 读取 `split_ids_by_dataset`
  - 读取 `split_structure_counts_by_dataset`
  - 初始化 `GrainPipeline(..., dataset_ids=...)`
  - 在 `sample_limit_per_split` 下同步裁剪 dataset-level split metadata
- `focusPriorityScore()`：
  - 增加 `precision_penalty = 1 - focus_precision`
  - 使用 `focus_monitor_precision_weight`
- `runTrainBatch()` / `runEvalBatch()`：
  - 现在返回 `dataset_index_per_molecule`
  - 总是返回 `num_structures_per_molecule`
- `splitBatchCount()`：
  - 当 `dataset_sampling_strategy == balanced` 时，训练 batch 数通过 `buildDatasetBalancedGroupedBatchIndices()` 估算
- `collectEpochMetrics()`：
  - 现在汇总：
    - `dataset_index_per_molecule`
    - `num_structures_per_molecule`
  - 调 `RegressionMetrics.summarize(..., dataset_index_per_molecule, dataset_names)`
  - 对 `per_dataset` 逐个计算 `focus_priority_score`
  - 生成 `macro_focus_priority_score`
- `runSplit()`：
  - 调 `createMoleculeIterator()` 时传：
    - `mixed_tail_top_fraction`
    - `dataset_sampling_strategy`
- `logEpochMetrics()`：
  - 可以打印 `macro_focus_score`
- `runTrainingLoop()`：
  - DATA 日志里输出 per-dataset molecule / structure 统计
  - BATCHING 日志里输出 `dataset_sampling`

#### 为什么这样改

这是训练主流程对 unified training 的承接层：

- metrics 要知道 dataset 维度；
- batch 数估算要对 balanced batching 兼容；
- best checkpoint 最终需要能基于 macro monitor 做；
- 日志必须足够详细，方便后续继续调实验。


### `e3_o3_property_test.py`

#### 关键改动

- 手写 `UnifiedBatch(...)` 的位置补上：
  - `dataset_index_per_molecule=batch.dataset_index_per_molecule`

#### 为什么这样改

`UnifiedBatch` schema 变了，这个测试辅助构造函数必须同步，否则运行测试会缺字段直接报错。


### `config_e2e_checked_vb_unified_topmass.yaml`

#### 新增内容

提供了一份可以直接作为 unified training 起点的配置模板：

- `data.datasets`
  - `checked`
  - `vb`
- `validation_monitor: macro_focus_priority_score`
- `focus_monitor_precision_weight: 0.15`
- `focus_monitor_tail_fpr_weight: 0.10`
- `top_mass_ranking_weight: 0.35`
- `top_mass_sample_strategy: focus_plus_mixed_tail`
- `mixed_tail_top_fraction: 0.5`
- `max_tail_samples_per_molecule: 96`
- `dataset_sampling_strategy: balanced`

#### 为什么这样改

用户明确希望统一训练，且前面讨论已经形成了第一版统一训练配方，所以给一份落地 config 模板，避免新会话重新拼字段。


## 4. 已运行命令 / 检查 / 测试 / 报错与结果

### 代码检索与阅读命令

本轮大量使用了以下只读命令做上下文确认：

- `rg -n "..." /pool1/home/ysxin/E3nn/E3VB -S`
- `sed -n '...' file`
- `nl -ba file | sed -n '...'`
- `git status --short`
- `git diff --stat -- ...`
- `tail -n ... file`
- `head -n ... file`

这些命令主要用于：

- 确认 top-mass objective 的实现位置
- 确认 metrics / monitor / batch / cache / predict 的口径
- 确认统一训练需要改到哪些模块
- 检查 worktree 是否脏


### 关键执行过的检查

#### 1. Python 语法检查

已执行：

```bash
python -m py_compile /pool1/home/ysxin/E3nn/E3VB/data/schema.py /pool1/home/ysxin/E3nn/E3VB/data/processor.py /pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py /pool1/home/ysxin/E3nn/E3VB/utils/config.py /pool1/home/ysxin/E3nn/E3VB/utils/metrics.py /pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py /pool1/home/ysxin/E3nn/E3VB/main.py
```

后来又追加：

```bash
python -m py_compile /pool1/home/ysxin/E3nn/E3VB/data/schema.py /pool1/home/ysxin/E3nn/E3VB/data/processor.py /pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py /pool1/home/ysxin/E3nn/E3VB/utils/config.py /pool1/home/ysxin/E3nn/E3VB/utils/metrics.py /pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py /pool1/home/ysxin/E3nn/E3VB/main.py /pool1/home/ysxin/E3nn/E3VB/e3_o3_property_test.py
```

结果：

- 均成功通过
- 说明本轮修改至少在语法层面是成立的


#### 2. 轻量运行时配置检查尝试

执行过：

```bash
python -c "import yaml; from utils.config import ConfigFactory; from data.processor import XmoDatasetProcessor; ..."
```

报错：

```text
ModuleNotFoundError: No module named 'yaml'
```

说明：

- 当前 shell Python 环境缺少 `PyYAML`
- 因此无法直接在当前环境做 config runtime smoke parse


#### 3. 轻量 processor 行为检查尝试

执行过：

```bash
python -c "from data.schema import DatasetSourceConfig; from data.processor import UnifiedDataConfig, XmoDatasetProcessor; ..."
```

报错：

```text
ModuleNotFoundError: No module named 'numpy'
```

说明：

- 当前 shell Python 环境缺少 `numpy`
- 不能在当前环境做 preprocess/runtime 行为 smoke test


### git / worktree 状态

执行过：

```bash
git status --short
```

观察到：

- worktree 是脏的
- 很多文件本来就已经有未提交修改，不是本轮新增的
- 本轮遵守了“不回滚用户已有改动”的原则

注意：

- 新会话继续改代码时，不要误清理已有未提交变更


## 5. 本轮保留的关键分析结论

### 关于当前 target 与 focus 的定义

- target 仍然是每个 molecule 内：
  - `normalized_weight = weight / weight_max`
  - 实现在：
    - `/pool1/home/ysxin/E3nn/E3VB/data/xmo_builder.py:422`
- `focus_cumulative_mass` 仍然是：
  - `0.98`
- top-mass focus 定义保持不变

### 关于 `focusPriorityScore`

现在的 `focusPriorityScore()` 已经变为：

- `focus_weighted_mae`
- `pair_penalty`
- `spearman_penalty`
- `recall_penalty`
- `precision_penalty`
- `tail_false_positive_rate`

也就是对 `focus_precision` 有了显式惩罚，不再只偏向 recall。


### 关于 unified training 的第一版配方

当前模板中保留的重要设置：

- `use_top_mass_objective: true`
- `focus_cumulative_mass: 0.98`
- `validation_monitor: macro_focus_priority_score`
- `top_mass_ranking_weight: 0.35`
- `top_mass_sample_strategy: focus_plus_mixed_tail`
- `mixed_tail_top_fraction: 0.5`
- `max_tail_samples_per_molecule: 96`
- `dataset_sampling_strategy: balanced`
- `molecule_balanced_sampling: true`


## 6. 当前未解决问题 / 潜在风险 / 下一步建议

### A. 最重要的未完成项：molecule-local loss

还没有改：

- `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`

需要继续做的事：

1. 重构 `topMassObjectiveLoss()`
2. 让以下项都变成 molecule-local：
   - `focus_regression_loss`
   - `tail_regression_loss`（是否也 local，需要再决定）
   - `focus_rank_loss`
   - `focus_tail_rank_loss`
3. 最终对 molecule 求平均，而不是全 batch 直接汇总归一化

原因：

- 这是前面分析里最关键的结构性问题；
- 仅靠 balanced dataset batching 还不够；
- 大结构数 molecule 仍可能在 loss 里权重更大。


### B. 运行时尚未验证

由于当前环境缺少：

- `numpy`
- `yaml`

所以还没有完成：

- preprocess 阶段真实跑通
- train 阶段真实 batch 构造跑通
- unified config 的完整 runtime parse

下一步建议：

1. 进入有完整 Python 依赖的环境
2. 先跑：
   - preprocess
   - smoke train
3. 再看：
   - cache 是否生成成功
   - logs 里 per-dataset 统计是否正确
   - `macro_focus_priority_score` 是否真的进入 best checkpoint 逻辑


### C. 旧缓存兼容风险

虽然很多地方都用 `getattr(..., default)` 做了兼容，但仍有几个实际风险：

- 旧的 `ProcessedDatasetCache` / `PackedDatasetCache` pickle 里没有：
  - `dataset_ids`
  - `split_ids_by_dataset`
  - `split_structure_counts_by_dataset`
  - `dataset_id`
- 老 cache 如果直接加载，理论上 dataclass 反序列化通常还能成功，但运行路径要靠 `getattr` 才能兜住

建议：

- 对 unified training **重新 preprocess**
- 不要直接复用老的单数据集 cache 作为 unified cache


### D. `validation_monitor` 虽已支持 macro，但 `runTrainingLoop()` 只做了通用读取

当前逻辑是：

- `monitor_name = self.validationMonitorName()`
- `val_metrics[monitor_name]`

只要 `collectEpochMetrics()` 已生成 `macro_focus_priority_score`，就能工作。

需要新会话在真实 runtime 中重点确认：

- `val_metrics` 中确实有 `macro_focus_priority_score`
- `BEST` 日志打印没有 KeyError


### E. `focus_plus_mixed_tail` 只是第一版启发式

当前实现：

- `mixed_top_fraction`
- 先取 top tail
- 剩余部分 random tail

需要后续验证：

- 对 `checked` 是否比纯 random 更稳
- 对 `vb` 是否比纯 top tail 更不伤 recall


### F. `e3_o3_property_test.py` 只补了 schema 兼容

目前只修了：

- `dataset_index_per_molecule`

但没有真的跑测试，因为环境依赖不完整。


## 7. 关键路径 / 类名 / 函数名 / 变量名 / 参数名备忘

### 新增 / 重要对象

- `DatasetSourceConfig`
- `UnifiedDataConfig.datasets`
- `UnifiedBatch.dataset_index_per_molecule`
- `ProcessedDatasetCache.split_ids_by_dataset`
- `PackedDatasetCache.split_structure_counts_by_dataset`

### 新增 / 重要函数

- `XmoDatasetProcessor.dataSources()`
- `XmoDatasetProcessor.usesMultipleDatasets()`
- `XmoDatasetProcessor.cacheMoleculeId()`
- `RegressionMetrics.splitPayloadByDataset()`
- `GrainPipeline.chunkGroupDatasetId()`
- `GrainPipeline.buildDatasetBalancedGroupedBatchIndices()`
- `validDatasetIndices()`

### 重要已有函数（本轮依赖 / 接口改变）

- `createMoleculeSplits()`
- `buildProcessedDatasetCache()`
- `buildPackedDatasetCache()`
- `packBatch()`
- `packPackedChunkBatch()`
- `createMoleculeIterator()`
- `collectEpochMetrics()`
- `focusPriorityScore()`

### 重要参数

- `focus_cumulative_mass`
- `validation_monitor`
- `focus_monitor_pair_acc_weight`
- `focus_monitor_spearman_weight`
- `focus_monitor_recall_weight`
- `focus_monitor_precision_weight`
- `focus_monitor_tail_fpr_weight`
- `top_mass_ranking_weight`
- `top_mass_sample_strategy`
- `mixed_tail_top_fraction`
- `max_tail_samples_per_molecule`
- `dataset_sampling_strategy`
- `molecule_balanced_sampling`

### 重要 monitor 名

- `focus_priority_score`
- `macro_focus_priority_score`
- `macro_focus_weighted_mae`
- `macro_mae`

### 重要 unified config 路径

- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml`


## 8. 下一次 Codex 会话建议从哪里继续

### 推荐继续顺序

1. **先不要继续大改别的模块**
2. 先进入有完整依赖的环境，做 unified config 的 runtime smoke test
3. 重点验证：
   - preprocess 能否成功生成 unified cache
   - train 一个 smoke run 能否成功构 batch
   - `per_dataset` / `macro_focus_priority_score` 是否进入日志和 checkpoint
4. 如果这一步通过，再改：
   - `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`
   - 做 molecule-local averaging
5. 最后再考虑：
   - subset-normalized auxiliary loss
   - `active_only_low_risk` ablation

### 新会话最值得先读的文件

1. `/pool1/home/ysxin/E3nn/E3VB/HANDOFF.md`
2. `/pool1/home/ysxin/E3nn/E3VB/AGENTS.md`（如果存在并被项目依赖）
3. `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml`
4. `/pool1/home/ysxin/E3nn/E3VB/data/processor.py`
5. `/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py`
6. `/pool1/home/ysxin/E3nn/E3VB/main.py`
7. `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`

### 新会话的第一条行动建议

优先做：

- 用 unified config 跑一次 preprocess / smoke train

原因：

- 当前最大不确定性不是思路，而是运行时链路是否完全打通；
- `molecule-local loss` 适合在运行链路确认无误之后再改。

## 2026-05-05 00:58 本轮工作总结

### 1. 本轮目标

- 先进入 `/pool1/home/ysxin/E3nn/E3VB`，读取 `HANDOFF.md` 和 `AGENTS.md`，确认上一轮交接信息与项目约束。
- 优先完成 unified config 的 runtime smoke test，验证 `preprocess -> packed batch -> macro monitor` 链路是否真正跑通。
- 在 smoke test 确认通过后，继续改 `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`，把 top-mass objective 中最关键的 loss 项改成 molecule-local averaging。
- 按用户要求在 `pc004` 上提交三个正式训练方向：
  - unified: `checked_outfile + out_file`
  - checked only: `/pool1/home/ysxin/autovb_newoutfile/checked_outfile`
  - vb only: `/pool1/home/ysxin/E3nnVB/VB/out_file`

### 2. 已完成内容

- 已读取并核对：
  - `/pool1/home/ysxin/E3nn/E3VB/HANDOFF.md`
  - `/pool1/home/ysxin/E3nn/E3VB/AGENTS.md`
- 已确认本地默认 `python` 环境无法直接运行项目：
  - 缺少 `numpy/yaml/jax/flax/optax/grain/jraph`
  - `e3vb_jax` / `e3vb_e2e` conda 环境在本机还受 `GLIBC` 版本约束，不能本地完成正式 runtime test
- 已转到 `pc004` 做 runtime 验证，并完成一套最小 unified smoke 数据子集：
  - 新建 `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_input/checked`
  - 新建 `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_input/vb`
  - 每个目录各链接 `10` 个真实 `.xmo` 文件
  - 新增 smoke 配置 `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_config.yaml`
- 已在 `pc004` 成功完成 unified preprocess smoke：
  - 作业号：`1946749`
  - 日志：`/pool1/home/ysxin/E3nn/E3VB/logs/preprocess_1946749.log`
  - 结果：
    - `train_molecules=16`
    - `val_molecules=2`
    - `test_molecules=2`
    - packed chunk 文件数 `906`
- 已在 `pc004` 成功完成 unified smoke train（loss 改动前）：
  - 作业号：`1946750`
  - 日志：`/pool1/home/ysxin/E3nn/E3VB/logs/smoke_1946750.log`
  - 确认内容：
    - GPU/JAX 运行正常（RTX 4090, backend=gpu）
    - `dataset_sampling=balanced` 已进入 runtime
    - `DATASET split=... dataset_id=checked/vb` 统计已打印
    - `macro_focus_score` 已在 train/val/test 日志中出现
    - `SMOKE status=finished`
- 已完成 `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py` 的 molecule-local 改造：
  - 新增 molecule 聚合辅助函数
  - 新增 molecule-local weighted MAE
  - 将 pairwise ranking / cross-set ranking 改成 per-molecule loss 后再平均
  - `topMassObjectiveLoss()` 现在使用 molecule-local 的：
    - `focus_regression_loss`
    - `tail_regression_loss`
    - `focus_rank_loss`
    - `focus_tail_rank_loss`
- 已在 `pc004` 成功完成 unified smoke train（loss 改动后复测）：
  - 作业号：`1946751`
  - 日志：`/pool1/home/ysxin/E3nn/E3VB/logs/smoke_1946751.log`
  - 结果：JIT / train / val / test 路径均可执行，`SMOKE status=finished`
- 已按用户要求提交 3 个正式训练方向：
  - unified 正式 preprocess：`1946752`
  - vb only 正式 train：`1946753`
  - checked only 正式 train：`1946754`
  - unified 正式 train：`1946755`
- 当前提交策略：
  - `1946755` 使用 `--dependency=afterok:1946752`
  - unified 正式训练会在 unified preprocess 成功后自动启动

### 3. 修改/新增/检查过的文件

#### 已修改

- `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`

#### 已新增

- `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_config.yaml`

#### 已创建/填充的临时目录

- `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_input/checked`
- `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_input/vb`

#### 已检查但未修改

- `/pool1/home/ysxin/E3nn/E3VB/main.py`
- `/pool1/home/ysxin/E3nn/E3VB/data/processor.py`
- `/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_outfile_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/submit_preprocess.sbatch`
- `/pool1/home/ysxin/E3nn/E3VB/submit_train.sbatch`

### 4. 关键实现逻辑

#### A. unified runtime smoke test 的做法

- 没有直接拿正式全量数据先跑，而是先构造了一个最小但真实的 unified smoke 子集：
  - `checked` 目录取前 `10` 个 `.xmo`
  - `vb` 目录取前 `10` 个 `.xmo`
- 原因：
  - 这样能快速验证 runtime 路径，而不必先等待正式全量 preprocess
  - 同时仍然覆盖 unified data path、dataset split、packed chunk、dataset-balanced batching、macro monitor
- `tmp_unified_smoke_config.yaml` 基本沿用 unified 正式模板，但把：
  - `data.datasets[*].xmo_dir`
  - `processed_cache_path`
  - `packed_cache_path`
  - `checkpoint_path`
  - `log_path`
  都改到了临时路径，避免污染正式产物

#### B. molecule-local loss 的具体改法

- 在 `utils/losses.py` 中新增：
  - `moleculeIndexFromCounts()`
  - `segmentSumByMolecule()`
  - `moleculeLocalWeightedMae()`
- `moleculeLocalWeightedMae()` 的逻辑：
  - 先用 `num_structures_per_molecule` 生成每个 structure 对应的 `molecule_index`
  - 在每个 molecule 内独立归一化 target-weight
  - 在每个 molecule 内计算 weighted absolute error
  - 最后只对 valid molecules 做平均
- `pairwiseRankLoss()` 现在不再是全 batch 直接按 pair 汇总归一化，而是：
  - 先限制到 same-molecule pair
  - 再用 `jnp.bincount` 聚合成 per-molecule pair-weight 和 per-molecule pair-loss
  - 最后对有有效 pair 的 molecules 做平均
- `crossSetMarginLoss()` 也采用同样的 per-molecule averaging 模式
- `topMassObjectiveLoss()` 现在调用：
  - `moleculeLocalWeightedMae(..., sample_mask=focus_sample_mask)` 作为 `focus_regression_loss`
  - `moleculeLocalWeightedMae(..., sample_mask=tail_sample_mask)` 作为 `tail_regression_loss`
  - 带 `sample_mask` 的 `pairwiseRankLoss()` 作为 `focus_rank_loss`
  - 带 `sample_mask` 的 `crossSetMarginLoss()` 作为 `focus_tail_rank_loss`
- 这样可以避免“大结构数 molecule 在同一个 batch 内自然拥有更大 loss 权重”的问题；即使 dataset-balanced batching 已经做了，大 molecule 也不再通过 loss 汇总再次主导梯度

#### C. 正式训练提交逻辑

- unified 正式训练需要先有 unified cache，因此先提交：
  - `1946752` preprocess
- 然后再提交：
  - `1946755` unified train
- `1946755` 加了 slurm 依赖：
  - `--dependency=afterok:1946752`
- checked only / vb only 两个单数据集训练由于 cache 已存在，所以直接提交：
  - `1946754`
  - `1946753`

### 5. 已运行命令与结果

#### 本地检查命令

- `date '+%F %R'`
  - 返回：`2026-05-05 00:58`
- `python - <<'PY' ... import numpy,yaml,jax,...`
  - 结果：默认本地环境缺少 `numpy/yaml/jax/flax/optax/grain/jraph`
- `bash -lc 'source ... conda activate e3vb_jax ...'`
  - 结果：`jax` 受 `GLIBC_2.27` 限制，无法在本机直接跑
- `bash -lc 'source ... conda activate e3vb_e2e ...'`
  - 结果：`python` 启动即受更高版本 `GLIBC` 限制

#### `pc004` smoke 验证

- 提交 unified smoke preprocess：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_config.yaml FORCE_REBUILD=1 sbatch -p pc -w pc004 submit_preprocess.sbatch`
  - 作业号：`1946749`
  - 结果：成功
- 提交 unified smoke train（loss 改动前）：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_config.yaml RUN_MODE=smoke sbatch -p pc -w pc004 submit_train.sbatch`
  - 作业号：`1946750`
  - 结果：成功
- 提交 unified smoke train（loss 改动后复测）：
  - 同上
  - 作业号：`1946751`
  - 结果：成功

#### `pc004` 正式作业提交

- unified 正式 preprocess：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml FORCE_REBUILD=1 sbatch -p pc -w pc004 submit_preprocess.sbatch`
  - 作业号：`1946752`
- vb only 正式训练：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass.yaml RUN_MODE=vb_only sbatch -p pc -w pc004 submit_train.sbatch`
  - 作业号：`1946753`
- checked only 正式训练：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_outfile_topmass.yaml RUN_MODE=checked_only sbatch -p pc -w pc004 submit_train.sbatch`
  - 作业号：`1946754`
- unified 正式训练（依赖 preprocess）：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml RUN_MODE=unified_checked_vb sbatch --dependency=afterok:1946752 -p pc -w pc004 submit_train.sbatch`
  - 作业号：`1946755`

#### 当前已知作业状态（写本条 handoff 时）

- `1946752`: `R`，正在做 unified preprocess
- `1946753`: `R`，`vb_only` 已进入 `ITERATOR_STAGE split=train ... batch_packed`
- `1946754`: `PD (Resources)`
- `1946755`: `PD (Dependency)`，等待 `1946752`

### 6. 当前报错/未解决问题

#### A. 本机环境问题

- 本地默认 shell 环境无法直接运行项目依赖
- `e3vb_jax` / `e3vb_e2e` 在本机受 `GLIBC` 版本限制
- 结论：后续 runtime 验证和正式训练仍应优先走 `pc004`

#### B. best checkpoint 逻辑还没被正式验证

- smoke 模式下虽然 `macro_focus_score` 已在日志中出现，但 smoke 运行本身不等价于正式 best-checkpoint 保存逻辑
- 所以还需要等 unified 正式训练 `1946755` 真正跑起来后确认：
  - `macro_focus_priority_score` 是否进入 best checkpoint monitor
  - 是否存在 `KeyError` / monitor 读取错误

#### C. 正式 unified preprocess 尚未完成

- 正式 unified cache：
  - `/pool1/home/ysxin/E3nn/E3VB/processed/processed_dataset_cache_checked_vb_unified.pkl`
  - `/pool1/home/ysxin/E3nn/E3VB/processed/packed_dataset_cache_checked_vb_unified.pkl`
  在本条 handoff 追加时仍由 `1946752` 生成中

#### D. GPU 驱动有一个非阻塞 warning

- 训练日志里出现：
  - CUDA driver `12.1` 低于 PTX compiler `12.9.86`
  - XLA 禁用了 parallel compilation
- 当前只是 warning，不影响 smoke 和训练启动，但可能拖慢编译速度

### 7. 重要路径、变量名、函数名、参数

#### 重要路径

- unified 正式 config：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml`
- checked only 正式 config：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_outfile_topmass.yaml`
- vb only 正式 config：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass.yaml`
- smoke config：
  - `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_config.yaml`
- smoke input：
  - `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_input/checked`
  - `/pool1/home/ysxin/E3nn/E3VB/tmp_unified_smoke_input/vb`
- unified smoke logs：
  - `/pool1/home/ysxin/E3nn/E3VB/logs/preprocess_1946749.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/smoke_1946750.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/smoke_1946751.log`
- 正式作业相关日志：
  - `/pool1/home/ysxin/E3nn/E3VB/logs/preprocess_1946752.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_preprocess_1946752.slurm.out`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946753.slurm.out`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946754.slurm.out`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946755.slurm.out`

#### 重要函数名

- `moleculeIndexFromCounts()`
- `segmentSumByMolecule()`
- `moleculeLocalWeightedMae()`
- `pairwiseRankLoss()`
- `crossSetMarginLoss()`
- `topMassObjectiveLoss()`
- `createMoleculeSplits()`
- `buildProcessedDatasetCache()`
- `createMoleculeIterator()`
- `collectEpochMetrics()`

#### 重要变量名 / 参数

- `num_structures_per_molecule`
- `sample_mask`
- `top_mass_focus_mask`
- `focus_sample_mask`
- `tail_sample_mask`
- `dataset_sampling_strategy`
- `molecule_balanced_sampling`
- `validation_monitor`
- `macro_focus_priority_score`
- `focus_monitor_precision_weight`
- `mixed_tail_top_fraction`
- `top_mass_ranking_weight`
- `tail_suppression_weight`
- `RUN_MODE`
- `CONFIG_PATH`
- `FORCE_REBUILD`

#### 重要作业号

- smoke preprocess: `1946749`
- smoke train before loss patch: `1946750`
- smoke train after loss patch: `1946751`
- unified formal preprocess: `1946752`
- vb only formal train: `1946753`
- checked only formal train: `1946754`
- unified formal train: `1946755`

### 8. 下一轮 Codex 应该继续做什么

1. 先检查 `1946752` 是否成功结束。
   - 若成功，确认以下文件确实生成：
     - `/pool1/home/ysxin/E3nn/E3VB/processed/processed_dataset_cache_checked_vb_unified.pkl`
     - `/pool1/home/ysxin/E3nn/E3VB/processed/packed_dataset_cache_checked_vb_unified.pkl`
   - 再确认 `1946755` 是否自动从 `Dependency` 变为 `R`

2. 继续盯三个正式训练日志：
   - `1946753`
   - `1946754`
   - `1946755`
   重点看：
   - 是否打印 `DATASET split=...`
   - 是否打印 `dataset_sampling=balanced`（仅 unified）
   - 是否出现 `macro_focus_score=...`（仅 unified）
   - 是否出现 OOM、NaN、monitor key 缺失、checkpoint 保存异常

3. 对 unified 正式训练 `1946755`，优先确认 best-checkpoint 逻辑。
   - 需要验证：
     - `validation_monitor: macro_focus_priority_score` 是否真实生效
     - 是否有 `BEST` / checkpoint 更新日志

4. 如果正式训练能稳定起跑，再考虑是否需要把单数据集 config 也同步到新的 molecule-local loss 策略配套超参。
   - 当前 `utils/losses.py` 已全局生效，但单数据集 config 的 `top_mass_ranking_weight`、`top_mass_sample_strategy` 还沿用旧配方

5. 如果后续出现训练不稳定或 monitor 表现异常，再按顺序排查：
   - `utils/losses.py` 的 molecule-local averaging 是否导致 loss scale 改变
   - `tail_suppression_weight` 是否需要重新调
   - `focus_monitor_precision_weight` / `focus_monitor_tail_fpr_weight` 是否需要重新平衡

## 2026-05-05 16:41 本轮工作总结

### 1. 本轮目标

- 检查三轮正式训练（`checked_only` / `vb_only` / `unified checked+vb`）的真实运行结果，确认成功与失败原因。
- 分析 `checked_only` 训练里为什么 `val/test spearman` 看起来偏低，并确认当前 `spearman` 的定义。
- 给日志新增“每个 molecule 单独计算 Spearman 再平均”的指标，避免继续被全局 pooled `spearman` 误导。
- 直接测试当前模型是否满足 E(3) 不变性 / 反演不变性：
  - 对一个 structure 做平移、旋转、反演
  - 检查预测结果是否不变
- 使用 `checked_outfile` 训练出的模型，拿 `vb_outfile` 数据做跨分布泛化检查：
  - 先随机抽 3 个 `vb_outfile` 分子做预测与不变性测试
  - 再随机抽 10 个 `vb_outfile` molecules 统计 pooled/focus Spearman 分布
- 在不减小 `batch_size` 的前提下，给 `vb_only` 和 unified 设计“长期可用”的低显存 bucket 配置，并在 `pc004` 上重新提交训练。

### 2. 已完成内容

#### A. 三轮正式训练结果已确认

- `checked_only` 正式训练 `1946754`：
  - 完整跑完
  - 最终日志出现：
    - `TRAINING_DONE best_epoch=1054`
    - `best_val_focus_priority_score=0.033014`
  - 最佳 checkpoint 存在：
    - `/pool1/home/ysxin/E3nn/E3VB/checkpoints/checked_only_1946754_best.pkl`
- `vb_only` 正式训练 `1946753`：
  - 在训练后期 GPU OOM 退出
  - 但在退出前已经保存过 best checkpoint：
    - `/pool1/home/ysxin/E3nn/E3VB/checkpoints/vb_only_1946753_best.pkl`
  - 最近一次已知 `BEST`：
    - `epoch=54`
    - `val_focus_priority_score=0.092252`
- unified 正式训练 `1946755`：
  - unified preprocess `1946752` 成功
  - 正式 unified cache 已生成
  - 训练在 `epoch=0` 中途 GPU OOM 退出
  - 没有生成 unified best checkpoint

#### B. `spearman` 指标定义已经确认并向用户解释

- 当前日志里的：
  - `train spearman`
  - `val spearman`
  - `test spearman`
  都是“当前 split 内所有 structure pooled 后一起算一次全局 Spearman”
- 不是：
  - 每个 molecule 单独算再平均
  - 也不是只看 focus structures
- 已向用户明确说明：
  - `focus_spearman` 才是“每个 molecule 的 true focus set 内部算 Spearman，再平均”
  - 这也是为什么 `checked_outfile` 上 `val/test spearman` 看起来一般，但 `focus_spearman` 很高

#### C. 新增 `molecule_macro_spearman`

- 已在 `utils/metrics.py` 中新增：
  - `computeMoleculeMacroSpearman()`
- 逻辑：
  - 按 `num_structures_per_molecule` 切分
  - 对每个 molecule 的全部 structures 单独算一次 Spearman
  - 对所有可计算 molecule 取平均
- 已把这个指标接入：
  - `RegressionMetrics.summarize()`
  - `main.py` 的 `EPOCH` 日志打印
- 后续新的训练日志会包含：
  - `spearman=...`
  - `molecule_macro_spearman=...`
  - `focus_spearman=...`

#### D. 不变性测试脚本已修复并跑通

- 已修改 `/pool1/home/ysxin/E3nn/E3VB/e3_o3_property_test.py`
  - 修正为与当前 `AtomE3Encoder.__call__()` 接口一致：
    - 显式传 `atom_number`
    - 显式传 `positions`
  - 新增可选参数：
    - `--checkpoint`
  - 现在可以测试“已训练好的当前模型”，不再只是随机初始化模型
- 已实际测试：
  - config: `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_outfile_topmass.yaml`
  - checkpoint: `/pool1/home/ysxin/E3nn/E3VB/checkpoints/checked_only_1946754_best.pkl`
  - split: `test`
  - sample_index: `0`
- 结果：
  - translation：最终预测绝对误差 `0`
  - rotation：最终预测绝对误差 `5.96e-08`
  - reflection：最终预测绝对误差 `0`
- 结论：
  - 对已测试 structure，当前训练好的 `checked_only` 模型满足 E(3) 不变性，并且在反演下保持不变

#### E. 用 `checked_only` 模型测试 `vb_outfile` 的跨分布输入

- 使用模型：
  - `/pool1/home/ysxin/E3nn/E3VB/checkpoints/checked_only_1946754_best.pkl`
- 使用 `vb_outfile` 配置：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass.yaml`
- 已先对 3 个随机 `vb_outfile` test molecules 做完整分子预测分析：
  - `5283331_VBSCF`
  - `57416843_VBSCF`
  - `7264_VBSCF`
- 已输出每个分子的：
  - `mae`
  - `rmse`
  - molecule-level pooled `spearman`
  - top-1 预测结构
  - true top-1 结构
  - top-5 predicted structures
  - top-5 true structures
- 结论：
  - 这个 `checked` 模型迁移到 `vb_outfile` 上不是完全失效
  - 3 个分子里：
    - 2 个 top-1 命中
    - 1 个 top-1 不命中

#### F. 对 3 个 molecule 和额外 10 个 molecule 都做了 focus-only 排序分析

- 注意：
  - 直接从 `materializeUnifiedSamples()` 拿到的 `sample.top_mass_focus` 在这条分析链路里不可靠，3 个 molecule 一开始都得到 `focus_count=0`
  - 后来改为用 `utils.top_mass.computeTopMassFocusMask()` 按真实 `target` 和 `focus_cumulative_mass=0.98` 现场重建 true focus mask
- 对最开始 3 个 `vb_outfile` molecules：
  - `5283331_VBSCF`: pooled `0.544` -> focus `0.780`
  - `57416843_VBSCF`: pooled `0.604` -> focus `0.632`
  - `7264_VBSCF`: pooled `0.958` -> focus `0.891`
  - summary:
    - `focus_spearman mean = 0.768`
    - `median = 0.780`
- 又随机抽了 10 个 `vb_outfile` test molecules，用 `checked` 模型做批量统计：
  - molecules:
    - `15523_VBSCF`
    - `18457362_VBSCF`
    - `22764375_VBSCF`
    - `34857_VBSCF`
    - `413657_VBSCF`
    - `5283331_VBSCF`
    - `57416843_VBSCF`
    - `6442690_VBSCF`
    - `7000_VBSCF`
    - `7264_VBSCF`
  - summary:
    - pooled `spearman mean = 0.82064`
    - pooled `spearman median = 0.95492`
    - pooled `spearman min = 0.32302`
    - pooled `spearman max = 0.97903`
    - focus `spearman mean = 0.84804`
    - focus `spearman median = 0.88771`
    - focus `spearman min = 0.63171`
    - focus `spearman max = 0.94367`
    - `mae_mean = 0.01177`
    - `rmse_mean = 0.02198`
    - `top1_hit_rate = 0.9`
    - `focus_minus_pooled_mean = +0.02740`
- 结论：
  - 对 `vb_outfile`，`checked` 模型的跨分布泛化比“全结构 pooled Spearman”看起来更好
  - 真正重要的 focus 区间排序质量整体优于全结构排序

#### G. 低显存长期方案已落实到新 config，并已重新提交训练

- 没有使用减小 `batch_size` 的方案
- 选择的长期可用方向是：
  - 保留 `batch_size=8`
  - 保留 `bucket_key=num_atoms_num_orbitals`
  - 收紧 fixed bucket 的步长，减少 padding 浪费和过大静态 shape
- 已新增两份 lowmem config：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass_lowmem.yaml`
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml`
- 新 config 统一改动：
  - `fixed_bucket_graph_step: 16 -> 8`
  - `fixed_bucket_static_atom_step: 32 -> 16`
  - `fixed_bucket_atom_step: 256 -> 128`
  - `fixed_bucket_atom_edge_step: 2048 -> 1024`
  - `fixed_bucket_orbital_step: 512 -> 256`
  - `fixed_bucket_rumer_edge_step: 4096 -> 2048`
  - `fixed_bucket_active_orbital_step: 128 -> 64`
  - `fixed_bucket_active_edge_step: 1024 -> 512`
- 已用这两份新 config 在 `pc004` 上重新提交：
  - `vb_only_lowmem`: `1946830`
  - `unified_checked_vb_lowmem`: `1946831`
- 截至本条 handoff 追加时：
  - 两个作业都已经进入 `R`
  - 两个作业都已经成功通过首个 `batch_packed`

### 3. 修改/新增/检查过的文件

#### 已修改

- `/pool1/home/ysxin/E3nn/E3VB/utils/metrics.py`
- `/pool1/home/ysxin/E3nn/E3VB/main.py`
- `/pool1/home/ysxin/E3nn/E3VB/e3_o3_property_test.py`

#### 已新增

- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass_lowmem.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml`

#### 已检查但未修改

- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_outfile_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass.yaml`
- `/pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py`
- `/pool1/home/ysxin/E3nn/E3VB/predict.py`
- `/pool1/home/ysxin/E3nn/E3VB/model/atom_encoder.py`
- `/pool1/home/ysxin/E3nn/E3VB/model/end_to_end.py`
- `/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py`
- `/pool1/home/ysxin/E3nn/E3VB/data/processor.py`
- `/pool1/home/ysxin/E3nn/E3VB/submit_train.sbatch`

### 4. 关键实现逻辑

#### A. `molecule_macro_spearman`

- 新增 `RegressionMetrics.computeMoleculeMacroSpearman()`
- 输入：
  - `prediction`
  - `target`
  - `num_structures_per_molecule`
- 逻辑：
  1. 用 `counts` 切回每个 molecule
  2. 对每个 molecule 的全部 structures 单独计算 `computeSpearman()`
  3. 跳过 `count <= 1` 的 molecule
  4. 对所有 molecule-level Spearman 取平均
- 在 `summarize()` 中，如果 `num_structures_per_molecule` 存在，就写入：
  - `summary["molecule_macro_spearman"]`
- 在 `main.py` 的 `logEpochMetrics()` 中，EPOCH 行新增：
  - `molecule_macro_spearman=...`

#### B. 不变性测试脚本修复与使用方式

- `e3_o3_property_test.py` 原先的测试脚本与当前 `AtomE3Encoder` 接口漂移：
  - 当前 `atom_encoder` 需要显式传 `atom_number` 和 `positions`
- 修复后：
  - 增加 `expandedAtomInputs()`，按当前主模型 forward 逻辑构造 expanded atom inputs
  - `reportTransform()` 中直接调用当前 `atom_encoder` 正确接口
- 新增参数：
  - `--checkpoint`
- 现在可以测试：
  - 随机初始化模型
  - 或训练完成后的 checkpoint-backed 模型
- 已说明：
  - 最终比较的“原始预测”和“变换后预测”，是同一个 structure-level sample 的标量预测
  - 不是整 molecule 聚合值，也不是所有 structures 一起比较

#### C. `vb_outfile` 跨分布分析的方法

- 用 `checked_only_1946754_best.pkl` 作为模型
- 用 `vb_outfile` 的 config 生成 test split 分子
- 对每个 molecule：
  - 用 `processor.materializeUnifiedSamples(molecule)` 展开为所有 structures
  - 用 `GraphPackingAdapter.packBatch(samples=..., num_structures_per_molecule=[len(samples)])` 打成单-molecule batch
  - 让当前 `checked` 模型输出这个 molecule 所有 structures 的预测值
- 统计：
  - `mae`
  - `rmse`
  - pooled molecule-level `spearman`
  - `pred_top`
  - `true_top`
  - `top1_hit`
- focus-only Spearman 的正确做法：
  - 不依赖 `sample.top_mass_focus`
  - 直接用 `computeTopMassFocusMask(target, [len(samples)], cumulative_mass=0.98)` 现场重建 true focus mask

#### D. 长期可用的低显存方案

- 本轮没有采用：
  - 减小 `batch_size`
  - 随机裁数据
  - 关闭 fixed bucket
- 采用的长期方案是：
  - 保持 `batch_size=8`
  - 保持分桶 key 不变：`num_atoms_num_orbitals`
  - 仅缩小 fixed bucket 步长
- 这样做的原因：
  - 当前 OOM 更像是超大 batch 命中过粗 bucket，触发巨大的静态 padding 和大 shape 分配
  - 通过减小 fixed bucket step，可以长期降低显存浪费，同时保留 fixed shape/JIT 稳定性

### 5. 已运行命令与结果

#### 训练结果检查

- 查看三轮训练日志和 checkpoint：
  - 结论：
    - `checked_only_1946754` 成功跑完
    - `vb_only_1946753` GPU OOM，但已有 best checkpoint
    - `unified_1946755` GPU OOM，且无 checkpoint
- 关键 OOM 报错：
  - `vb_only`：
    - `RESOURCE_EXHAUSTED: Failed to allocate request for 5.41GiB on device ordinal 0`
  - `unified`：
    - `RESOURCE_EXHAUSTED: Failed to allocate request for 18.18GiB on device ordinal 0`
- 结论：
  - 两次失败都是 GPU 显存炸，不是 CPU 内存炸
  - 也不是“三个任务并发把同一张卡挤爆”，因为时间线和日志不支持这个解释

#### 不变性测试

- 最初尝试：
  - `sbatch` 直接跑 GPU 版不变性测试
  - 第一次因为 `--wrap` 默认 `/bin/sh` 不认 `source` 失败
  - 第二次 GPU 版进入 JAX 后在 `pc004` 上段错误，未拿到数值
- 后续改为直接 `ssh pc004` 并使用 CPU 后端：
  - `export JAX_PLATFORMS=cpu`
  - 命令成功
- CPU 版测试结果（`checked_only_1946754_best.pkl`，`test sample_index=0`）：
  - translation:
    - `prediction_abs_error = 0`
  - rotation:
    - `prediction_abs_error = 5.96046448e-08`
  - reflection:
    - `prediction_abs_error = 0`

#### `vb_outfile` 的 3-molecule 预测与不变性分析

- 在 `pc004` 上随机抽取 `vb_outfile test split` 的 structure index：
  - 初始随机 sample indices：`3331, 4150, 5541`
  - 后来纠正：用户要的是 3 个 molecules，不是 3 个 structure samples
- 实际用 3 个 molecules 做预测分析：
  - `5283331_VBSCF`
  - `57416843_VBSCF`
  - `7264_VBSCF`
- 结果已记录在本轮总结第 2 节

#### `vb_outfile` 的 10-molecule 分布分析

- 用 `seed=1` 随机抽取 `10` 个 test molecules
- 统计并输出了：
  - 每个 molecule 的：
    - `num_structures`
    - `focus_count`
    - `mae`
    - `rmse`
    - pooled `spearman`
    - `focus_spearman`
    - `pred_top`
    - `true_top`
    - `top1_hit`
  - 最终 summary：
    - pooled mean/median/min/max
    - focus mean/median/min/max
    - `mae_mean`
    - `rmse_mean`
    - `top1_hit_rate`
    - `focus_minus_pooled_mean`

#### lowmem 训练重提

- 使用 `ssh pc004` 直接提交：
  - `vb_only_lowmem`：
    - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass_lowmem.yaml`
    - `RUN_MODE=vb_only_lowmem`
    - 作业号：`1946830`
  - `unified_checked_vb_lowmem`：
    - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml`
    - `RUN_MODE=unified_checked_vb_lowmem`
    - 作业号：`1946831`
- 当前状态（写本条 handoff 时）：
  - `1946830`：`R`
  - `1946831`：`R`
  - 两个作业都在 `e3vb_train_*.slurm.out` 中出现了首个：
    - `ITERATOR_STAGE ... batch_packed`

### 6. 当前报错/未解决问题

#### A. GPU 版不变性测试仍有段错误

- 用 `sbatch` 在 `pc004` 上直接跑 GPU 版 `e3_o3_property_test.py` 时，出现：
  - `Segmentation fault (core dumped)`
- 这不是模型不满足不变性的证据
- CPU 版测试已经正常跑通并给出数值结果
- 下一轮如果要追 GPU 版段错误，可以单独开调试，但不是当前主线 blocker

#### B. `vb_only` 与 unified 原始正式训练都已知 OOM

- 原始 `vb_only` 正式 config：
  - `1946753`
  - 已知 GPU OOM
- 原始 unified 正式 config：
  - `1946755`
  - 已知 GPU OOM
- 新一轮 `lowmem` config 是否足够解决 OOM，还需要继续盯：
  - `1946830`
  - `1946831`

#### C. `sample.top_mass_focus` 在当前分析链路下不可靠

- 直接从 `materializeUnifiedSamples()` 拿 `top_mass_focus`，对 `vb_outfile` 的 molecule 分析曾得到 `focus_count=0`
- 已经用 `computeTopMassFocusMask(target, ...)` 现场重建 true focus 解决
- 但这说明：
  - 如果后续还要做离线 molecule-level focus 分析，不能盲信 sample 自带的 `top_mass_focus`

#### D. 训练总时长只能给近似墙钟时间

- 由于当前环境里 `sacct` 无法访问 slurm 数据库
- 对 `checked_only_1946754`，只能根据：
  - 提交时间约 `00:58`
  - 日志写完时间约 `08:09`
  近似判断总墙钟 `~7小时`
- 不能给精确到秒的正式 elapsed

### 7. 重要路径、变量名、函数名、参数

#### 新增/重要文件路径

- 低显存 `vb_only` config：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass_lowmem.yaml`
- 低显存 unified config：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml`
- `checked_only` best checkpoint：
  - `/pool1/home/ysxin/E3nn/E3VB/checkpoints/checked_only_1946754_best.pkl`
- `vb_only` best checkpoint：
  - `/pool1/home/ysxin/E3nn/E3VB/checkpoints/vb_only_1946753_best.pkl`
- lowmem 训练日志：
  - `/pool1/home/ysxin/E3nn/E3VB/logs/vb_only_lowmem_1946830.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_1946831.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946830.slurm.out`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946831.slurm.out`

#### 新增/重要函数

- `RegressionMetrics.computeMoleculeMacroSpearman()`
- `RegressionMetrics.summarize()`
- `InvarianceTestRunner.expandedAtomInputs()`
- `NNXCheckpointManager.load()`
- `computeTopMassFocusMask()`

#### 重要字段 / 日志键

- `molecule_macro_spearman`
- `spearman`
- `focus_spearman`
- `focus_count`
- `pred_top`
- `true_top`
- `top1_hit`
- `fixed_bucket_graph_step`
- `fixed_bucket_static_atom_step`
- `fixed_bucket_atom_step`
- `fixed_bucket_atom_edge_step`
- `fixed_bucket_orbital_step`
- `fixed_bucket_rumer_edge_step`
- `fixed_bucket_active_orbital_step`
- `fixed_bucket_active_edge_step`

#### 本轮重要作业号

- 原始 `vb_only`: `1946753`
- 原始 `checked_only`: `1946754`
- 原始 unified: `1946755`
- GPU 不变性测试第一次失败（shell/source 问题）：`1946762`
- GPU 不变性测试第二次段错误：`1946763`
- lowmem `vb_only`: `1946830`
- lowmem unified: `1946831`

### 8. 下一轮 Codex 应该继续做什么

1. 优先盯 `1946830` 和 `1946831`。
   - 重点确认：
     - 是否继续稳定推进 epoch
     - 是否再次在更大的 bucket 上 OOM
     - 是否成功进入 `val/test`
     - 是否产生新的 best checkpoint

2. 如果 lowmem config 仍然 OOM，再继续收紧最贵的 fixed bucket 维度，优先顺序：
   - `fixed_bucket_atom_step`
   - `fixed_bucket_atom_edge_step`
   - `fixed_bucket_orbital_step`
   - `fixed_bucket_rumer_edge_step`
   - `fixed_bucket_active_orbital_step`
   - `fixed_bucket_active_edge_step`
   不要先改 `batch_size`，除非用户明确同意

3. 如果用户继续追问 `checked_only` 的训练总时长，可以尝试在有权限的环境下获取精确 `sacct` / `scontrol show job` 结果；当前 handoff 里只有墙钟近似值

4. 如果用户想正式验收不变性：
   - 用 `e3_o3_property_test.py` 继续在 CPU 上批量抽样
   - 最好对：
     - `checked_outfile`
     - `vb_outfile`
     - unified 未来训练出的 checkpoint
     各抽若干 samples
   - 汇总：
     - 平移误差
     - 旋转误差
     - 反演误差
     的 max/mean/median

5. 如果用户继续做跨分布泛化分析，优先复用本轮脚本思路：
   - 用 `checked` 模型测 `vb`
   - 同样也应该做反向：
     - 用 `vb` 模型测 `checked`
   - 保持输出字段一致：
     - `mae`
     - `rmse`
     - pooled `spearman`
     - `focus_spearman`
     - `top1_hit`

6. 如果后续要把本轮分析沉淀为正式工具，建议新增一个独立脚本，例如：
   - `analyze_cross_dataset_generalization.py`
   把目前在 `ssh pc004 ... python - <<PY` 里做的 molecule-level分析整理成可复用命令行工具
