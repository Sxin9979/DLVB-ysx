# HANDOFF

## 0. 最新更新：active orbital 连续方向与 active-space bucketing

本次更新围绕两个目标：

- 只给 `active orbital / active slot` 显式学习局域 3D 方向，inactive/core 不再逐轨道分方向。
- 参考 `/pool1/home/ysxin/E3nn/E3VB_active_rumer` 的活性空间大小分组思路，在当前 JAX/Grain pipeline 中加入 `bucket_key: active_space`。

### 0.1 Active slot learned direction

已修改：

- `/pool1/home/ysxin/E3nn/E3VB/model/orbital_projection.py`
- `/pool1/home/ysxin/E3nn/E3VB/model/end_to_end.py`

核心语义：

- 原来的 `ActiveSlotMatcher` 四候选匹配没有删除，仍保留：
  - `s`
  - `p/e1`
  - `p/e2`
  - `p/e3`
- 新增一个 active-only continuous direction 分支。
- 每个原子 `i` 上第 `k` 个 active slot 现在会学习自己的方向：

```text
d_ik in R^3
```

方向预测使用：

- atom scalar feature
- atom vector feature
- slot embedding
- active Rumer pairing topology 推出的 role prior
- local frame `e1/e2/e3`

当前实现的主要公式是：

```text
query_ik = MLP([h_i, Emb(k)]) + Linear(q_ik^role)

raw_learned_direction_ik =
    beta_ik,1 * e_i1
  + beta_ik,2 * e_i2
  + beta_ik,3 * e_i3

d_ik =
  normalize(
      raw_learned_direction_ik
    + q_ik^bond * gate_ik^bond * e_i3
    + q_ik^lp   * gate_ik^lp   * l_ik
  )

p_ik,c = V_i,c dot d_ik

x_ik^dir = MLP([h_i, p_ik, query_ik, q_ik^role])

x_ik^active = x_ik^old_slot_matching + x_ik^dir
```

其中：

- `q_ik^role = [bond, lone_pair, radical]`
- `bond` 来自跨原子 active pair
- `lone_pair` 来自同原子 active pair
- `radical/unpaired` 来自没有 active pair incidence 的 active slot
- `l_ik` 是 lone-pair 分支自己从 `e1/e2/e3` 学出的方向，不是手写死方向

方向正交约束也已加入 `EndToEndE3VBModel.forwardWithAux()` 的 auxiliary：

```text
L_dir_orth =
  sum_i sum_{k != l} m_ik m_il (d_ik dot d_il)^2
  /
  (sum_i sum_{k != l} m_ik m_il + eps)
```

当前为了低风险接入，它被加进原有 `slot_diversity_weight` 控制的 `slot_diversity_penalty`：

```text
slot_diversity_penalty =
  slot_alpha_diversity_penalty
  + active_direction_orthogonality_penalty
```

注意：

- 这版不是用真实 top-K Rumer 权重做 prior。
- 这版 role prior 来自当前结构的 active Rumer pairing topology，所以训练/推理都可用。
- 当前成键 prior 暂时软偏向 `e3`，适合当前主要希望区分 pi-like active orbital 的设定；如果后续 active space 包含大量 sigma-active orbital，需要再区分 `sigma/pi` bond prior。
- 方向是模型预测量，训练过程中会随参数更新而变化；收敛后推理阶段参数固定，但方向仍随分子几何/结构输入不同而不同。
- `slot_direction` 和 `slot_role_prior` 已放进 `projection_output`，可用于后续 debug/可视化。

### 0.2 Active-space bucketing

已修改：

- `/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py`
- `/pool1/home/ysxin/E3nn/E3VB/utils/config.py`

新增合法配置：

```yaml
training:
  bucketed_batching: true
  bucket_key: active_space
```

作用：

- 参考 `E3VB_active_rumer` 中 `nae*_nao*` bucket 的思路。
- 先按 active-space size 分组：

```text
(num_active_electrons, num_active_orbitals)
```

- 再在同一 active-space bucket 内按 packed graph size 做 coarse bucketing：

```text
total_atoms
total_orbitals
num_structures
```

当前实现细节：

- 对 `ProcessedMoleculeChunk` 和完整 `PackedMoleculeChunk`：
  - `num_active_orbitals` 从 `active_orbital_index` 或 `active_rumer_n_node` 读取。
  - `num_active_electrons` 从 atom feature 第 1 通道求和估计，即 `atom_node_features[:, 1]`。
- 对旧的轻量 `PackedChunkReference`：
  - `num_active_orbitals = total_active_orbitals / num_structures`
  - 如果没有 `num_active_electrons` metadata，则使用 `-1`，这样不会强制 eager load chunk，但仍能按 active orbital 数分组。

预期效果：

- 减少不同活性空间大小混 batch 带来的 padding 浪费。
- 减少 JAX shape 抖动和 recompilation。
- 降低某个大 active-space 样本混入小 batch 导致显存突然炸掉的概率。
- 不保证大分子本身变便宜；它主要降低 padding/recompile/shape-mixing 成本。

推荐组合：

```yaml
training:
  bucketed_batching: true
  bucket_key: active_space
  fixed_bucket_batching: true
```

如果仍然显存不稳，再配合：

```yaml
training:
  max_batch_cost: <budget>
```

### 0.3 验证状态

已通过：

```bash
python -m py_compile data/grain_pipeline.py utils/config.py model/orbital_projection.py model/end_to_end.py
```

2026-05-30 已在 `pc001` 上使用训练环境 `e3vb_jax` 完成 JAX forward smoke test：

```bash
ssh pc001
cd /pool1/home/ysxin/E3nn/E3VB
source /pool1/home/ysxin/miniconda3/etc/profile.d/conda.sh
conda activate e3vb_jax
```

环境确认：

```text
jax 0.4.31
backend gpu
devices [CudaDevice(id=0)]
```

Projection stack cross-atom active pair smoke test 通过：

```text
orbital_feature (4, 7)
orbital_feature_q3_flipped (4, 7)
slot_direction (3, 3, 3)
slot_direction_norm [[1. 0. 0.]
                     [1. 0. 0.]
                     [0. 0. 0.]]
slot_role_prior shape (3, 3, 3)
active_capacity [1 1 0]
```

该测试中两个不同原子的 active slot 形成 pair，role prior 正确给到 `bond`：

```text
slot_role_prior atom0 slot0 = [1, 0, 0]
slot_role_prior atom1 slot0 = [1, 0, 0]
```

Projection stack same-atom active pair smoke test 通过：

```text
slot_role_prior [[[0. 1. 0.]
                  [0. 1. 0.]
                  [0. 0. 0.]]
                 [[0. 0. 0.]
                  [0. 0. 0.]
                  [0. 0. 0.]]]
slot_direction_norm [[1. 1. 0.]
                     [0. 0. 0.]]
active_capacity [2 0]
orbital_feature (2, 7)
```

该测试中同一个原子的两个 active slot 形成 pair，role prior 正确给到 `lone_pair`，两个有效 slot 都产生单位方向。

2026-05-30 已在 `pc001` 上补充 projection-level 不变性/等变性测试，覆盖刚新增的 active direction 分支：

```text
== translation
orbital_feature_abs 0.0
orbital_feature_q3_flipped_abs 0.0
slot_role_prior_abs 0.0
slot_direction_equiv_abs 0.0

== rotation
orbital_feature_abs 2.9802322387695312e-08
orbital_feature_q3_flipped_abs 2.9802322387695312e-08
slot_role_prior_abs 0.0
slot_direction_equiv_abs 1.1920928955078125e-07

== reflection
orbital_feature_abs 0.0
orbital_feature_q3_flipped_abs 0.0
slot_role_prior_abs 0.0
slot_direction_equiv_abs 0.0
```

结论：

- `orbital_feature` 对平移/旋转/反射保持不变。
- `slot_role_prior` 对几何变换不变。
- `slot_direction` 对旋转/反射按预期等变，误差在 `1e-7` 量级。

同日已在 `pc001` 上运行完整端到端 `e3_o3_property_test.py`：

```bash
python e3_o3_property_test.py --config tmp_unified_smoke_config.yaml --split train --sample_index 0 --seed 0
```

结果：

```text
translation:
  scalar_invariance_abs_error: 1.87754631e-06
  vector_equivariance_abs_error: 1.31130219e-05
  prediction_abs_error: 3.72529030e-08

rotation:
  scalar_invariance_abs_error: 2.50339508e-06
  vector_equivariance_abs_error: 2.09808350e-05
  prediction_abs_error: 2.23517418e-08

reflection:
  scalar_invariance_abs_error: 7.15255737e-07
  vector_equivariance_abs_error: 7.15255737e-07
  prediction_abs_error: 1.04308128e-07
```

说明：

- 最终 scalar prediction 在 E(3) 与 reflection 下保持不变，误差约 `2e-8` 到 `1e-7`。
- atom encoder 中间特征的 absolute error 在 `1e-6` 到 `2e-5`，相对误差有时较大主要是因为局部参考值接近 0。
- 测试使用随机初始化模型，未加载 checkpoint。

仍建议后续补充：

- 用真实训练配置测试 `bucket_key: active_space` 是否减少 batch shape 波动。
- 观察是否触发额外 JAX recompilation。

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
  - loss 侧已改成 molecule-local averaging，再继续做 runtime / ablation 验证。

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

8. `utils/losses.py` 已完成 top-mass objective 的 molecule-local averaging：
   - `focus_regression_loss`
   - `focus_rank_loss`
   - `focus_tail_rank_loss`
   - 现在都是先按 molecule 聚合，再对 valid molecule 平均

### 当前尚未完成
本轮**没有继续实现**以下内容：

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

## 2026-05-05 18:35 本轮工作总结

### 1. 本轮目标
- 继续定位 `vb_only` / `unified` 在 4090 上的 GPU OOM 原因，判断是 `train` 还是 `val/test` 引起。
- 在不降低 `train batch_size=8` 的前提下，测试两类最小改动：
  - 给 `val/test` 单独设置更小的 `eval_batch_size`
  - 下调 unified 训练侧 `max_tail_samples_per_molecule`
- 将当前代码和配置先保存到 git 远端，避免上下文切换时丢失状态。

### 2. 已完成内容
- 完成 git 提交与推送：
  - 本地分支：`clean`
  - commit：`21e6045`
  - message：`Add top-mass unified training pipeline and diagnostics`
  - 已推送到 `origin/clean`
- 在配置系统中增加了 `training.eval_batch_size`，使 `val/test` 可独立于 `train batch_size`。
- 修改训练主循环：
  - `train` 继续使用 `training.batch_size`
  - `val/test` 改用 `training.eval_batch_size`
  - `splitBatchCount()` 也同步按 train/eval 两套 batch size 计算
  - 日志 `BATCHING` 新增 `train_molecules_per_batch` 和 `eval_molecules_per_batch`
- unified 低显存配置先后做了两轮测试：
  - 第一轮：`eval_batch_size=1`，`max_tail_samples_per_molecule=96`
  - 第二轮：`eval_batch_size=1`，`max_tail_samples_per_molecule=48`
- vb 低显存配置也增加了 `eval_batch_size=1`，并重新提交 `vb_only` 训练。
- 通过日志确认：仅把 `val/test batch size` 降到 `1` 后，`val` 确实被拆成很多单分子 batch，但 unified 仍然在训练阶段 OOM，说明主矛盾已经转移到 train 侧。

### 3. 修改/新增/检查过的文件
- 修改：
  - [utils/config.py](/pool1/home/ysxin/E3nn/E3VB/utils/config.py:23)
  - [main.py](/pool1/home/ysxin/E3nn/E3VB/main.py:839)
  - [config_e2e_checked_vb_unified_topmass_lowmem.yaml](/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml:52)
  - [config_e2e_vboutfile_topmass_lowmem.yaml](/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass_lowmem.yaml:48)
  - [HANDOFF.md](/pool1/home/ysxin/E3nn/E3VB/HANDOFF.md:1)
- 检查过：
  - [logs/unified_checked_vb_lowmem_1946831.log](/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_1946831.log:1)
  - [logs/e3vb_train_1946831.slurm.err](/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946831.slurm.err:1)
  - [logs/unified_checked_vb_lowmem_evalbs1_mem48_1946846.log](/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_evalbs1_mem48_1946846.log:1)
  - [logs/e3vb_train_1946846.slurm.out](/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946846.slurm.out:1)
  - [logs/e3vb_train_1946846.slurm.err](/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946846.slurm.err:1)
  - [logs/vb_only_lowmem_1946830.log](/pool1/home/ysxin/E3nn/E3VB/logs/vb_only_lowmem_1946830.log:1)
  - [logs/e3vb_train_1946830.slurm.err](/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946830.slurm.err:1)
  - [logs/vb_only_lowmem_evalbs1_1946851.log](/pool1/home/ysxin/E3nn/E3VB/logs/vb_only_lowmem_evalbs1_1946851.log:1)
  - [logs/unified_checked_vb_lowmem_evalbs1_tail48_1946853.log](/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_evalbs1_tail48_1946853.log:1)

### 4. 关键实现逻辑
- `TrainingConfig` 新增字段：
  - `eval_batch_size: int`
  - 默认值为 `training.batch_size`
  - 允许 `val/test` 单独设成 `1` 或更小值
- `main.py` 当前行为：
  - `training=True` 时：
    - `batch_size = self.config.training.batch_size`
  - `training=False` 时：
    - `batch_size = self.config.training.eval_batch_size`
  - 该逻辑同时用于：
    - `splitBatchCount()`
    - `createMoleculeIterator()`
    - `createIterator()`
- unified 低显存实验得到的关键结论：
  - `1946846` 中 `eval_batch_size=1` 已经生效，`val` 从原来的 2~3 个大 batch 变成 `00000/00022` 这类多单分子 batch。
  - 但 `1946846` 仍在 `runTrainBatch -> trainStepTopMass` 中 OOM，且请求显存仍为 `18.16 GiB`。
  - 因此“只缩 `val/test batch size`”不能根治 unified OOM；统一训练的主矛盾已经是 train 侧 shape / train 侧 sampled structures 数。
- `vb_only_lowmem` 的结论：
  - `1946830` 能跑到 `epoch=323` 左右，说明 lowmem bucket 方案有效，但仍会在 train 侧 OOM。
  - 该轮最终额外申请 `3.67 GiB` GPU 显存失败，比旧版 `5.41 GiB` 更好，但仍不稳定。
- 当前判断：
  - unified 后续最值得先动的不是继续只调 eval，而是继续压 train 侧 `max_tail_samples_per_molecule`，并视情况把 `top_mass_sample_strategy` 从 `focus_plus_mixed_tail` 进一步收紧。

### 5. 已运行命令与结果
- git 状态/远端检查后，完成本地提交与推送：
  - `git commit -m "Add top-mass unified training pipeline and diagnostics"`
  - `git push origin clean`
  - 结果：`origin/clean` 更新到 `21e6045`
- unified，`eval_batch_size=1`，`--mem=100G`：
  - 作业：`1946844`
  - 结果：排队，后续未作为主对照使用
- unified，`eval_batch_size=1`，`--mem=48G`：
  - 作业：`1946846`
  - 结果：失败
  - 失败点：
    - 运行到约 `epoch=50`
    - `val` 已拆成单分子 batch
    - 最终仍在 train 阶段 OOM
    - 报错：`Failed to allocate request for 18.16GiB`
- vb，低显存旧方案：
  - 作业：`1946830`
  - 结果：失败
  - 失败点：
    - 跑到 `epoch=323` 左右
    - train 阶段 OOM
    - 报错：`Failed to allocate request for 3.67GiB`
- vb，`eval_batch_size=1`：
  - 作业：`1946851`
  - 当前状态：仍在运行
  - 最新观察：
    - `epoch=159` 的 `val` 已变成 `batch=00000/00008`
    - 说明 `eval_batch_size=1` 生效
- unified，`eval_batch_size=1` 且 `max_tail_samples_per_molecule=48`：
  - 作业：`1946853`
  - 当前状态：运行中
  - 最新日志已确认：
    - `train_molecules_per_batch=8`
    - `eval_molecules_per_batch=1`
    - `max_tail_samples_per_molecule=48`

### 6. 当前报错/未解决问题
- 未解决的核心问题仍是 unified 的 GPU OOM：
  - 即使 `eval_batch_size=1`，`1946846` 仍然在 train 阶段 OOM。
  - 因此 eval 不是唯一瓶颈，train 侧 sampled structure 数和 shape 多样性仍过重。
- `vb_only` 也未彻底稳定：
  - `1946830` 说明 lowmem bucket 只能缓解、不能根治。
  - 需要继续观察 `1946851` 是否能越过旧炸点。
- `pc004` 调度状态不稳定：
  - 之前 `1946846` 一度出现 `ReqNodeNotAvail, UnavailableNodes:pc004`
  - 当前 `1946851` / `1946853` 已在 `pc004` 上运行
- 当前工作树相对 `21e6045` 仍有未提交改动：
  - `utils/config.py`
  - `main.py`
  - `config_e2e_checked_vb_unified_topmass_lowmem.yaml`
  - `config_e2e_vboutfile_topmass_lowmem.yaml`
  - 这些是本轮新增的 `eval_batch_size` / `tail48` 相关修改，尚未再提交到 git。

### 7. 重要路径、变量名、函数名、参数
- git：
  - 分支：`clean`
  - 最近已推送 commit：`21e6045`
- 新增配置字段：
  - `training.eval_batch_size`
- 仍在重点使用的训练参数：
  - `training.batch_size`
  - `training.eval_batch_size`
  - `training.top_mass_sample_strategy`
  - `training.max_tail_samples_per_molecule`
  - `training.dataset_sampling_strategy`
  - `training.fixed_bucket_*`
- 关键代码位置：
  - `TrainingConfig`： [utils/config.py](/pool1/home/ysxin/E3nn/E3VB/utils/config.py:16)
  - `createTrainingConfig()`： [utils/config.py](/pool1/home/ysxin/E3nn/E3VB/utils/config.py:316)
  - `splitBatchCount()`： [main.py](/pool1/home/ysxin/E3nn/E3VB/main.py:835)
  - `runSplit()`： [main.py](/pool1/home/ysxin/E3nn/E3VB/main.py:987)
  - `runTrainBatch()`： [main.py](/pool1/home/ysxin/E3nn/E3VB/main.py:720)
  - `createMoleculeIterator()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:1841)
- 当前主要配置文件：
  - unified 低显存： [config_e2e_checked_vb_unified_topmass_lowmem.yaml](/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml:1)
  - vb 低显存： [config_e2e_vboutfile_topmass_lowmem.yaml](/pool1/home/ysxin/E3nn/E3VB/config_e2e_vboutfile_topmass_lowmem.yaml:1)
- 当前重点作业与日志：
  - `1946830`：
    - [vb_only_lowmem_1946830.log](/pool1/home/ysxin/E3nn/E3VB/logs/vb_only_lowmem_1946830.log:1)
    - [e3vb_train_1946830.slurm.err](/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946830.slurm.err:1)
  - `1946846`：
    - [unified_checked_vb_lowmem_evalbs1_mem48_1946846.log](/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_evalbs1_mem48_1946846.log:1)
    - [e3vb_train_1946846.slurm.err](/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946846.slurm.err:1)
  - `1946851`：
    - [vb_only_lowmem_evalbs1_1946851.log](/pool1/home/ysxin/E3nn/E3VB/logs/vb_only_lowmem_evalbs1_1946851.log:1)
  - `1946853`：
    - [unified_checked_vb_lowmem_evalbs1_tail48_1946853.log](/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_evalbs1_tail48_1946853.log:1)

### 8. 下一轮 Codex 应该继续做什么
- 第一优先级：继续盯运行中的两个作业
  - `1946851`：确认 `vb_only` 在 `eval_batch_size=1` 下能否越过旧炸点 `epoch~320`
  - `1946853`：确认 unified 在 `tail48 + eval_batch_size=1` 下能否越过 `1946846` 的旧炸点 `epoch~50`
- 第二优先级：如果 `1946853` 仍然 OOM，下一步优先继续压 train 侧，而不是继续只调 eval
  - 候选方向：
    - `max_tail_samples_per_molecule: 48 -> 32`
    - `top_mass_sample_strategy: focus_plus_mixed_tail -> focus_plus_top_tail`
  - 用户当前明确不希望通过减小 `train batch_size` 解决问题，因此先不动 `batch_size=8`
- 第三优先级：如果 `1946851` 明显比 `1946830` 更稳，说明 `eval_batch_size=1` 对 vb 是有效的，应将这套策略保留
- 第四优先级：在确认新参数有效后，把本轮未提交的代码/配置变更再做一次 git 提交与推送
  - 重点包括：
    - `utils/config.py`
    - `main.py`
    - `config_e2e_checked_vb_unified_topmass_lowmem.yaml`
    - `config_e2e_vboutfile_topmass_lowmem.yaml`
- 第五优先级：如果 unified 即使 `tail32` 仍炸，则需要进入更系统的方案设计
  - `budget-based batching`
  - `chunked_full_molecule eval`
  - 或者按 epoch 分段续训、周期性重启进程，减少长期 shape/缓存累积带来的显存压力

## 2026-05-05 19:02 本轮工作总结

### 1. 本轮目标
- 用户明确要求：
  - 不再沿着 `max_tail_samples_per_molecule: 96 -> 48 -> 32` 这种短期试参路线继续兜圈；
  - 先把 unified 低显存配置中的 `max_tail_samples_per_molecule` 改回 `96`；
  - 然后落地一个适用于未来继续加入更多 dataset 的长期方案；
  - 最后用新版 unified 配置重新提交一轮训练。
- 基于上一轮日志的判断，本轮的核心目标不是继续压 `eval_batch_size`，而是解决 unified train side 没有明确显存上界的问题：
  - `1946853` 已经确认 `tail48` 仍然在 `runTrainBatch -> trainStepTopMass` 中 OOM；
  - 因此长期解必须落在 train side batching / sampling，而不是只改 eval。

### 2. 已完成内容
- 已把 unified 低显存配置恢复为：
  - `max_tail_samples_per_molecule: 96`
  - 同时保留上一轮已经加进去的 `eval_batch_size: 1`
- 已在配置层新增长期方案所需的两个预算参数：
  - `training.max_batch_cost`
  - `training.max_structure_cost_per_molecule`
- 已在 packed chunk metadata 层增加预算估计所需字段，写入 `PackedChunkReference`：
  - `num_focus_structures`
  - `static_num_atoms`
  - `total_atom_edges`
  - `total_rumer_edges`
  - `total_active_orbitals`
  - `total_active_rumer_edges`
- 已在 top-mass sampling 层增加 per-molecule 预算裁剪逻辑：
  - 新增 `capTopMassTailSamplesByBudget()`
  - 该逻辑保证 focus structures 总是优先保留，只对 tail capacity 做预算收缩
- 已在 `GrainPipeline` 中落地 budget-based batching 第一版：
  - 新增 chunk / molecule 的 heuristic cost estimator
  - 新增基于 cost metadata 的 whole-molecule cost 估计
  - 新增 `buildOrderedGroupIndices()`，先保留 dataset-aware / bucket-aware 顺序
  - 新增 `buildBudgetedGroupedBatchIndices()`，在既有顺序上做 cost-budget batching
  - 新增 `materializeSampledMoleculeGroup()`，把 top-mass sampling 与 per-molecule budget 收紧合并到一个统一入口
  - 新增 `emitPackedMoleculeBatch()`，统一 whole-molecule batch 的 pack + logging
- 已在 `main.py` 中把新预算参数接到训练主循环：
  - `splitBatchCount()` 在 training path 下支持按 budget 估算 batch 数
  - `runSplit()` 在 molecule-group 训练路径下支持把 `max_batch_cost` / `max_structure_cost_per_molecule` 传给 pipeline
  - `BATCHING` runtime 日志新增：
    - `train_molecules_per_batch`
    - `eval_molecules_per_batch`
    - `max_batch_cost`
    - `max_structure_cost_per_molecule`
- 已完成语法级验证：
  - `python3 -m py_compile` 对以下文件全部通过：
    - `utils/config.py`
    - `data/schema.py`
    - `data/processor.py`
    - `utils/top_mass.py`
    - `data/grain_pipeline.py`
    - `main.py`
- 已按新版代码提交流程作业：
  - `1946856`：重建 unified preprocess / packed cache
  - `1946857`：依赖 `1946856 afterok` 的 unified train 作业

### 3. 修改/新增/检查过的文件
- 已修改：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/config.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/schema.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/processor.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py`
  - `/pool1/home/ysxin/E3nn/E3VB/main.py`
  - `/pool1/home/ysxin/E3nn/E3VB/HANDOFF.md`
- 已检查但未修改的关键文件：
  - `/pool1/home/ysxin/E3nn/E3VB/submit_preprocess.sbatch`
  - `/pool1/home/ysxin/E3nn/E3VB/submit_train.sbatch`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_evalbs1_tail48_1946853.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946853.slurm.err`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_lowmem_evalbs1_mem48_1946846.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/vb_only_lowmem_evalbs1_1946851.log`

### 4. 关键实现逻辑
- 这轮长期方案的核心思想是：
  - 不再把 training stability 绑定在固定 `batch_size=8` 和固定 `max_tail_samples_per_molecule`
  - 而是给每个 molecule 和每个 whole-molecule batch 显式定义 complexity budget
- `utils/config.py`
  - `TrainingConfig` 新增：
    - `eval_batch_size`
    - `max_batch_cost`
    - `max_structure_cost_per_molecule`
  - `createTrainingConfig()` 对 budget 参数做正整数校验：
    - 如果提供，则必须大于 `0`
- `data/schema.py`
  - `PackedChunkReference` 现在不再只是轻量路径引用；
  - 它同时携带离线统计出的 chunk cost metadata，供 runtime 预算估计使用；
  - 这样未来再加 dataset 时，只要走统一 preprocess/export，就会自然获得同样的 budget metadata。
- `data/processor.py`
  - `savePackedChunk()` 在保存每个 packed chunk 时，把上述 metadata 一并写入；
  - 注意：旧 packed cache 仍然可读，但旧 cache 不会自动补齐这些新字段；
  - 因此为了让预算预估更准确，本轮专门重新提交了一次 `FORCE_REBUILD=1` 的 preprocess。
- `utils/top_mass.py`
  - 新增 `capTopMassTailSamplesByBudget()`：
    - 输入是 `num_structures`、`num_focus_structures`、`structure_cost`、`max_tail_samples`、`max_structure_cost`
    - 输出是一个“当前 molecule 实际允许保留的 tail 数”
  - 逻辑上：
    - 先保住 focus structures
    - 剩余预算才分给 tail
    - 如果 molecule 本身已经太大，则自动把 tail cap 收得更紧
- `data/grain_pipeline.py`
  - 新增 `estimateChunkCost()`：
    - 使用 `static_num_atoms + total_atoms + total_atom_edges + total_orbitals + total_rumer_edges + total_active_orbitals + total_active_rumer_edges`
    - 对 dynamic 部分按 selected structure ratio 线性缩放
  - 新增 `estimateMoleculeGroupCost()`：
    - 按 chunk 汇总 whole-molecule cost
  - 新增 `estimateMoleculeGroupSelectedStructures()`：
    - 对还未真正 materialize 的 molecule，先基于 metadata 估计在当前 top-mass policy 下大概会贡献多少 structures
  - 新增 `buildOrderedGroupIndices()`：
    - 先维持原有 dataset-balanced / bucketed 的顺序语义
    - 预算重组是在这个顺序之上进行，而不是完全推翻 dataset balancing
  - 新增 `buildBudgetedGroupedBatchIndices()`：
    - 以 `max_batch_cost` 为硬预算构造 whole-molecule batches
    - 当 `current_cost + estimated_cost > max_batch_cost` 时切 batch
    - 仍然保留 `batch_size` 作为 molecule 数上限
  - 新增 `materializeSampledMoleculeGroup()`：
    - 对单个 molecule 先加载 chunk group
    - 再生成 focus mask
    - 再根据 `max_structure_cost_per_molecule` 得到 budget-aware tail cap
    - 最后调用 `selectTopMassStructureIndices()`
  - `createMoleculeIterator()` 现在支持两条路径：
    - 如果 `max_batch_cost is None`：走原来的 molecule batching 逻辑
    - 如果 `max_batch_cost > 0`：走新的 budget-based batching 逻辑
- `main.py`
  - `splitBatchCount()` 已支持 budget path，因此 epoch 内部的 batch 计数日志不再明显失真
  - `runSplit()` 会把以下参数传入新版 iterator：
    - `eval_batch_size`
    - `max_batch_cost`
    - `max_structure_cost_per_molecule`
  - `BATCHING` 日志现在能直接看出：
    - train / eval molecule batch size
    - 是否启用了 budget batching
    - 当前预算值是多少

### 5. 已运行命令与结果
- 读取上下文与要求：
  - `sed -n '1,220p' /pool1/home/ysxin/E3nn/E3VB/HANDOFF.md`
  - `sed -n '1,260p' /pool1/home/ysxin/E3nn/E3VB/AGENTS.md`
  - 结果：确认必须继续使用项目内一致的技术口径与语言要求
- 检查 `1946853` / `1946846` / `1946851` 相关日志：
  - 结果：
    - `1946853` 已炸；
    - `1946853` 报错为 `RESOURCE_EXHAUSTED: Failed to allocate request for 17.96GiB`
    - `1946846` 的 train-side OOM 为 `18.16GiB`
    - 说明 `96 -> 48` 只是轻微降低峰值申请，并未形成长期解决方案
- 代码一致性 / 定位检查：
  - 使用多次 `sed` / `rg` 查看：
    - `utils/config.py`
    - `data/schema.py`
    - `data/processor.py`
    - `data/grain_pipeline.py`
    - `utils/top_mass.py`
    - `main.py`
  - 结果：确认 budget 方案可以直接嫁接在现有 unified molecule batching 主干上
- 语法校验：
  - `python3 -m py_compile /pool1/home/ysxin/E3nn/E3VB/utils/config.py /pool1/home/ysxin/E3nn/E3VB/data/schema.py /pool1/home/ysxin/E3nn/E3VB/data/processor.py /pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py /pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py /pool1/home/ysxin/E3nn/E3VB/main.py`
  - 结果：成功，无编译级错误
- 提交 preprocess：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml FORCE_REBUILD=1 sbatch -p pc -w pc004 /pool1/home/ysxin/E3nn/E3VB/submit_preprocess.sbatch`
  - 结果：`Submitted batch job 1946856`
- 提交 train：
  - `CONFIG_PATH=/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml RUN_MODE=unified_checked_vb_budget96 sbatch --dependency=afterok:1946856 -p pc -w pc004 /pool1/home/ysxin/E3nn/E3VB/submit_train.sbatch`
  - 结果：`Submitted batch job 1946857`
- 备注：
  - 两个 `sbatch` 都是以新版本地工作树直接提交；
  - 这些改动目前仍未提交到 git。

### 6. 当前报错/未解决问题
- 当前还没有验证新版 budget-based batching 在真实 GPU 训练里是否已经彻底解决 unified train-side OOM；
  - 本轮只完成了代码落地和作业提交；
  - 下一轮必须盯 `1946856` 和 `1946857` 的运行状态。
- 旧 packed cache 仍然存在兼容性问题：
  - 代码可以兼容旧 `PackedChunkReference`
  - 但旧 cache 没有新的 cost metadata
  - 因此预算预估的精度会变差
  - 这也是为什么本轮必须先重建 packed cache
- `config_e2e_vboutfile_topmass_lowmem.yaml` 仍有上一轮留下的 `eval_batch_size` 相关未提交改动；
  - 本轮没有继续动 vb-only 配置逻辑
  - 但 `git status` 里它仍然是 modified
- 当前工作树仍是 dirty：
  - `HANDOFF.md`
  - `config_e2e_checked_vb_unified_topmass_lowmem.yaml`
  - `data/grain_pipeline.py`
  - `data/processor.py`
  - `data/schema.py`
  - `main.py`
  - `utils/config.py`
  - `utils/top_mass.py`
  - 以及若干临时文件 / 临时目录

### 7. 重要路径、变量名、函数名、参数
- 关键配置文件：
  - unified 低显存： [config_e2e_checked_vb_unified_topmass_lowmem.yaml](/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem.yaml:1)
- 关键新增训练参数：
  - `training.eval_batch_size`
  - `training.max_batch_cost`
  - `training.max_structure_cost_per_molecule`
  - `training.max_tail_samples_per_molecule`
  - `training.dataset_sampling_strategy`
  - `training.top_mass_sample_strategy`
- 关键新增 / 重要函数：
  - `capTopMassTailSamplesByBudget()`： [utils/top_mass.py](/pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py:175)
  - `estimateChunkCost()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:1417)
  - `estimateMoleculeGroupCost()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:1460)
  - `estimateMoleculeGroupSelectedStructures()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:1488)
  - `buildOrderedGroupIndices()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:1520)
  - `buildBudgetedGroupedBatchIndices()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:1590)
  - `materializeSampledMoleculeGroup()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:2166)
  - `emitPackedMoleculeBatch()`： [data/grain_pipeline.py](/pool1/home/ysxin/E3nn/E3VB/data/grain_pipeline.py:2045)
  - `splitBatchCount()`： [main.py](/pool1/home/ysxin/E3nn/E3VB/main.py:835)
  - `runSplit()`： [main.py](/pool1/home/ysxin/E3nn/E3VB/main.py:987)
- 关键 schema / metadata：
  - `PackedChunkReference.num_focus_structures`
  - `PackedChunkReference.static_num_atoms`
  - `PackedChunkReference.total_atom_edges`
  - `PackedChunkReference.total_rumer_edges`
  - `PackedChunkReference.total_active_orbitals`
  - `PackedChunkReference.total_active_rumer_edges`
- 本轮关键作业：
  - preprocess：`1946856`
  - train：`1946857`
- 本轮关键日志路径：
  - preprocess 运行日志：`/pool1/home/ysxin/E3nn/E3VB/logs/preprocess_1946856.log`
  - preprocess slurm out：`/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_preprocess_1946856.slurm.out`
  - preprocess slurm err：`/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_preprocess_1946856.slurm.err`
  - train 运行日志：`/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_budget96_1946857.log`
  - train slurm out：`/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946857.slurm.out`
  - train slurm err：`/pool1/home/ysxin/E3nn/E3VB/logs/e3vb_train_1946857.slurm.err`

### 8. 下一轮 Codex 应该继续做什么
- 第一优先级：盯 `1946856`
  - 确认新版 preprocess 是否成功完成
  - 确认 unified 的 processed / packed cache 是否已经用新版代码重建
  - 如果失败，优先检查：
    - `preprocess_1946856.log`
    - `e3vb_preprocess_1946856.slurm.err`
- 第二优先级：盯 `1946857`
  - 确认它是否因 `afterok:1946856` 正常启动
  - 启动后第一时间检查 `BATCHING` 日志中是否出现：
    - `max_batch_cost=300000`
    - `max_structure_cost_per_molecule=60000`
    - `train_molecules_per_batch=8`
    - `eval_molecules_per_batch=1`
  - 同时确认 unified 是否真的走到了 budget-based iterator path
- 第三优先级：如果 `1946857` 仍然 OOM
  - 不要第一反应再改 `eval_batch_size`
  - 先看报错发生在：
    - 训练前几 batch 就炸
    - 还是若干 epoch 后 shape 演化才炸
  - 优先继续收紧预算，而不是继续直接砍全局 tail：
    - `max_batch_cost: 300000 -> 240000` 或更低
    - `max_structure_cost_per_molecule: 60000 -> 45000` 或更低
  - 只有当 budget 收紧后仍然不稳，再考虑改：
    - `top_mass_sample_strategy`
    - `mixed_tail_top_fraction`
- 第四优先级：如果 `1946857` 能稳定跑过旧炸点
  - 记录：
    - 训练前几 epoch 的 batch 数变化
    - 是否出现明显更小 / 更稳定的 peak allocation
    - val/test 是否仍然保持合理吞吐
  - 然后考虑把这轮预算方案推广到 vb-only / checked-only lowmem 配置
- 第五优先级：在作业确认稳定后，清理并提交本轮代码
  - 重点文件：
    - `config_e2e_checked_vb_unified_topmass_lowmem.yaml`
    - `utils/config.py`
    - `data/schema.py`
    - `data/processor.py`
    - `utils/top_mass.py`
    - `data/grain_pipeline.py`
    - `main.py`
    - `HANDOFF.md`

## 2026-05-06 17:25 本轮工作总结

### 1. 本轮目标
- 读取并梳理当前仓库 `/pool1/home/ysxin/E3nn/E3VB` 的主训练链路、loss/metric 定义、top-mass 任务设计与近期日志表现。
- 结合用户的真实目标“前 98% 累计权重结构的权重要回归准确”给出下一步实验方向，不做代码逻辑修改，只做分析与 handoff 记录。

### 2. 已完成内容
- 确认当前项目不是旧版 PyTorch 目录，而是 JAX + Flax NNX 的统一训练版本，入口为 `main.py`，模型主干在 `model/end_to_end.py`。
- 过了一遍主配置与近期关键日志，区分了三类实验路线：
  - 早期 vb-only / edge-content / 全局 MAE 强基线。
  - vb-only top-mass / lowmem / fixed-bucket 路线。
  - checked + vb unified / top-mass / balanced sampling / lowmem 路线。
- 确认目前“最好看的全局 MAE”仍来自早期单数据集基线，而更真实的 unified 设置会显著更难，且 train-val gap 主要表现为 focus 集合选择和 tail 误报问题，而不是简单欠拟合。
- 把以下概念在代码里逐一定义清楚并向用户解释：
  - `recipe`：一整套训练配方，不是单独函数，主要体现在配置文件里。
  - `top_mass_regression_weight`：top-mass loss 中 focus 集合回归项的权重。
  - `top_mass_ranking_weight`：top-mass loss 中排序项总权重。
  - `focus_precision`：预测出来的 focus 集合有多“干净”。
  - `macro_focus_weighted_mae`：跨数据集宏平均的 focus 集合加权 MAE。
- 确认当前用户目标已经收敛为：
  - 第一优先级不是“排序更漂亮”，而是“真实前 98% 结构权重要回归准确”。
- 基于这个目标给出建议：
  - 把 early-stop / best checkpoint monitor 从 `macro_focus_priority_score` 切到 `macro_focus_weighted_mae`。
  - 减弱排序项的主导地位，优先优化 focus 内回归。
  - 保留 `focus_precision` 与 `tail_false_positive_rate` 作为护栏指标，而非第一目标。

### 3. 修改/新增/检查过的文件
- 本轮未修改训练代码、模型代码、配置文件。
- 本轮检查过的关键文件：
  - `/pool1/home/ysxin/E3nn/E3VB/main.py`
  - `/pool1/home/ysxin/E3nn/E3VB/model/end_to_end.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/metrics.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/config.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/xmo_builder.py`
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem_excl4.yaml`
  - `/pool1/home/ysxin/E3nn/E3VB/experiments/configs/config_e2e_edge_content.yaml`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/edgecontent_seed0_1938902.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/weightedmid_seed0_pc001_1942997.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/vboutfile_fixedbucket_20260501_1946155.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/vboutfile_topmass_1946552.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/checked_only_1946754.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/vb_only_lowmem_evalbs1_1946851.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/unified_checked_vb_budget96_1946857.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/unified_excl4_1947268.log`
  - `/pool1/home/ysxin/E3nn/E3VB/logs/checked_recipe1946851_pc004_48g_1947276.log`
- 本轮唯一实际写入的文件是当前 `HANDOFF.md`，用于补充本段总结。

### 4. 关键实现逻辑
- 当前配置 `use_top_mass_objective: true` 时，训练走的是 `trainStepTopMass()`，不是普通全局 MAE 路线。
- 顶层训练 loss 在 `main.py` 中由 `topMassObjectiveLoss(...) + slot_diversity_weight * slot_diversity_penalty` 组成。
- `topMassObjectiveLoss()` 的核心结构：
  - `focus_regression_loss`
    - 只在 true focus 集合上做 `moleculeLocalWeightedMae`
    - 是当前最接近用户目标“前 98% 权重回归准确”的项
  - `tail_regression_loss`
    - 对 tail 结构做低优先级回归/抑制
  - `focus_rank_loss`
    - 只在 true focus 集合内部做 pairwise hinge ranking
  - `focus_tail_rank_loss`
    - 强制 focus 结构整体高于 tail 结构
- true focus 集合的定义不是“固定 top-k”，而是：
  - 先按同一分子内真实 `normalized_weight` 从大到小排序
  - 取最短前缀，使累计质量达到 `focus_cumulative_mass=0.98`
  - 这一定义在 `utils/top_mass.py::computeTopMassFocusMask()`，并在 `data/xmo_builder.py` preprocess 时写入 `top_mass_focus_mask`
- `focus_precision` 的意义：
  - 在预测为 focus 的结构中，有多少真的属于 true focus 集合
  - 用于衡量“预测 focus 集合是否干净”
- `tail_false_positive_rate` 的意义：
  - 在真实 tail 结构中，有多少被误判为预测 focus
  - 这是 focus 集合“混入尾部结构”的直接刻画
- `macro_focus_weighted_mae` 的意义：
  - 先按数据集分别算 `focus_weighted_mae`
  - 再对各个数据集做宏平均
  - 更适合未来持续增量数据集的统一训练场景，避免大数据集完全主导 monitor
- 当前 unified `excl4` 配置中：
  - `validation_monitor` 仍是 `macro_focus_priority_score`
  - 这个 monitor 不是纯回归指标，而是把 `focus_weighted_mae`、`focus_pair_acc`、`focus_spearman`、`focus_recall`、`focus_precision`、`tail_false_positive_rate` 混成一个综合分数
  - 如果第一目标已经明确为“前 98% 结构权重回归准确”，那么这个 monitor 与目标并不完全一致

### 5. 已运行命令与结果
- `du -sh /pool1/home/ysxin/E3nn/E3VB`
  - 结果约 `19G`，仓库较大，包含代码、cache、checkpoints、logs。
- `rg --files /pool1/home/ysxin/E3nn/E3VB`
  - 用于识别主目录结构与关键文件。
- `sed -n '1,220p' /pool1/home/ysxin/E3nn/E3VB/main.py`
  - 确认训练入口、`trainStep()` / `trainStepTopMass()` 分流逻辑。
- `sed -n '1,260p' /pool1/home/ysxin/E3nn/E3VB/model/end_to_end.py`
  - 确认端到端模型是 atom encoder -> orbital / slot -> Rumer encoder 的统一架构。
- `sed -n '1,260p' /pool1/home/ysxin/E3nn/E3VB/utils/losses.py`
  - 确认 `weightedMae`、`moleculeLocalWeightedMae`、`pairwiseRankLoss`、`crossSetMarginLoss`、`topMassObjectiveLoss` 的具体定义。
- `sed -n '1,260p' /pool1/home/ysxin/E3nn/E3VB/utils/metrics.py`
  - 确认 `focus_weighted_mae`、`focus_precision`、`tail_false_positive_rate`、`macro_*` 汇总逻辑。
- `sed -n '1,240p' /pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py`
  - 确认 true/predicted focus 集合都采用 cumulative mass prefix 规则，而不是固定数量结构。
- `sed -n '400,450p' /pool1/home/ysxin/E3nn/E3VB/data/xmo_builder.py`
  - 确认 preprocess 阶段直接用 `normalized_weight` 构建 `top_mass_focus_mask`。
- `rg -n "BEST|TRAINING_DONE|best_val|EARLY_STOP" /pool1/home/ysxin/E3nn/E3VB/logs/*.log`
  - 用于定位近期关键实验日志。
- 读日志得到的重要结果：
  - `edgecontent_seed0_1938902.log`
    - 早期强基线，`best_val_mae=0.001405`，test 约 `0.000927`
  - `weightedmid_seed0_pc001_1942997.log`
    - 加权版本未超过上述强基线
  - `vboutfile_fixedbucket_20260501_1946155.log`
    - 后期 vb-only/fixedbucket 强结果，`best_val_mae=0.001719`
  - `vboutfile_topmass_1946552.log`
    - focus 指标可优化，但全局 MAE 明显不如早期最优基线
  - `checked_only_1946754.log`
    - checked 数据集本身明显更难
  - `unified_checked_vb_budget96_1946857.log`
    - unified 训练下 train 很低但 val 仍明显更高，说明主要瓶颈是跨数据集泛化与 focus/tail 校准
  - `unified_excl4_1947268.log`
    - 从已查看尾部看，指标弱于 `1946857` 这条 unified 路线
- `date '+%Y-%m-%d %H:%M'`
  - 当前写入 handoff 的时间戳为 `2026-05-06 17:25`

### 6. 当前报错/未解决问题
- 本轮没有遇到代码报错，也没有运行训练作业，因此没有新增 runtime error。
- 当前未解决的是“目标与 monitor/loss 权重的对齐问题”，不是语法或环境问题。
- 尚未动手改配置或代码，所以以下判断仍需实验验证：
  - 把 `validation_monitor` 切到 `macro_focus_weighted_mae` 后，是否会稳定提升真实 focus 回归质量
  - 降低 `top_mass_ranking_weight` 是否会改善 focus 内回归，且不会导致 focus 集合质量明显崩坏
  - `target_weight_power=0.5` 是否过度强调了 true focus 集合中最重的少数结构，从而损伤“前 98% 内所有结构”的均衡回归

### 7. 重要路径、变量名、函数名、参数
- 重要路径
  - `/pool1/home/ysxin/E3nn/E3VB/main.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/metrics.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/top_mass.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/config.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/xmo_builder.py`
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_checked_vb_unified_topmass_lowmem_excl4.yaml`
- 关键函数
  - `trainStep()`：普通 weighted regression + optional rank loss 路线
  - `trainStepTopMass()`：当前启用的 top-mass 训练路线
  - `topMassObjectiveLoss()`：当前主 loss 组合定义
  - `moleculeLocalWeightedMae()`：按分子平均的局部加权 MAE
  - `pairwiseRankLoss()`：分子内 pairwise hinge ranking
  - `crossSetMarginLoss()`：focus-vs-tail 边界排序约束
  - `computeTopMassFocusMask()`：true focus 集合构造
  - `buildPredictedTopMassMask()`：根据预测值构造 predicted focus 集合
  - `focusPriorityScore()`：当前复合 monitor 计算逻辑
- 关键变量/参数
  - `use_top_mass_objective`
  - `focus_cumulative_mass`
  - `validation_monitor`
  - `top_mass_regression_weight`
  - `top_mass_ranking_weight`
  - `tail_suppression_weight`
  - `target_weight_power`
  - `target_weight_offset`
  - `top_mass_focus_mask`
  - `macro_focus_weighted_mae`
  - `focus_precision`
  - `tail_false_positive_rate`
  - `macro_focus_priority_score`
- 当前 `excl4` 配置的重要现值
  - `focus_cumulative_mass: 0.98`
  - `validation_monitor: macro_focus_priority_score`
  - `top_mass_regression_weight: 1.0`
  - `top_mass_ranking_weight: 0.35`
  - `tail_suppression_weight: 0.05`
  - `target_weight_power: 0.5`
  - `top_mass_sample_strategy: focus_plus_mixed_tail`
  - `mixed_tail_top_fraction: 0.5`
  - `molecule_balanced_sampling: true`
  - `dataset_sampling_strategy: balanced`

### 8. 下一轮 Codex 应该继续做什么
- 第一优先级：围绕“真实前 98% 结构权重回归准确”重排配置目标，不要再默认把排序、precision、tail FPR 与回归目标并列。
- 建议的下一步执行顺序：
  - 复制当前 unified 配置，生成一份新的实验配置，不要直接覆盖旧配置。
  - 把 `validation_monitor` 改为 `macro_focus_weighted_mae`。
  - 第一轮只做最小改动 ablation，不改代码：
    - 实验 A：只改 `validation_monitor`
    - 实验 B：在 A 基础上把 `top_mass_ranking_weight: 0.35 -> 0.10`
    - 实验 C：在 A 基础上把 `top_mass_ranking_weight: 0.35 -> 0.00`
    - 实验 D：在 B 或 C 的基础上扫 `target_weight_power: 0.5 / 0.25 / 0.0`
  - 每组实验必须同时记录：
    - `macro_focus_weighted_mae`
    - `macro_focus_precision`
    - `macro_tail_false_positive_rate`
    - `macro_focus_spearman`
    - checked / vb 两个数据集各自的 `focus_weighted_mae`
- 如果最小改动实验显示：
  - `macro_focus_weighted_mae` 改善明显，且 precision/FPR 未灾难性恶化
    - 继续沿现有 top-mass 框架调参
  - `macro_focus_weighted_mae` 仍然卡住
    - 再考虑中等改动：拆分 rank 项，把 `focus_intra_rank` 和 `focus_tail_rank` 分开加权
  - 如果仍然无效
    - 再考虑结构性改动：新增一个显式的 focus-membership 分类头，回归头专门负责 focus 内权重
- 下一轮如果要真正动手改文件，优先修改：
  - `config_e2e_checked_vb_unified_topmass_lowmem_excl4.yaml` 的派生副本
  - 如需新增更贴目标的 monitor 或 loss，再改 `main.py` / `utils/losses.py`

## 2026-05-27 02:30 本轮工作总结

### 1. 本轮目标
- 回答用户关于当前模型能否直接做预测、预测输出与训练 target 的关系、以及为什么 `prediction_raw` 最大值经常不到 `1.0` 的问题。
- 用现有 checkpoint 对多个外部输入样本做真实推理，并把预测值和真实 `Lowdin weight` 做对比。
- 根据用户“top 级别结构数值不够准”的反馈，给出低风险调参方案，并实际修改主训练配置、提交新训练任务。

### 2. 已完成内容
- 确认当前仓库可以通过 `predict.py` 直接做推理，支持：
  - 单个 `.xmo`
  - 成对 `.xmi + .str`
- 确认当前主 checkpoint 可用：
  - `/pool1/home/ysxin/E3nn/E3VB/checkpoints/autovb_ocelot_2ds_1955935_best.pkl`
- 解释清楚了训练/推理中三类量的区别：
  - 训练 target 是 `t_i = y_i / y_max`
  - `prediction_raw` 是模型对这个 normalized target 的原始回归输出，不保证最大值为 `1`
  - `prediction_norm_by_pred_max` 是后处理得到的“最大值归一化”
  - `prediction_norm_by_full_sum` 是把 `prediction_raw` 反推成近似原始权重分布后的“总和归一化”
- 对用户给的 `c54h18_vb.xmi`：
  - 发现推理脚本当前不支持直接从 `.xmi` 的 `$str` block 读结构列表
  - 现场用 `awk` 从 `$str` block 提取出临时 `.str`
  - 首次在本机运行失败，原因是本机 `e3vb_jax` 环境导入 `jax/flax` 时受 `GLIBC` 版本限制
  - 切到 `pc004` 后成功完成推理
- 对 `c54h18` 做了多轮结果分析：
  - 用真实 `Lowdin weight` 的 top-98% focus 集与预测 `predicted_top_mass_focus` 做集合比较
  - 计算了 precision / recall / F1 / Jaccard
  - 说明该样本是“高召回、集合偏宽”的典型案例
  - 进一步区分了：
    - 原始权重尺度比较
    - 训练 target `y / y_max` 尺度比较
    - 按最大值重新缩放后的比较
- 对用户提供的 Xiatao 数据做了实际推理：
  - `TPS-HF.xmo`
    - 不能直接用于当前模型推理，因为缺少 `$orb` block 与 `Lowdin weights`
  - `163754619_VBSCF_COV.xmo`
    - 成功推理并与真实 `Lowdin Weights` 对比
  - `163754619_VBSCF.xmo`
    - 成功推理并与真实 `Lowdin Weights` 对比
- 对“是否要改成 softmax / 是否要在训练里强制 max=1 / 是否只调参”的几个方向做了取舍分析：
  - 结论是先不要大改目标形式
  - 第一优先级先做调参，优先加强 top 数值回归、弱化 ranking
- 根据用户确认，直接修改了当前主训练配置，具体改动：
  - `top_mass_regression_weight: 1.0 -> 1.50`
  - `top_mass_ranking_weight: 0.35 -> 0.20`
  - `max_tail_samples_per_molecule: 96 -> 64`
  - `max_structure_cost_per_molecule: 60000 -> 45000`
  - `batch_log_interval: 20 -> 100`
  - `iterator_log_interval: 20 -> 100`
  - 未修改训练周期相关参数
- 已提交一版新训练：
  - `job id = 1959140`
  - `RUN_MODE = autovb_ocelot_topreg150_rank020_tail64`

### 3. 修改/新增/检查过的文件
- 已修改：
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_autovb_ocelot_unified_topmass_lowmem_u1_macrofwmae.yaml`
- 已检查但未修改：
  - `/pool1/home/ysxin/E3nn/E3VB/predict.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/xmi_str_parser.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/xmo_parser.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/config.py`
  - `/pool1/home/ysxin/E3nn/E3VB/submit_train.sbatch`
- 本轮实际推理/分析过的外部数据文件：
  - `/pool1/home/ysxin/autovb_input/c54h18_vb.xmi`
  - `/pool1/home/ysxin/autovb_input/c54h18_980.vbo`
  - `/pool1/home/ysxin/Xiatao/dpvbs_data/DPS/TPS-HF.xmo`
  - `/pool1/home/ysxin/Xiatao/dpvbs_data/Test/result/163754619/163754619_VBSCF_COV.xmo`
  - `/pool1/home/ysxin/Xiatao/dpvbs_data/Test/result/163754619/163754619_VBSCF.xmo`

### 4. 关键实现逻辑 / 分析结论
- 关于 `prediction_raw`
  - 它不是“已经归一化后的最终权重”
  - 它是模型对训练 target `y / y_max` 的原始回归输出
  - 因为输出层没有做分子内 `max=1` 硬约束，所以即使训练标签最大值恒为 `1`，`prediction_raw` 的最大值也常常小于 `1`
- 关于“是否能从 `prediction_raw` 反推真实权重”
  - 可以近似反推
  - 公式是：
    - `pred_weight_i = max(prediction_raw_i, 0) / sum_j max(prediction_raw_j, 0)`
  - 这正是 `prediction_norm_by_full_sum`
- 关于 `c54h18`
  - 如果用真实 `Lowdin weight` 的 top-98% focus 集和预测 focus 集比较：
    - 真实 focus 约 `792` 或 `805`（取决于所用 block 与浮点截断；本轮最终用于 `y/y_max` / loss 分析时是 `792`）
    - 预测 focus `954`
    - recall 极高，precision 中等，说明模型不太漏重要结构，但会把 focus 集预测得偏宽
  - 如果按训练 target `y / y_max` 比较：
    - 排序还可以
    - 数值有明显压缩，尤其 top 结构 `prediction_raw` 偏低
- 关于 `163754619_VBSCF_COV`
  - 结构数只有 `14`
  - 预测权重对真实 `Lowdin weight`：
    - `MAE ≈ 0.01197`
    - `RMSE ≈ 0.01550`
    - `Spearman ≈ 0.94286`
  - 前 5 个主结构基本都抓到了，前 1/2 名有轻微互换
- 关于 `163754619_VBSCF`
  - 结构数 `1764`
  - 预测权重对真实 `Lowdin weight`：
    - `MAE ≈ 1.893e-4`
    - `RMSE ≈ 7.306e-4`
    - `Spearman ≈ 0.90648`
  - 这个 full 版本比对应的 COV 版本整体更稳，前 5 个主结构排序接近正确，数值仍有轻微压缩
- 关于本轮调参策略
  - 用户明确表示不想继续把 loss 权重结构搞复杂，也不想立刻大改成 softmax 或硬性 `max=1`
  - 因此本轮采用的是最保守且最有针对性的路径：
    - 提高 `top_mass_regression_weight`
    - 降低 `top_mass_ranking_weight`
    - 同时减少 tail 采样与 per-molecule 预算，加快训练并降低 tail 噪声

### 5. 已运行命令与结果
- 本地检查推理入口与 checkpoint：
  - `rg -n "argparse|if __name__ == '__main__'|def main|checkpoint|xmo|xmi|str" /pool1/home/ysxin/E3nn/E3VB/predict.py`
  - `ls -1 /pool1/home/ysxin/E3nn/E3VB/checkpoints`
- 本地检查/提取 `c54h18_vb.xmi`：
  - `awk 'BEGIN{flag=0; idx=0} /^\\$str$/{flag=1; next} /^\\$end$/{if(flag) exit} flag && NF {idx++; printf("%8d   *****  %s\\n", idx, $0)}' ... > /tmp/c54h18_from_xmi.str`
  - 成功生成 `980` 条结构
- 本机直接跑 `predict.py`：
  - 在 `e3nn` 环境中缺 `flax`
  - 在 `e3vb_jax` 环境中又因 `GLIBC_2.27` / `ml_dtypes` 失败
  - 结论：本机不能直接跑正式 JAX 推理
- 在 `pc004` 上成功跑推理：
  - `c54h18_vb.xmi + /tmp/c54h18_from_xmi.str`
  - `163754619_VBSCF_COV.xmo`
  - `163754619_VBSCF.xmo`
- 生成的重要输出文件：
  - `/pool1/home/ysxin/autovb_input/c54h18_top10_predictions.tsv`
  - `/pool1/home/ysxin/Xiatao/dpvbs_data/Test/result/163754619/163754619_VBSCF_COV_predictions.tsv`
  - `/pool1/home/ysxin/Xiatao/dpvbs_data/Test/result/163754619/163754619_VBSCF_predictions.tsv`
- 成功提交训练：
  - `ssh pc004 'CONFIG_PATH=... RUN_MODE=autovb_ocelot_topreg150_rank020_tail64 sbatch ... submit_train.sbatch'`
  - 返回：`Submitted batch job 1959140`

### 6. 当前未解决问题 / 潜在风险
- 本轮没有改 loss / model 代码，只改了主训练配置。
- 还没有看到 `1959140` 的实际 early-stage 指标，因此：
  - 不知道 `top_mass_regression_weight=1.50` 是否足以改善 top 数值精度
  - 也不知道 `top_mass_ranking_weight=0.20` 是否会让 `focus_pair_acc` 掉太多
- 当前推理脚本 `predict.py` 仍然不支持：
  - 直接从 `.xmi` 的 `$str` block 读取结构列表
  - 所以遇到只有 `.xmi` 没有外部 `.str` 的输入时，还需要手工提取临时 `.str`
- 本机 `e3vb_jax` / `e3vb_e2e` 环境仍受 `GLIBC` 限制，正式 runtime 仍建议上 `pc004` 或 Slurm 节点执行

### 7. 重要路径、变量名、函数名、参数
- 关键预测文件
  - `/pool1/home/ysxin/E3nn/E3VB/predict.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/xmi_str_parser.py`
  - `/pool1/home/ysxin/E3nn/E3VB/data/xmo_parser.py`
- 关键训练/配置文件
  - `/pool1/home/ysxin/E3nn/E3VB/config_e2e_autovb_ocelot_unified_topmass_lowmem_u1_macrofwmae.yaml`
  - `/pool1/home/ysxin/E3nn/E3VB/submit_train.sbatch`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/losses.py`
  - `/pool1/home/ysxin/E3nn/E3VB/utils/config.py`
- 关键参数（本轮新值）
  - `top_mass_regression_weight: 1.50`
  - `top_mass_ranking_weight: 0.20`
  - `max_tail_samples_per_molecule: 64`
  - `max_structure_cost_per_molecule: 45000`
  - `batch_log_interval: 100`
  - `iterator_log_interval: 100`
- 关键运行标识
  - `RUN_MODE=autovb_ocelot_topreg150_rank020_tail64`
  - `job_id=1959140`

### 8. 下一轮 Codex 应该继续做什么
- 第一优先级：盯住 `1959140` 的早期训练日志，重点看：
  - `focus_wmae`
  - `focus_pair_acc`
  - `focus_spearman`
  - `reg_loss`
  - `rank_loss`
- 如果结果趋势是：
  - `focus_wmae` 改善明显，`focus_pair_acc` 只小幅下降
    - 继续这条“回归更强、排序略弱”的路线
  - `focus_pair_acc` 明显崩掉
    - 下一版把 `top_mass_ranking_weight` 回调到 `0.25`
  - `focus_wmae` 几乎没改善
    - 下一版考虑再调：
      - `target_weight_power`
      - `target_weight_offset`
      - 或再降低 tail 预算
- 如果下一轮要继续改训练而不想加复杂 loss：
  - 最优先还是继续走调参路线
  - 不建议第一步就改成分子内 softmax
  - 不建议第一步就加硬性的 `max=1` 约束
