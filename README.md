# 数字绘画涂抹（Smudge）引擎

一个确定性的、基于颜料输运（paint transport）的涂抹笔刷。笔刷沿指针路径从画布
**拾色**到一个容量有限的独立**储量（reservoir）**，再按压力、速度与混合参数把颜料
**沉积**到后续位置，从而产生带方向和历史的颜色输运——不是对鼠标周围做高斯模糊，也
不会每一步重新读取原始图像。

## 运行

```bash
pip install -r requirements.txt
python tests/test_smudge.py      # 22 个测试 / 85 个断言
python tests/test_migration.py   # 9 个迁移、存档与重放测试
python tests/demo_runner.py      # 生成 outputs/ 下的 PNG、.npy 与账本
python tests/migration_demo.py   # 生成 10_* 迁移记录、半程存档与对比导出
python tests/verify_exports.py   # 不依赖引擎数学，独立核对导出像素
```

## 核心模型

所有颜料量都用 **预乘 RGBA（premultiplied RGBA）float64** 表示，第四分量是颜料的
alpha 体积。

- 画布 `Canvas` 存直色（straight alpha）；笔刷储量 `Reservoir` 存预乘颜料，二者分离。
- 储量有 `capacity` 上限，永远不会超量；因此拖得再慢也不会“繁殖”颜料。
- 每个空间步严格按 **先全部读、后统一写** 的顺序：先快照足迹内所有像素 → 由快照计算
  拾色、由储量快照计算沉积（此阶段不改任何数据）→ 一次性向量化写回，再更新储量。
  因此结果与像素遍历方向无关（测试用正/反两种遍历顺序验证逐位一致）。

### 两种结算办法

| 模型 | 含义 | 实现 |
|------|------|------|
| `contact`（A，每次接触按覆盖面积结算） | 每个接触戳内，足迹内每个像素按自身覆盖权重独立拾色与释放 | `smudge.model.step_contact` |
| `flux`（B，沿笔路积分拾放通量） | 维护一张覆盖图，某像素在本笔触内只有第一次被扫过时才拾色，重复来回不重复取色 | `smudge.model.step_flux` |

两模型共享同一套足迹、读/写协议、储量与账本。

### 关键确定性设计

- **跨批次一致**：路径在固定的弧长网格上细分，会话只保留两个跨批次的延续量
  （锚点、距下一戳的剩余距离 `need`）。同一批样本无论怎么切分喂入，戳的位置、
  压力、速度因子完全相同，重放差异为精确的 `0.0`（见 `outputs/09_replay_diff.txt`）。
- **采样/写入次序**：所有画布读取集中在 `_gather` 的快照里完成，写回在最后进行，
  单步结果不依赖遍历顺序。
- **透明区处理**：透明像素的预乘颜料为 0，拾色什么也取不到；沉积时按储量自身颜色在
  新覆盖的 alpha 上落色，因此颜料可以穿过透明间隙且颜色不变；未接触区域逐位保持。
- **边缘裁剪**：落在画布外的沉积不销毁也不虚构，留在储量里并记入 `clipped_paint`。

### 旧笔尖颜色档迁移

`smudge.migration.migrate_legacy_brush` 把只含 `tip_color`/`nib_color` 的旧笔刷读成
带 `capacity` 的颜料仓配方，并为每个字段保留来源与确认状态：

- 显式 `"capacity": 0` 是已确认的零容量；从来没有存过容量键则保持 `pending`，不会被
  误读成零。可用 `provisional_capacity` 仅供重放，决定仍标记为待确认。
- `humidity` 缺失时保持待确认；引擎传输仍由显式 `pickup_rate` / `deposit_rate` 决定，
  不把未证实的湿度公式伪装成历史事实。
- `infer_capacity_bounds` 从 `picked` 与实际 `debit` 事件回推容量前缀区间。若存在
  `pickup_lost > 0` 和事件前储量，可在饱和点恢复精确容量；否则只给最低下界。
- 无法还原真实持续笔画起点时，第一个可重放事件会写成带坐标的 `StrokeOrigin` 标记，状态
  为 `marked`/待确认，而不是冒充历史真起点。
- `BrushWorkbench.open_project` 同时保留未改写的旧 raw 句柄和迁移后的新句柄。旧工程只读
  打开；确认项和转换结果写入另存的 conversion JSON。`SmudgeSession.save_checkpoint` /
  `load_checkpoint` 可在半笔时保存画布、储量、弧长细分锚点和账本，重载后继续的结果与一次
  跑完逐位一致。
- `outputs/10_migration_record.json` 是红区拾色、穿透明区、进入蓝区的可回查记录，包含
  拾色损耗、画布增量、命名边缘差异、剩余色量、容量推断和源档 SHA-256。

### 主要参数（`SmudgeConfig`）

`pickup_rate` / `deposit_rate`（拾/放比例）、`capacity`（储量体积）、`flow`（颜色混合
强度，默认 1.0 为严格质量守恒的物理模式）、`blend`（1 = 凸组合颜色置换涂抹，
0 = 加式预乘混合）、`deposit_concentration`（向笔刷核心集中）、`speed_mode` /
`speed_power`（速度调制）、`step_length`（空间采样间距）。

## 质量守恒

- 默认 `flow = 1`、`blend = 1` 下，画布颜料 + 储量在每个预乘通道上**精确守恒**
  （机器精度，约 1e-16）。
- `flow > 1` 仅放大颜色混合强度：alpha 覆盖度仍精确守恒，RGB 允许有界偏差（颜色被更强
  地重新混合）。高 flow 下反复拾取自己沉积的颜色会像真实颜料一样逐渐变浊，需要严格守恒
  时保持 `flow = 1`。
- 加式模式下 alpha 超过 1 的溢出计入 `spill` 账本，绝不静默丢弃。

## 导出物（`outputs/`）

每个场景导出三件套：`<name>.png`（8 位 RGBA）、`<name>.npy`（精确 float64 像素，供
离线核对）、`<name>.report.txt`（拾/放/遮挡/裁剪/溢出与质量账本）。场景覆盖相邻色块、
1px 细线、半透明边缘、透明间隙跨越、压力/速度调制以及两笔触历史传递。
