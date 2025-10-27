# Agentic Distillation 的 QA 清单（中文）

在运行蒸馏循环时使用此清单来保持数据集质量。

## 运行前

- [ ] Teacher 与 reviewer 的 API 密钥已导出（例如 `TEACHER_API_KEY`、`REVIEWER_API_KEY` 等）。
- [ ] 已配置 `teacher_pool`/`reviewer_pool`，并确认权重与优先顺序符合预期。
- [ ] 场景配置已审核：目标回合数、配额、随机种子与语言策略（英文优先/必要时中文回顾）。
- [ ] 并发级别（concurrency）已根据 API 速率限制与预算进行调整。
- [ ] 如果执行真实命令/API，已连接可选的工具处理器。

## 运行中

- [ ] 监控日志中是否存在高丢弃或审核者拒绝率（>20% 为告警信号）。
- [ ] 验证审核者反馈是否被正确解析（在元数据中查找 `review_feedback` 字段）。
- [ ] 当启用 `require_tool_calls` 时，检查轨迹中确实出现工具调用。
- [ ] 监控各端点延迟；如果某个模型被限流，请调整池权重或并发设置。

## 运行后

- [ ] 检查分片样本：

```bash
head -n 1 data/exports/terminal/shard-00000.jsonl | jq .
```

- [ ] 确认语言混合：以英文叙述为主，必要时仅包含简洁的中文摘要。
- [ ] 审查审核者得分与场景验证得分的漂移情况。
- [ ] 随机抽查回合，留意幻觉（hallucinations）与被对抗的失败案例（red-team failures）。
- [ ] 将用于运行的配置文件与生成的分片元数据一并版本化。

