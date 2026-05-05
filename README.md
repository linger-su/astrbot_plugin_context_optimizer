# 上下文优化器插件 (astrbot_plugin_context_optimizer)

极致省 token 的上下文优化器。默认只发最新消息给 LLM，LLM 看不懂时自动补上下文。

## 核心功能

- **极简模式**：每次只发最新消息，极致省 token（10k → 几百）
- **均衡模式**：保留最近 N 轮 + 压缩更早历史为摘要
- **困惑自动展开**：LLM 回复"我不太理解"时自动补入压缩后的历史
- **双引擎压缩**：支持 TF-IDF（本地零成本）或 LLM（智能摘要）
- **工具调用过滤**：自动清理 tool_calls 和 tool 结果

## 压缩方式

| 方式 | 说明 | 成本 |
|------|------|------|
| TF-IDF | 本地算法，基于词频提取关键句子 | 零 |
| LLM | 用 AstrBot 已配置的 provider 做智能压缩 | 少量 token |
| LLM+TF-IDF | 先尝试 LLM，失败自动回退 TF-IDF | 少量 token |

## 配置说明

在 AstrBot 插件配置页面可调整：

- `context_mode`: 上下文模式（minimal / balanced / full）
- `compress_method`: 压缩方式（tfidf / llm / llm+tfidf）
- `compress_provider_id`: 用于压缩的 LLM provider ID（留空=默认）
- `compress_ratio`: TF-IDF 压缩保留比例
- `max_compressed_tokens`: 压缩后最大 token 数
- `confusion_threshold`: 困惑检测灵敏度

## 安装

1. 在 AstrBot WebUI → 插件管理 → 从文件安装
2. 选择 `astrbot_plugin_context_optimizer.zip`
3. 启用插件，按需调整配置

## 作者

AutoClaw

## 版本

v2.1.0
