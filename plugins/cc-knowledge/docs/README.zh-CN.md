# CC Knowledge Cultivator（知识沉淀器）

从 Claude Code 日常交互中自动沉淀领域知识，构建结构化、可检索的知识 Wiki。

---

## 简介

### 痛点

每次 Claude Code 会话都会产生宝贵的领域知识——错误修复方案、工作流程发现、平台踩坑经验、架构决策。但这些知识在会话结束后就消失了。下次遇到相同问题时，你（或 Claude）只能从零开始。

### 解决方案

CC Knowledge Cultivator 自动从你的会话中提取经验教训，存储到结构化 Wiki 中，让 Claude 在未来的会话中自动读取。随着时间推移，Claude 对*你的*特定领域会越来越聪明——记住踩过的坑、应用验证过的工作流、避免重复过去的错误。

### 工作原理

```
会话结束
   ↓
第一阶段：加权启发式门控（≥8 条消息 + 评分 ≥ 4）
   ↓ 通过
第二阶段：Haiku 分诊（是否可能存在可迁移经验？YES/NO，出错时放行）
   ↓ YES
由 Opus 从会话记录中提取经验教训
   ↓
写入 ~/wiki/topics/<领域>/raw/notes/
   ↓
重新生成召回技能 ~/.claude/skills/cc-knowledge-<领域>/
   ↓
下次会话：Claude 自动发现该技能 → 读取领域知识
```

### 核心特性

- **零操作捕获** — SessionEnd 钩子自动触发，无需手动命令
- **智能门控** — 两段式门控（加权启发式评分 + 廉价的 Haiku 分诊），只对可能蕴含可迁移经验的会话进行知识沉淀
- **按领域组织** — 知识按主题分类（ml-training、aws-infra 等）
- **自动召回** — 生成的技能让 Claude 无需显式查询即可感知过去的经验
- **分层自治** — 新经验自动写入；对已有文章的修改需人工审核
- **llm-wiki 兼容** — 存储格式与 [llm-wiki](https://github.com/nvk/llm-wiki) 兼容，支持编译、查询和 Obsidian 浏览
- **完全独立** — 无运行时依赖，不安装 llm-wiki 也能正常工作

---

## 安装

### 前提条件

- Claude Code（CLI 或 IDE 扩展）
- Node.js ≥18（用于钩子脚本）

### 启用插件

如果你从 llm-cc-market 市场安装：

```bash
# 在 Claude Code 设置（~/.claude/settings.json）的 enabledPlugins 下添加：
"cc-knowledge@llm-cc-market": true
```

### 初始化 Wiki Hub

启用后，启动新的 Claude Code 会话并运行：

```
/cc-knowledge:init
```

这将创建：
- `~/wiki/` — 知识中心
- `~/wiki/wikis.json` — 主题注册表
- `~/.config/llm-wiki/config.json` — 路径配置
- `~/.claude/cc-knowledge-pending/` — 待处理队列

---

## 快速开始

### 1. 正常工作

像往常一样使用 Claude Code。修 bug、配置环境、写代码。

### 2. 会话结束 → 自动沉淀

当你的会话结束时，钩子检查：
- 你是否交换了 ≥8 条消息？
- 加权信号评分是否达标（错误→修复、用户纠正、多文件编辑）？
- Haiku 快速分诊是否认为可能存在可迁移的持久经验？

全部通过 → 自动提取和存储经验教训。

### 3. 下次会话 → Claude 记住了

在下次相关主题的会话中，Claude 会在可用技能列表中看到召回技能。当对话匹配时（比如你再次进行 ML 训练），Claude 会调用该技能并读取你积累的知识。

### 示例流程

```
会话 1：你配置 Megatron 训练，遇到 cuDNN 错误并修复。
  → 沉淀器提取："cuDNN 需要先安装 cuda-keyring 包"
  → 写入 ~/wiki/topics/ml-training/raw/notes/2026-05-22-ll-megatron-setup.md
  → 生成技能 ~/.claude/skills/cc-knowledge-ml-training/SKILL.md

会话 2：你开始另一个训练任务。
  → Claude 发现 "cc-knowledge-ml-training" 技能，调用它
  → Claude 现在知道："安装 NVIDIA apt 包前必须先装 cuda-keyring"
  → 主动提醒你或直接应用修复，避免重复踩坑
```

---

## 命令参考

### `/cc-knowledge:init`

初始化 Wiki Hub。

```
/cc-knowledge:init
/cc-knowledge:init --path ~/my-wiki
/cc-knowledge:init --topic ml-training
```

| 参数 | 说明 |
|------|------|
| `--path <路径>` | 自定义 Hub 位置（默认：`~/wiki/`） |
| `--topic <名称>` | 同时创建第一个主题 |

### `/cc-knowledge:cultivate`

手动从当前会话中提取经验。

```
/cc-knowledge:cultivate
/cc-knowledge:cultivate "EC2 上的 CUDA 配置"
/cc-knowledge:cultivate --wiki ml-training --dry-run
/cc-knowledge:cultivate --retry
```

| 参数 | 说明 |
|------|------|
| `"主题提示"` | 帮助分类会话（可选） |
| `--wiki <名称>` | 指定目标主题 |
| `--dry-run` | 预览经验而不写入 |
| `--retry` | 处理之前失败的提取 |
| `--include-archived` | 显式允许归档主题 Wiki |

### `/cc-knowledge:review`

审核和接受/拒绝待处理的提案。

```
/cc-knowledge:review
/cc-knowledge:review --wiki ml-training
/cc-knowledge:review --accept-all
```

| 参数 | 说明 |
|------|------|
| `--wiki <名称>` | 只显示某个主题的提案 |
| `--accept-all` | 接受所有待处理提案 |
| `--reject <id>` | 拒绝指定提案 |
| `--include-archived` | 包含归档主题 Wiki |

### `/cc-knowledge:status`

显示知识沉淀仪表盘。

```
/cc-knowledge:status
/cc-knowledge:status --wiki ml-training
/cc-knowledge:status --include-archived
```

---

## 详细工作原理

### SessionEnd 钩子

钩子脚本（`scripts/session-end-cultivate.js`）在每次会话结束时触发，应用**两段式门控**，确保只有有实质内容的会话才会触达（昂贵的）Opus 提取器：

**第一阶段 — 确定性加权启发式（零 LLM token）：**
1. 会话必须有 ≥8 条用户消息。
2. 单遍扫描会话记录计算加权评分：
   - 错误→修复序列（出错后被某次编辑修复）：**每次 +3**
   - 用户纠正（"不对"、"错了"、"用 X 代替"等）：**每次 +2**
   - 编辑涉及的不同文件数：**每个 +1**（上限 5）
   - 明显非琐碎的编辑量（>3 次编辑）：**+1**
3. 只有 `评分 ≥ 4` 才通过。单次琐碎编辑只得 1 分会被拒绝——这正是阻止无关紧要话题被沉淀的机制。

**第二阶段 — 廉价 LLM 分诊（Haiku）：** 在投入完整的 Opus 提取之前，先用一次简短的 Haiku 调用看一段 token 受限的用户发言摘要，回答 YES/NO：「是否可能存在 ≥1 条可迁移的持久经验？」只有 YES 才继续。该阶段**出错即放行**——任何错误、超时或缺少 `claude` 二进制都会继续提取，而不是悄悄丢弃会话。

两个阶段都通过后，脚本：
1. 从最近消息中提取主题提示
2. 启动 `claude -p`（Opus）运行沉淀引擎
3. 在 `~/.claude/cc-knowledge-pending/` 写入待处理标记（包含完整门控信号）

### 提取流水线（5 个阶段）

| 阶段 | 动作 | 输出 |
|------|------|------|
| 1. 会话扫描 | 找出错误→修复模式、用户纠正、发现、踩坑 | 候选列表 |
| 2. 经验提取 | 结构化为：类别/背景/症状/根因/修复/规则 | 2-7 条经验 |
| 3. Wiki 定位 | 分类到已有主题或创建新主题 | 主题路径 |
| 4. 分层写入 | 原始笔记 → 自动写入；文章追加 → `.librarian/proposals/` | 文件 |
| 5. 收尾 | 更新索引、日志，重新生成召回技能 | 完成 |

### Wiki 目录结构

```
~/wiki/                                 # Hub（知识中心）
├── wikis.json                          # 主题注册表
├── _index.md                           # Hub 索引
├── log.md                              # 全局活动日志
└── topics/
    └── ml-training/                    # 示例主题
        ├── raw/notes/                  # 经验教训在这里
        │   ├── _index.md
        │   └── 2026-05-22-ll-cuda-setup.md
        ├── wiki/                       # 编译后的文章（通过 llm-wiki）
        │   ├── concepts/
        │   ├── topics/
        │   └── references/
        ├── .librarian/proposals/       # 待审核的文章修改
        ├── _index.md
        ├── config.md
        └── log.md
```

### 经验格式

每个原始笔记遵循以下结构：

```yaml
---
title: "Lessons Learned: <主题>"
type: lessons-learned
source: session
date: 2026-05-22
tags: [lessons-learned, cuda, gpu-training]
lesson_count: 3
category: notes
confidence: high
summary: "EC2 上的 cuDNN 安装和 CUDA 版本匹配"
---
```

```markdown
## Lesson 1: cuDNN 需要 cuda-keyring

**Category**: gotcha
**Context**: 在 EC2 上为训练安装 cuDNN 9.x
**Symptom**: `apt-get install cudnn` 报 GPG 验证错误
**Root cause**: NVIDIA 仓库要求先安装 cuda-keyring 包
**Fix**: 先执行 `apt-get install cuda-keyring-1.1`，再安装 cudnn
**Rule**: 安装任何 NVIDIA apt 仓库包之前，必须先安装 cuda-keyring
```

### 召回技能生成

每次沉淀后，会在 `~/.claude/skills/cc-knowledge-<主题>/SKILL.md` 重新生成技能，包含：
- 排名前 10 的规则（按频率 × 时效排序）
- 快速踩坑表（症状 → 根因 → 修复）
- 深入查看的完整文件链接
- `description:` 字段中的触发关键词（用于自动发现）

### 分层自治

| 操作 | 行为 |
|------|------|
| 新原始笔记（经验文件） | 自动应用 |
| 新主题创建 | 自动应用 |
| 索引/日志更新 | 自动应用 |
| 召回技能重新生成 | 自动应用 |
| 向已有文章追加规则 | → `.librarian/proposals/`（需审核） |
| 修改已有笔记 | → `.librarian/proposals/`（需审核） |

---

## 配置

### Hub 路径

Wiki Hub 位置存储在 `~/.config/llm-wiki/config.json`：

```json
{ "hub_path": "~/wiki" }
```

修改方式：
```
/cc-knowledge:init --path ~/my-custom-wiki
```

### 门控阈值

默认：≥8 条用户消息 **且** 加权信号评分 ≥ 4。如需调整：
- `MIN_MESSAGES`（消息下限）在 `plugins/cc-knowledge/scripts/session-end-cultivate.js`
- `MIN_SESSION_SCORE` 与 `SCORE_WEIGHTS`（信号评分）在 `plugins/cc-knowledge/scripts/lib/utils.js`

调低 `MIN_SESSION_SCORE` 会让门控更宽松；调高则让 wiki 更干净。第二阶段的 Haiku 分诊仅在第一阶段通过后运行，且出错即放行，因此它本身不会单独拦下会话。

---

## llm-wiki 互操作

沉淀器以 llm-wiki 兼容格式写入文件。如果你安装了 [llm-wiki](https://github.com/nvk/llm-wiki)，可以免费获得以下能力：

| 功能 | 命令 |
|------|------|
| 将原始笔记编译为精美文章 | `/wiki:compile --wiki ml-training` |
| 查询知识库 | `/wiki:query "如何在 EC2 上修复 cuDNN？"` |
| 生成报告/摘要 | `/wiki:output report --wiki ml-training` |
| 审计过时内容 | `/wiki:librarian --wiki ml-training` |
| 在 Obsidian 中浏览 | 打开 `~/wiki/topics/ml-training/` 作为 Vault |

### Obsidian 支持

每个主题 Wiki 使用双链格式（`[[wikilinks]]` + 标准 Markdown 链接），可以在 Obsidian 中原生浏览，支持图谱视图和反向链接。

---

## 常见问题

### "Not initialized" 错误

运行 `/cc-knowledge:init` 创建 Wiki Hub。

### 沉淀从未触发

检查门控条件：
- 你的会话需要 ≥8 条用户消息
- 并且加权信号评分 ≥ 4（错误→修复 +3、用户纠正 +2、不同文件 +1、>3 次编辑 +1）
- 单次琐碎编辑按设计无法达标——随时使用 `/cc-knowledge:cultivate` 进行手动提取

### 待处理标记堆积

如果 `~/.claude/cc-knowledge-pending/` 中文件堆积：
- 运行 `/cc-knowledge:cultivate --retry` 处理它们
- 或手动删除过期的 `.json` 文件

### 召回技能未加载

验证技能是否存在：
```bash
ls ~/.claude/skills/cc-knowledge-*/SKILL.md
```
如果缺失，手动重新生成：
```bash
node <插件路径>/scripts/regen-skill.js <主题名称>
```

### 与 llm-wiki 冲突

不会冲突——两个工具写入同一个 `~/wiki/` 结构。CC Knowledge Cultivator 将经验写入 `raw/notes/`，将审核材料写入 `.librarian/proposals/`；llm-wiki 管理 `wiki/`（编译后的文章）。两者互补。

---

## 架构

```
plugins/cc-knowledge/
├── .claude-plugin/plugin.json        # 插件清单
├── hooks/hooks.json                  # SessionEnd + SessionStart 钩子定义
├── commands/
│   ├── init.md                       # /cc-knowledge:init
│   ├── cultivate.md                  # /cc-knowledge:cultivate
│   ├── review.md                     # /cc-knowledge:review
│   └── status.md                     # /cc-knowledge:status
├── skills/cultivator-engine/
│   ├── SKILL.md                      # 提取流水线 Prompt
│   └── references/                   # 格式规范和算法
├── scripts/
│   ├── lib/utils.js                  # 共享工具函数
│   ├── session-end-cultivate.js      # SessionEnd 钩子（门控 + 启动）
│   ├── session-start-check.js        # SessionStart 钩子（待处理检查）
│   └── regen-skill.js               # 召回技能重新生成
└── docs/
    ├── README.md                     # 英文文档
    └── README.zh-CN.md              # 本文档
```

---

## 许可证

MIT
