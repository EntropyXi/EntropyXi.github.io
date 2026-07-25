---
title: 上下文准确度（Context Precision）
description: 梳理Ragas中Context Precision指标族，涵盖LLM驱动、无参考答案、非LLM字符串匹配与ID比对四种变体的原理与适用场景。
date: 2026-07-25 10:00:00
tags:
  - 深度学习
  - RAG学习
mathjax: true
categories:
  - 深度学习
  - RAG学习
---
<!-- more -->

### Motivation
在 RAG 系统里，检索器通常会返回多个上下文片段（chunk）。哪怕召回率是100%，如果真正有用的片段被排在列表末尾、而无关片段占据前排，LLM 在生成时仍然可能被噪音带偏。所以我们不仅关心「是否检出了相关片段」，更关心「相关片段是否排在前面」。

Context Precision 就是 Ragas 中用来量化这一点的指标族。它的核心思路跟信息检索中的 Average Precision（AP）基本一致：遍历排序列表的每个位置，计算截至当前位次的精确率，然后只在真正相关的片段上加权累加，无关片段不贡献任何分数。这等价于给排在后面的相关片段施加惩罚——排得越靠后，前面的假阳性越多，Precision@k 就越低。

但 Ragas 对 Context Precision 做了不少工程化的扩展：用 LLM 做 relevance 判断、用字符串距离绕开 LLM、用 ID 直接比对、甚至不需要 reference 也能评估。底下按版本/变体分别梳理。

### A. 基础形式：LLM 驱动的 Context Precision

这是 Ragas 中最标准的 Context Precision 实现。给定一个问答对和一组检索到的上下文列表，LLM 逐条判断每个 chunk 是否对回答该问题有帮助（verdict 为 `"1"` 表示相关，`"0"` 表示无关）。然后按 AP 公式算分：

$$
\text{Context Precision@K}
=
\frac{\sum_{k=1}^{K} \left( \text{Precision@k} \cdot v_k \right)}
{\text{Total number of relevant items in the top } K}
$$

其中：
- $K$ 是 `retrieved_contexts` 中的 chunk 总数
- $v_k \in \{0, 1\}$ 表示第 $k$ 个位置上的 chunk 是否相关
- $\text{Precision@k} = \frac{\sum_{i=1}^k v_i}{k}$，即前 $k$ 个位置中相关片段的比例

`分子里乘以 v_k 这一步是关键——如果 chunk 不相关，v_k=0，该项直接消失，所以无关片段不会贡献分子。分母是所有相关片段的总数，用来做归一化。`

从代码层面看，核心逻辑等价于：

```python
verdict_list = [1 if verdict == "1" else 0 for each chunk]
denominator = sum(verdict_list) + 1e-10
numerator = sum([
    (sum(verdict_list[:i+1]) / (i+1)) * verdict_list[i]
    for i in range(len(verdict_list))
])
score = numerator / denominator
```

`分母加 1e-10 是为了防止没有相关片段时除以零。`

**输入要求**：需要 `question`、`contexts`（即 `retrieved_contexts`）、`ground_truth`（reference answer）。evaluation_mode 为 `qcg`（question–context–ground_truth）。

**得分解释**：
- 所有相关片段都在最前面 → 分数接近 1.0
- 第一个位置是无关片段，相关片段从第二个才开始 → 分数大约 0.5
- 全部无关 → 分数为 0（或接近 0，因为分母的 1e-10）

`注意：即使只有一个无关片段插在相关片段前面，分数也会显著下降。这就是 AP 类指标的 rank-sensitive 特性。`

参数上可以传 `llm` 自定义判断模型，也可以传 `context_precision_prompt` 自定义 prompt 模板。默认 batch_size 为 15。

### B. Context Utilization：不需要 ground truth 的变体

标准 Context Precision 需要一个 reference answer 来让 LLM 判断 chunk 是否相关。但在很多场景里，我们可能压根没有标准答案——比如在做在线评估，或者只是想快速检查检索质量。

Context Utilization 就是为此设计的。它和 Context Precision 共享完全相同的公式和 prompt 结构，唯一的区别是：**它用系统自己生成的 response 代替 ground truth 来做 relevance 判断**。

`直觉上，如果检索到的某个 chunk 确实对生成当前这个 response 有帮助，那它就是相关的。这绕开了对人工标注 reference 的依赖。`

在 Ragas 内部，Context Utilization 的 `evaluation_mode` 被设为 `qac`（question–answer–context），也就是要求 `question`、`answer`（生成的 response） 和 `contexts` 三个字段。代码层面它本质上是 `ContextPrecision` 的子类：

```python
class ContextUtilization(ContextPrecision):
    name: str = "context_utilization"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
```

等价于新版 Ragas 中的 `LLMContextPrecisionWithoutReference`。

**使用场景**：
- 有 ground truth → 用 Context Precision（`LLMContextPrecisionWithReference`）
- 没有 ground truth，只有模型自己的回答 → 用 Context Utilization（`LLMContextPrecisionWithoutReference`）

`值得注意的一个陷阱：因为 ContextUtilization 是 ContextPrecision 的子类，某些旧版本的 Ragas（如 v0.1.8）在做列名校验时，会用 isinstance(m, ContextPrecision) 判断是否缺 ground truth，结果连 ContextUtilization 也被误杀了。这个 bug 在后续版本中已修复，但如果你还在用老版本，可能需要手动绕过。`

### C. Non-LLM Based Context Precision：绕过 LLM 的确定性评估

上面两种方案都要调 LLM 做逐条判断，成本和延迟都比较高。如果你有一个高质量的 reference_contexts 集合（不是 reference answer，而是一组标注好的、真正应该被检索到的上下文），那其实可以用纯字符串匹配来做 relevance 判断。

NonLLM Context Precision 的思路是：对 `retrieved_contexts` 中的每一个 chunk，用传统的字符串距离/相似度方法（底层依赖 `rapidfuzz` 库，核心是 Levenshtein 距离）跟 `reference_contexts` 中的每一条逐一比对。只要有一条 reference context 的相似度超过阈值，该 chunk 就算"命中"。

$$
\text{verdict for chunk}_k =
\begin{cases}
1, & \max_{j} \text{sim}(\text{chunk}_k,\ \text{ref\_context}_j) \geq \tau \\
0, & \text{otherwise}
\end{cases}
$$

`本质上就是把"LLM 判断相关性"这一步替换成了"字符串匹配阈值判断"。其余 AP 计算逻辑完全一样。`

得到 verdict 列表后，后续的 AP 公式计算与基础 Context Precision 完全相同。

**输入要求**：需要 `user_input`、`retrieved_contexts`、`reference_contexts`。不需要 LLM。

**优势**：
- 完全确定性，每次运行结果一致
- 不依赖 LLM API，速度快，成本低
- 适合在 CI/CD 管线中做自动化回归测试

**局限**：
- 前提是你得有一组精心标注的 reference_contexts，标注成本可能很高
- 字符串匹配是 surface-level 的，语义相似但措辞不同的内容可能被漏判
- 对 embedding 检索的评估可能偏严——LLM 能从 paraphrasing 中判断相关性，但 Levenshtein 不行

### D. ID Based Context Precision：最简单的集合比对

如果你已经给每个文档/chunk 分配了唯一 ID，那评估精确率甚至可以退化成一个纯集合操作——完全不需要看任何文本内容。

ID Based Context Precision 的定义非常朴素：检索到的 chunk ID 列表与参考 ID 列表的交集大小，除以检索到的 chunk 总数。

$$
\text{ID-Based Context Precision}
=
\frac{|\ \text{retrieved\_ids} \cap \text{reference\_ids}\ |}
{|\ \text{retrieved\_ids}\ |}
$$

`这实际上就是二分类问题里的 Precision。不需要加权，不需要逐位次累积，因为 ID 比对没有"部分相关"这一说——命中了就是1，没命中就是0。`

**输入要求**：只需要 `retrieved_context_ids` 和 `reference_context_ids` 两个字段。ID 可以是字符串或整数。

**示例**：如果检索返回了 `["doc_1", "doc_2", "doc_3", "doc_4"]`，而参考 ID 集合是 `["doc_1", "doc_4", "doc_5", "doc_6"]`，那命中 2 个，推测 4 个，分数就是 0.5。

**优势与局限**：
- 计算开销可以忽略不计，不依赖任何外部库
- 完全不看内容，所以对 ID 系统的设计有强依赖——你需要确保 ID 粒度与评估需求匹配
- 不考虑排名顺序，只看集合层面的命中率。所以它衡量的是"检索的质量"而非"排序的质量"

`如果 ID 粒度太粗（比如一个 ID 对应一整本书），这个指标基本没用。如果 ID 粒度是 chunk 级别的，且有可信的标注，那它是性价比最高的方案。`

### E. 各变体对比

| 变体 | 判断方式 | 需要 reference？ | 考量排序？ | 核心依赖 |
|---|---|---|---|---|
| LLMContextPrecisionWithReference | LLM 逐条判 relevance | 是 (ground truth answer) | 是 (AP) | LLM API |
| LLMContextPrecisionWithoutReference / ContextUtilization | LLM 逐条判 relevance（对照生成回答） | 否 | 是 (AP) | LLM API |
| NonLLMContextPrecisionWithReference | 字符串距离匹配（rapidfuzz / Levenshtein） | 是 (reference contexts) | 是 (AP) | rapidfuzz |
| IDBasedContextPrecision | ID 集合取交集 | 是 (reference IDs) | 否 | 无 |

`选择哪一个本质上是在标注成本、计算成本、评估精度三者之间做权衡。LLM 版语义最灵活但最贵，Non-LLM 版需要标注 reference contexts 但确定性强，ID 版最朴素但粒度决定了上限。`

### F. 一个可能被忽略的细节

所有基于 AP 公式的变体（A, B, C）都有一个隐含假设：**每个相关 chunk 对最终生成同等重要**。但实际情况可能是某些 chunk "锦上添花"而另一个 chunk "必不可少"。AP 公式不会区分这种差异——它只看二值的相关/无关。

另外，精度的分母归一化方式意味着：如果检索结果里 10 个 chunk 中有 1 个相关且排在第 1 位，分数就是 1.0（完美）；但如果 10 个 chunk 中有 10 个相关但排在第 1 位的是无关的，分数反而更低。`这很反直觉，但它恰好反映了"排序质量"的含义——一个完美的检索结果应该是所有相关片段紧密地排在前面，中间没有无关片段穿插。`
