---
title: Context Recall
description: 介绍RAG检索阶段的Context Recall指标，说明其衡量参考答案事实被检索上下文覆盖的程度，并解析LLM-based实现的原理与局限。
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
在 RAG 中，检索器会根据用户问题返回一组 `retrieved_contexts`，然后生成模型再依据这些上下文回答问题。这里最基本的问题是：**正确答案所需要的信息，是否真的被检索出来了？**

Context Recall（上下文召回率）就是用来回答这个问题的。它不关心检索结果中混入了多少无关内容，也不关心相关片段排在第几位，而只关心：**参考答案中应当被覆盖的事实，有多少能够从检索上下文中得到支持。**

因此，Context Recall 是一个检索阶段指标。它评价的是

$$
\text{reference 中需要的事实}
\longrightarrow
\text{retrieved\_contexts 是否覆盖}
$$

它与 Context Precision 的区别是：

- Context Recall 关注"有没有漏掉重要信息"；
- Context Precision 关注"检索结果中垃圾多不多，以及相关内容是否排在前面"。

一个系统可以同时具有高 Recall 和低 Precision。例如把整个知识库都返回，关键事实大概率不会漏掉，但是无关上下文会非常多。

### Construction
##### A. 输入与评价对象
当前 collections API 的调用形式为

```python
result = await scorer.ascore(
    user_input=...,
    retrieved_contexts=...,
    reference=...
)
```

其中：

- `user_input` 是用户问题；
- `retrieved_contexts` 是检索器输出的多个 chunk；
- `reference` 是人工给出的参考答案。

这里没有直接传入真正的 `reference_contexts`。原因是，人工标注"哪些原始文档片段属于完整正确证据"通常很昂贵，所以 Ragas 使用 `reference` 作为 `reference_contexts` 的代理：先观察参考答案包含哪些事实，再判断这些事实能不能从检索结果中推出。

所以 LLM-based Context Recall 实际测量的不是严格的"相关文档召回率"，而是：

> **参考答案中的事实主张，有多少能够归因于检索上下文。**

这两者通常有关联，但并不完全相同。一个 chunk 可能没有逐字出现参考答案，却能够语义上支持其中的事实；反过来，一个 chunk 可能出现了相同关键词，却没有提供足够的逻辑支持。

##### B. 将 reference 看成若干事实主张
设参考答案可以拆成 $M$ 条事实主张：

$$
C=\{c_1,c_2,\ldots,c_M\}
$$

对于每条事实 $c_i$，让 LLM 判断它是否能够由所有检索上下文共同支持。定义

$$
a_i=
\begin{cases}
1, & c_i \text{ 可以归因于 retrieved contexts}\\
0, & c_i \text{ 不可以归因于 retrieved contexts}
\end{cases}
$$

则 Context Recall 为

$$
\operatorname{ContextRecall}
=
\frac{1}{M}\sum_{i=1}^{M}a_i
$$

也就是

$$
\operatorname{ContextRecall}
=
\frac{\text{被检索上下文支持的 reference 主张数}}
{\text{reference 主张总数}}
$$

这个公式说明，每个 reference claim 在最终分数中的权重相同。它不会因为某条事实更重要，就自动赋予更高权重。

##### C. 一个完整例子
用户问题为：

```text
Where is the Eiffel Tower and when was it completed?
```

参考答案为：

```text
The Eiffel Tower is located in Paris and was completed in 1889.
```

可以拆成两条事实：

$$
c_1=\text{埃菲尔铁塔位于巴黎}
$$

$$
c_2=\text{埃菲尔铁塔于 1889 年建成}
$$

假设检索器只返回：

```text
The Eiffel Tower is located in Paris, France.
```

那么 LLM 的理想判断是

$$
a_1=1,\qquad a_2=0
$$

所以

$$
\operatorname{ContextRecall}
=
\frac{1+0}{2}
=0.5
$$

如果再检索到

```text
Construction of the Eiffel Tower was completed in 1889.
```

则

$$
a_1=1,\qquad a_2=1
$$

从而

$$
\operatorname{ContextRecall}=1
$$

注意，召回率为 1 只说明答案所需的信息都找到了，并不说明检索结果中没有垃圾。例如再额外返回十个完全无关的 chunk，Context Recall 仍然可以保持为 1，但 Context Precision 会下降。

#### D. 当前 collections 源码实际做了什么
当前实现并没有先调用一个独立函数，把 `reference` 显式拆分成 claims，然后逐条再次调用 LLM。它把

```text
question = user_input
context  = "\n".join(retrieved_contexts)
answer   = reference
```

一起送入一个结构化提示词。提示词要求 LLM：

1. 分析 `answer` 中的每条 statement；
2. 为每条 statement 给出解释；
3. 输出二元分类 `attributed=1/0`。

结构化输出近似为

```python
class ContextRecallClassification:
    statement: str
    reason: str
    attributed: int
```

获得全部分类以后，源码直接计算

```python
attributions = [c.attributed for c in result.classifications]
score = sum(attributions) / len(attributions)
```

因此，所谓"拆分 reference 并判断每条事实"，在当前实现中是由一次结构化 LLM 调用共同完成的，而不是一个完全确定、可复现的规则算法。

完整流程可以概括为

$$
\text{user input, reference, contexts}
\xrightarrow{\text{LLM}}
\{(c_i,\text{reason}_i,a_i)\}_{i=1}^{M}
\xrightarrow{\text{平均}}
\text{Context Recall}
$$

如果 LLM 没有返回任何 classification，源码会返回 `NaN`。如果 `user_input`、`reference` 或 `retrieved_contexts` 为空，当前 API 会直接抛出 `ValueError`。

### 对官方 Eiffel Tower 示例的分析
官方文档给出的示例是

```python
user_input="Where is the Eiffel Tower located?"
retrieved_contexts=["Paris is the capital of France."]
reference="The Eiffel Tower is located in Paris."
```

文档展示的输出为

```text
Context Recall Score: 1.0
```

但是严格按照"只允许依据 retrieved context 判断"的标准，检索内容

```text
Paris is the capital of France.
```

并不能推出

```text
The Eiffel Tower is located in Paris.
```

两句话只是共享了 `Paris`，前者没有提供埃菲尔铁塔位置的证据。因此理想的归因判断应当是

$$
a_1=0
$$

从而

$$
\operatorname{ContextRecall}=0
$$

文档中的 1.0 更适合作为一个警告：**Context Recall 的公式是确定的，但是 claim 的拆分和归因判断由 LLM 完成，因此结果不一定严格可靠。** 评价模型可能利用自己的世界知识，也可能因为关键词或语义接近而做出过宽松判断。

这不是"程序直接看到 Paris 就加一分"，而是 LLM judge 在结构化提示词下作出了错误或过度宽松的归因。

### 三种 Context Recall
#### A. LLM-based Context Recall
输入：

```text
reference + retrieved_contexts
```

通过 LLM 判断 reference 中的事实是否被支持。优点是可以处理改写、同义表达和隐含语义；缺点是成本较高，并且存在 LLM judge 偏差。

#### B. Non-LLM Context Recall
这种形式直接比较

```text
retrieved_contexts + reference_contexts
```

使用非 LLM 字符串相似度或距离指标，判断每个参考上下文是否被检索到。其概念公式为

$$
\operatorname{Recall}
=
\frac{\text{检索到的相关 reference contexts 数}}
{\text{reference contexts 总数}}
$$

它不需要 LLM，成本较低、可复现性更强，但是对改写和语义等价的处理能力取决于所选字符串指标。

#### C. ID-based Context Recall
如果每个文档或 chunk 都具有稳定的唯一 ID，可以直接计算

$$
\operatorname{IDRecall}
=
\frac{|I_{\mathrm{retrieved}}\cap I_{\mathrm{reference}}|}
{|I_{\mathrm{reference}}|}
$$

这种方法不比较文本内容，只比较 ID。它最确定、最快，也最适合拥有人工标注 gold document IDs 的检索测试集。

### Context Recall 能说明什么
Context Recall 高，说明参考答案需要的事实大部分能够在检索结果中找到。通常意味着：

- top-k 足够覆盖关键证据；
- query 与文档的匹配没有严重漏召回；
- chunk 切分没有把关键语义完全破坏；
- hybrid search、query expansion 等召回策略可能有效。

Context Recall 低，说明答案需要的信息存在缺失。常见原因包括：

1. `top_k` 太小；
2. embedding 对当前领域语义建模不足；
3. chunk 太小，事实被切断；
4. chunk 太大，关键信息在向量表示中被稀释；
5. 用户问题需要多跳证据，但检索器只找到了其中一步；
6. reference 本身包含知识库中不存在的信息；
7. LLM judge 对 claim 的拆分或归因发生错误。

因此，Context Recall 低并不能直接证明"向量数据库不好"，必须结合具体失败样本观察到底是检索失败、数据缺失、标注问题，还是评价模型误判。

### Context Recall 不评价什么
Context Recall 不直接评价：

- 检索结果中有多少无关 chunk；
- 相关 chunk 是否排在前面；
- 最终 response 是否忠实于上下文；
- 最终 response 是否正确、完整或切题。

例如，检索器已经找到了全部证据，但生成模型仍然可能胡编。此时 Context Recall 可以为 1，而 Faithfulness 很低。

可以将几个常见指标放在同一个流程中理解：

$$
\text{Retriever}
\xrightarrow[\text{有没有漏证据}]{\text{Context Recall}}
\text{Contexts}
\xrightarrow[\text{垃圾和排序}]{\text{Context Precision}}
\text{LLM}
\xrightarrow[\text{是否依据上下文}]{\text{Faithfulness}}
\text{Response}
$$

### 使用与解读建议
第一，不要只看数据集平均分。应当保留每个样本的 reference、retrieved contexts、claim 分类和 reason，人工检查低分和异常高分样本。

第二，尽量使用确定性较高、能力足够的 judge 模型，并把温度设置得较低。即使如此，LLM-based 指标仍然不是数学真值。

第三，当拥有 gold chunk IDs 时，优先同时计算 ID-based Recall。它可以作为更稳定的基准，帮助发现 LLM judge 是否偏离。

第四，Context Recall 应和 Context Precision 一起看：

- Recall 低、Precision 高：检索得很干净，但漏掉了证据；
- Recall 高、Precision 低：证据找到了，但同时返回大量噪声；
- 两者都低：召回和排序整体存在问题；
- 两者都高：检索阶段总体较理想。

第五，Context Recall 的上限受 reference 质量限制。如果 reference 包含多个不必要事实，或者事实粒度不一致，分母就会改变，导致不同样本之间的分数不再完全可比。

### API 示例
```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
scorer = ContextRecall(llm=llm)

result = await scorer.ascore(
    user_input="Where is the Eiffel Tower and when was it completed?",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris, France."
    ],
    reference=(
        "The Eiffel Tower is located in Paris "
        "and was completed in 1889."
    ),
)

print(result.value)
```

这里的 `retrieved_contexts` 在真实 RAG 中应当由 retriever 输出。示例中手写只是为了单独测试指标，而不是完整执行检索流程。

同步调用可以使用

```python
result = scorer.score(...)
```

异步调用使用

```python
result = await scorer.ascore(...)
```

二者评价逻辑相同，差别主要在调用方式和并发能力。

### 总结
Context Recall 的核心逻辑可以压缩为

$$
\boxed{
\text{Context Recall}
=
\frac{\text{reference 中被 contexts 支持的事实}}
{\text{reference 中全部事实}}
}
$$

它评价的是 RAG 检索阶段有没有漏掉答案所需的信息。当前 Ragas 的 LLM-based 实现通过结构化提示词，让 LLM 同时完成 statement 拆分和 attribution 分类，再对二元结果取平均。因此，公式虽然简单，但实际分数会受到参考答案粒度、judge 模型能力、提示词和随机性的共同影响。

### References
- [Ragas: Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)
- [ContextRecall collections implementation](https://github.com/vibrantlabsai/ragas/blob/main/src/ragas/metrics/collections/context_recall/metric.py)
- [ContextRecall structured prompt](https://github.com/vibrantlabsai/ragas/blob/main/src/ragas/metrics/collections/context_recall/util.py)
