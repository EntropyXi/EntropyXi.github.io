---
title: Response Relevancy
description: 讲解Response Relevancy指标如何通过反向生成问题并计算与用户问题的余弦相似度，来评价RAG回答是否切题，并分析其实现与局限。
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
在 RAG 或一般问答系统中，模型生成的回答可能事实正确，却没有真正回答用户的问题；也可能直接回答了问题，但答案本身是错误的。为了区分这两种情况，需要单独评价：**response 与 user input 的意图是否一致。**

Response Relevancy 就是用来评价回答是否切题的指标。当前 Ragas 文档页面使用标题 `Response Relevancy`，但 collections API 中的类名是

```python
AnswerRelevancy
```

其默认指标名仍为

```text
answer_relevancy
```

而旧版 API 使用过 `ResponseRelevancy`。这些名称指向的是同一类核心思想：评价回答与用户问题的相关程度。

它不直接判断事实正确性。假设用户问

```text
法国的首都是哪里？
```

回答

```text
法国的首都是里昂。
```

虽然事实错误，但它确实直接回答了"首都是哪里"，所以 Response Relevancy 仍可能很高。答案是否正确需要由 Factual Correctness、Answer Correctness 等指标评价。

### Construction
##### A. 为什么不直接计算 question 与 response 的相似度
最直接的想法是把用户问题和模型回答分别做 embedding，然后计算余弦相似度：

$$
\cos(E_{\mathrm{question}},E_{\mathrm{response}})
$$

但是问题与回答的语言形式天然不同。例如

```text
问题：埃菲尔铁塔在哪里？
回答：它位于法国巴黎。
```

一个是疑问句，一个是陈述句，而且回答中还可能使用代词或省略问题中的关键词。因此，直接比较 question embedding 和 response embedding，可能无法稳定表示"回答是否回答了这个问题"。

Ragas 采用了反向构造：

> 根据 response 反推出"什么问题可以由这个 response 回答"，然后把反推出的问题与原始 user input 比较。

也就是

$$
\text{response}
\xrightarrow{\text{LLM}}
\text{generated questions}
\xrightarrow{\text{embedding similarity}}
\text{user input}
$$

其基本假设是：**如果 response 真正回答了原问题，那么仅从 response 出发，应当能够重构出与原问题语义接近的问题。**

##### B. 生成反向问题
设原始用户问题为 $q_o$，模型回答为 $r$。LLM 根据 $r$ 生成 $N$ 个问题：

$$
q_{g_1},q_{g_2},\ldots,q_{g_N}
$$

例如

```text
user_input:
Where is the Eiffel Tower located?

response:
The Eiffel Tower is located in Paris, France.
```

可能生成：

```text
Where is the Eiffel Tower located?
In which city can the Eiffel Tower be found?
What city and country is the Eiffel Tower in?
```

这里的"artificial questions"不是人工手写问题，而是 LLM 自动生成的 synthetic questions。

##### C. Embedding 与余弦相似度
将原问题和每个生成问题映射为 embedding：

$$
q_o\longrightarrow E_o
$$

$$
q_{g_i}\longrightarrow E_{g_i}
$$

然后计算

$$
s_i
=
\operatorname{cosine}(E_{g_i},E_o)
=
\frac{E_{g_i}\cdot E_o}
{\|E_{g_i}\|\|E_o\|}
$$

最后取平均：

$$
\operatorname{AnswerRelevancy}
=
\frac{1}{N}\sum_{i=1}^{N}s_i
$$

也就是

$$
\operatorname{AnswerRelevancy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{E_{g_i}\cdot E_o}
{\|E_{g_i}\|\|E_o\|}
$$

其中：

- $E_o$ 是原始 user input 的 embedding；
- $E_{g_i}$ 是第 $i$ 个反向问题的 embedding；
- $N$ 由 `strictness` 控制，默认值为 3。

##### D. 一个数值例子
假设根据回答生成了三个问题，与原问题的余弦相似度分别为

$$
s_1=0.94,\qquad s_2=0.88,\qquad s_3=0.82
$$

那么

$$
\operatorname{AnswerRelevancy}
=
\frac{0.94+0.88+0.82}{3}
=0.88
$$

这表示回答内容反推出的问题，与原始问题整体较为一致。

如果原问题是

```text
埃菲尔铁塔在哪里？
```

但回答是

```text
埃菲尔铁塔于 1889 年建成，由古斯塔夫·埃菲尔设计。
```

LLM 可能反推出

```text
埃菲尔铁塔于哪一年建成？
谁设计了埃菲尔铁塔？
埃菲尔铁塔的设计者是谁？
```

这些问题都围绕建造时间和设计者，而不是位置，因此与原问题的 embedding 相似度会降低，最终 Relevancy 也会降低。

### 当前源码中的额外机制：noncommittal
官方公式只展示了平均余弦相似度，但当前 collections 源码还有一个重要步骤：LLM 在生成问题的同时，会输出

```python
noncommittal: int
```

其含义为：

- `0`：回答具有实质内容；
- `1`：回答含糊、回避或没有作出承诺。

例如

```text
I don't know.
I'm not sure.
It depends.
```

会被提示词视为 noncommittal 回答。

源码会重复调用 LLM `strictness` 次，收集

```python
generated_questions
noncommittal_flags
```

然后计算

```python
all_noncommittal = np.all(noncommittal_flags)
score = cosine_sim.mean() * int(not all_noncommittal)
```

因此实际公式更准确地写成

$$
\operatorname{AnswerRelevancy}
=
\left(
\frac{1}{N}\sum_{i=1}^{N}
\operatorname{cosine}(E_{g_i},E_o)
\right)
\cdot
\mathbf{1}[\text{not all noncommittal}]
$$

如果每一次生成都把 response 判为 noncommittal，则最后分数被强制设为 0。

注意源码使用的是 `all_noncommittal`。也就是说，只要多次判断中至少有一次认为回答是 substantive，分数就不会因为 noncommittal 机制被清零。这会使边界样本受到 LLM 随机性的影响。

### 当前 collections 源码实际流程
完整过程为：

1. 检查 `user_input` 和 `response` 非空；
2. 重复 `strictness` 次调用 LLM；
3. 每次根据 response 生成一个 question，并判断 noncommittal；
4. 对 user input 做一次 embedding；
5. 对所有 generated questions 批量做 embedding；
6. 计算每个生成问题与原问题的余弦相似度；
7. 取平均；
8. 如果全部判断为 noncommittal，则乘以 0；
9. 返回 `MetricResult(value=score)`。

可以表示为

$$
r
\xrightarrow[\times N]{\text{LLM}}
\{(q_{g_i},n_i)\}_{i=1}^{N}
$$

$$
(q_o,q_{g_1},\ldots,q_{g_N})
\xrightarrow{\text{embedding}}
(E_o,E_{g_1},\ldots,E_{g_N})
$$

$$
\{E_{g_i}\},E_o
\xrightarrow{\text{cosine mean + noncommittal gate}}
\text{score}
$$

`strictness=3` 并不是让一个 LLM 请求一次返回三个问题，而是当前源码使用循环执行三次结构化生成。因此，`strictness` 增大通常会线性增加 LLM 调用次数、费用和延迟，同时能在一定程度上减少单次生成问题带来的偶然性。

### 为什么余弦相似度可能为负数
Embedding 向量的分量并不要求全部为正。例如

$$
E_1=(0.8,-0.3,0.5)
$$

$$
E_2=(-0.6,0.2,0.4)
$$

余弦相似度的分母

$$
\|E_1\|\|E_2\|
$$

始终非负，但点积

$$
E_1\cdot E_2
$$

可以为负。因此数学上

$$
-1\leq \operatorname{cosine}(E_1,E_2)\leq1
$$

当前实现直接对余弦相似度取平均，没有执行

$$
\frac{\cos\theta+1}{2}
$$

这样的区间变换，也没有把结果裁剪到 $[0,1]$。所以尽管官方文档说分数通常在 0 到 1 之间，理论上仍然可能出现负值。

这也形成了一个需要注意的实现细节：类的说明文字将分数描述为 0 到 1，但公式和源码没有严格保证这一点。实际文本 embedding 通常使相关或普通句子的相似度落在正数范围，所以负分较少见，但并非不可能。

### 它为什么能够惩罚不完整回答
官方示例的问题为

```text
Where is France and what is its capital?
```

低相关回答为

```text
France is in western Europe.
```

高相关回答为

```text
France is in western Europe and Paris is its capital.
```

从低相关回答反推的问题大多只会围绕"法国位于欧洲哪里"，无法恢复"法国首都是哪里"这一部分。于是反向问题与原始双重问题的语义覆盖不完整，相似度会降低。

因此，它对不完整回答的惩罚不是通过逐条检查原问题的子问题完成的，而是通过下面的间接机制产生：

$$
\text{response 缺失一部分意图}
\Longrightarrow
\text{反向问题也缺失该部分}
\Longrightarrow
\text{与原问题 embedding 不完全一致}
$$

同理，回答加入大量无关细节时，生成问题可能转向这些细节，从而拉低平均相似度。但是这种惩罚是间接的，并不保证每次都发生。

### 它评价什么，不评价什么
Response Relevancy 评价的是：

> 回答的内容是否直接、完整地对齐用户问题的意图。

它不评价：

- 回答事实是否正确；
- 回答是否被 retrieved contexts 支持；
- 检索器是否召回了完整证据；
- 回答是否包含幻觉。

例如：

```text
问题：法国的首都是哪里？
回答：法国的首都是里昂。
```

这个回答错误，但反向生成的问题仍可能是

```text
What is the capital of France?
```

它与原问题高度相似，所以 Relevancy 可能很高。

相反：

```text
问题：法国的首都是哪里？
回答：法国位于西欧，拥有丰富的历史和文化。
```

内容大体正确，但没有回答首都，因此 Relevancy 较低。

可以将几个指标分开理解：

- Response Relevancy：有没有回答用户问的内容；
- Factual Correctness：回答内容是否正确；
- Faithfulness：回答是否能够由检索上下文推出；
- Context Recall：检索上下文有没有漏掉答案需要的事实；
- Context Precision：检索结果是否混入大量无关内容，以及相关结果排序是否合理。

### 指标的局限
#### A. 依赖 LLM 生成问题
生成问题不是唯一的。同一个 response 在不同运行中可能得到不同问题，从而导致分数波动。LLM 还可能加入 response 中没有的信息，或者生成过于宽泛的问题。

#### B. 依赖 embedding 模型
最终相似度由 embedding 空间决定。不同 embedding 模型对领域术语、多语言表达和否定关系的敏感度不同，所以不能把不同 embedding 配置下的绝对分数直接横向比较。

#### C. 对事实错误不敏感
只要回答形式上对准了问题，即使答案事实错误，也可能取得高分。这不是指标失效，而是因为它本来就不负责正确性。

#### D. 对否定和细微语义差异可能不敏感
例如

```text
巴黎是法国首都。
巴黎不是法国首都。
```

两句话共享几乎全部词语，某些 embedding 模型仍可能给出较高相似度。因此语义相似度不能代替逻辑真值判断。

#### E. 长回答中的无关信息不一定稳定受罚
文档说该指标会惩罚不必要细节，但实现没有显式计算"无关事实比例"。只有当这些细节影响 LLM 反向生成的问题时，分数才会下降。

#### F. 分数不是严格概率
0.9 不表示"回答有 90% 的概率相关"。它只是特定 LLM、特定 embedding、特定 `strictness` 和当前提示词下的平均几何相似度。

### 使用与解读建议
第一，固定评价配置，包括 judge LLM、embedding 模型、提示词版本和 `strictness`。否则不同实验之间的分数不可直接比较。

第二，在模型或 prompt 对比实验中，更关注同一配置下的相对变化，而不是把某个固定阈值视为普遍标准。

第三，至少同时配合一个正确性指标和一个 groundedness/faithfulness 指标。高 Relevancy 只能证明"切题"，不能证明"正确且有依据"。

第四，抽样查看 generated questions。它们是解释分数的关键中间变量：

- 生成问题偏离原问题，说明 response 可能跑题；
- response 明明切题但生成问题偏离，说明 judge LLM 出错；
- 生成问题很接近但事实答案错误，说明需要正确性指标补充。

第五，对中文或专业领域数据，使用适合该语言和领域的 embedding 模型。否则相似度可能主要反映词面重合，而不是实际意图。

第六，`strictness` 不是越大越好。增大它会增加成本，并可能生成更多重复问题。应通过验证集观察分数稳定性，再决定是否从默认的 3 调高。

### API 示例
```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=client,
)

scorer = AnswerRelevancy(
    llm=llm,
    embeddings=embeddings,
    strictness=3,
)

result = await scorer.ascore(
    user_input="Where is France and what is its capital?",
    response="France is in western Europe and Paris is its capital.",
)

print(result.value)
```

这个指标只需要 `user_input` 和 `response`，不需要 `reference` 或 `retrieved_contexts`。这正说明它评价的是回答与问题之间的对齐关系，而不是检索质量或参考答案正确性。

同步调用可以写成

```python
result = scorer.score(
    user_input=...,
    response=...,
)
```

异步调用写成

```python
result = await scorer.ascore(
    user_input=...,
    response=...,
)
```

### 总结
Response Relevancy 的核心逻辑可以写成

$$
\boxed{
\text{response}
\xrightarrow{\text{反向生成问题}}
\text{generated questions}
\xrightarrow{\text{与 user input 比较}}
\text{平均余弦相似度}
}
$$

当前 collections 实现还会判断回答是否 noncommittal，并在所有生成结果都判为回避性回答时把分数归零。它衡量的是"回答是否切题"，而不是"回答是否正确"。因此，最合理的用法不是单独依赖该分数，而是把它与正确性、Faithfulness、Context Recall 和 Context Precision 组合起来，形成对 RAG 检索与生成过程的分层评价。

### References
- [Ragas: Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)
- [AnswerRelevancy collections implementation](https://github.com/vibrantlabsai/ragas/blob/main/src/ragas/metrics/collections/answer_relevancy/metric.py)
- [AnswerRelevancy structured prompt](https://github.com/vibrantlabsai/ragas/blob/main/src/ragas/metrics/collections/answer_relevancy/util.py)
