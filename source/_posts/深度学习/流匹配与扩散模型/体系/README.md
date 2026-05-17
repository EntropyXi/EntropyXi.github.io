这个体系的构建可能不会很细节，我们将会围绕以下几个问题展开：
1. **扩散模型究竟在学什么对象？**
2. **Flow Matching 与扩散模型到底是并列关系、包含关系，还是重参数化关系？**
3. **从 DDPM、Score SDE、Probability Flow ODE、CNF、Flow Matching，怎样串成一条单一逻辑链？**

在这里我们先用一句话总结这个体系的理论流：

**扩散模型**首先通过前向加噪过程定义一条从数据分布到简单先验的连续概率路径，并通过学习各时刻边缘分布的 **score** 来构造逆时间生成动力学；在连续时间极限下，这一过程由 **Score-based SDE** 统一描述。进一步地，由 **Fokker–Planck equation** 可知，同一族边缘分布路径并不唯一对应于某个随机动力学，而可对应一族等价动力学；其中零扩散极限给出 **Probability Flow ODE**，从而把扩散模型嵌入 **Continuous Normalizing Flow** 的确定性输运框架。**Flow Matching** 则进一步把视角提升到“先选概率路径，再直接学习其速度场”，于是扩散路径成为其可选路径家族的一个特例，而 OT displacement interpolation 等非扩散路径则构成更广泛的设计空间。故而，从理论上看，DDPM、Score SDE、Probability Flow ODE、CNF 与 Flow Matching 并非彼此割裂的方法，而是围绕“连续概率路径的构造、参数化与求解”这一核心问题的不同坐标系。

本文的体系基于 _[https://spaces.ac.cn/archives/9209](https://spaces.ac.cn/archives/9209 "生成扩散模型漫谈（五）：一般框架之SDE篇")_与_[https://spaces.ac.cn/archives/9228](https://spaces.ac.cn/archives/9228 "生成扩散模型漫谈（六）：一般框架之ODE篇")_构建

推导过程与理论完整性存在不足之处，请见谅