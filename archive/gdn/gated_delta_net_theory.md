
# Gated Delta Networks：数学理解与基础代码实现

本文介绍Gated Delta Networks数学原理与基础的torch recurrent/chunk 实现。

## 普通Attention与Linear Attention

标准的transformer softmax attention

$$
\begin{equation}
O = \text{softmax}(QK^\top)V
\end{equation}
$$

展开第$`t`$个token：

$$
\begin{equation}
o_t = \sum_{i=1}^t v_i \cdot \text{softmax}(k_i^\top q_t)
\end{equation}
$$

意思是query_t和$`k_i`$上的投影（算相似度）然后softmax变概率和value $`v_i`$ 加权

## Linear Transformer

把 attention 改写为

$$
\begin{equation}
o_t = \sum_{i=1}^t v_i (k_i^\top q_t)
\end{equation}
$$

因为**矩阵的乘法的结合律**:

$$
\begin{equation}
v_i(k_i^\top q_t) = (v_i k_i^\top) q_t
\end{equation}
$$

于是可以写成：

$$
\begin{equation}
o_t = \left(\sum_{i=1}^t v_i k_i^\top \right) q_t
\end{equation}
$$

定义$`S`$,相当于把历史信息压缩成一个 memory matrix

$$
\begin{equation}
S_t = \sum_{i=1}^t v_i k_i^\top
\end{equation}
$$

于是：

$$
\begin{equation}
o_t = S_t q_t
\end{equation}
$$

以上就是linear attention的简化形式
这里似乎标准attention里的softmax被扔掉了, linear attention额外引入了 kernel feature map 起到了相同的作用，有兴趣的话可以参考论文[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)，这里不多赘述。

## Gated Delta Net

GDN在linear attention基础上引入两个gate

 **memory 更新**：

$$
\begin{equation}
S_t =
S_{t-1}(\alpha_t(I-\beta_t k_t k_t^\top))
+
\beta_t v_t k_t^\top
\end{equation}
$$

**attention**输出

$$
\begin{equation}
o_t = S_t q_t
\end{equation}
$$

$`\alpha_t S_{t-1}`$ 这步控制整体记忆衰减 $`\alpha`$趋近于1几乎不忘，趋近于0几乎全忘。

$`(I-\beta_t k_t k_t^\top)`$ 可以理解为删除 memory 在当前key也就是$`k_t`$方向上的旧信息。$`k_t k_t^\top`$是当前key的投影矩阵(假设$`k`$已经完全归一化)，

有趣的是 如果$`\beta=1`$会类似施密特正交化 Gram–Schmidt orthogonalization (memory在$`k_t`$方向被完全清空)，旧的$`k_t`$的旧信息被清空，通过$`\beta_t v_t k_t^\top`$写入新的value，从而实现该key的完全覆盖。

反之$`\beta=0`$退化成$`S_t={\alpha}S_{t-1}`$完全没记忆更新。

总结一下两个gate系数的作用

| 参数 | 作用                  |
| -- | ------------------- |
| β  | 控制 **当前 key 的更新强度** |
| α  | 控制 **整体记忆衰减**       |

### 附加: kkt理解
memory更新

$$
S_t =
S_{t-1}(\alpha_t(I-\beta_t k_t k_t^\top))
+
\beta_t v_t k_t^\top
$$

最不直观的就是$`k_t k_t^\top`$这项的理解，从纯几何上简单解释，为了方便理解这里假设$`k_t`$都是单位向量，满足

$$
\begin{equation}
\Vert k_t \Vert = 1
\end{equation}
$$

先看一下$`S`$的维度

$$
\begin{equation}
S\in\mathbb{R}^{d_v\times{d_k}},k_t\in\mathbb{R}^{d_k}
\end{equation}
$$

那么

$$
\begin{equation}
P= k_t k_t^\top, P\in\mathbb{R}^{d_k\times{d_k}}
\end{equation}
$$

就是投影矩阵,对向量$`x`$且$`x\in\mathbb{R}^{d_k}`$

$$\begin{equation}
Px= k_t k_t^\top{x}=(k_t^\top{x})k_t
\end{equation}
$$

这里$`k_t^\top{x}`$是一个内积（结果为标量），是$`x`$在$`k_t`$方向投影的长度，在乘以单位向量$`k_t`$就是$`x`$在$`k_t`$方向的**向量投影**。

再看$`S_{t-1}k_t k_t^\top`$

首先$`S_{t-1}k_t`$的结果是个维度为$`d_v`$的向量, 这是状态矩阵 $`S`$ 对当前键 $`k_t`$ 的“检索”结果。因为它是一个矩阵乘以一个向量，结果是一个向量（我们可以把它看作是从记忆中提取出的旧特征）.

$`S_{t-1}k_t k_t^\top`$维度为$`d_v\times{d_k}`$ ,又重新投影到$`k_t`$的方向上。

### Recurrent版本GDN的torch实现
Run这段代码 运行debug torch版本的gdn [troch impl commit](https://github.com/apinge/flash-linear-attention/commit/4911d4c410f67c8a47a3244a2e07a84de3cec688)

为了方便理解 下面加了很多打印的尺寸 ，来自[Qwen3.5-397B-A17B config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)，tp8的log
```python
def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional 
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    #[ naive_recurrent_gated_delta_rule] 
    # after transpose:
    # q.shape=torch.Size([1, 8, 8000, 128]),
    # k.shape=torch.Size([1, 8, 8000, 128]),
    # v.shape=torch.Size([1, 8, 8000, 128]),
    # beta.shape=torch.Size([1, 8, 8000]),
    # g.shape=torch.Size([1, 8, 8000])
    B, H, T, K, V = *k.shape, v.shape[-1]
    """
    这里 B，H T K  V分别代表 batch, attention head num, token len, d_k,d_v
    [naive_recurrent_gated_delta_rule] B=1, H=8, T=8000, K=128, V=128
    对应config里的(https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64, 
    "linear_value_head_dim": 128,
    head num 的8是64//8
    """

    o = torch.zeros(B, H, T, V).to(v)
  
    h = torch.zeros(B, H, K, V).to(v)
   
    if initial_state is not None:
        h = initial_state.to(torch.float32)
        # 走了这个分支
        # 这里的H就是公式里的S的transpose，表示memory的状态
        # [naive_recurrent_gated_delta_rule] initial_state set: h.shape=torch.Size([8, 128, 128])
        

    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5) # 这就是 1/sqrt(d) MHA 里也是类似这种
    q = q * scale
    # scale=0.08838834764831843 是1/sqrt(128)

    # 循环每个token
    for i in range(T):
        b_q = q[:, :, i] # torch.Size([1, 8, 128]) $q_t$ 
        b_k = k[:, :, i]  # torch.Size([1, 8, 128])  $k_t$
        b_v = v[:, :, i].clone()  # torch.Size([1, 8, 128])  $v_t$
        """
        g[:, :, i].shape = [1,8]
        h.shape = [8,128,128] 
        g[:, :, 0].exp()[..., None, None].shape=torch.Size([1, 8, 1, 1])
        这里是每128X128的矩阵乘以一个标量g
        公式里的$\alpha$和g的关系： $\alpha_t=\exp(g_t)$
        这样写是为了保证$\alpha_t>0$

        注意之前公式的S都是[V,K]这里h实际上是公式S的transpose shape为[K,V]
        """
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i] # [1,8]
        # h.shape = torch.Size([1, 8, 128, 128])
        # b_k[..., None].shape = torch.Size([1, 8, 128, 1])
        # b_v.shape = torch.Size([1, 8, 128])
        
        """
         (h.clone() * b_k[..., None]).sum(-2)对应$S_{t-1}k_t$
        h:[K,V]
        b_k[..., None]:[K,1]
        [K,V] * [K,1] → [K,V]
        -2维度对应K维度，也就是对K求和
        这行实际上是$h^Tk$
        $(h*k[:,None])_{i,j}=h_{i,j}k_i$
        $\sum_{i}h_{i,j}k_i$

        从矩阵角度 h^T:[V,K]
        [V,K]@[K]
        为[V]
        """
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]


        """ 这句对应 
         $$S_{t+1} = S_t + k_t \otimes \Delta v_t$$
        b_k.unsqueeze(-1)：把形状从 [1, 8, 128] 变成 [1, 8, 128, 1]。
        这代表一个列向量。
        b_v.unsqueeze(-2)：把形状从 [1, 8, 128] 变成 [1, 8, 1, 128]。
        这代表一个行向量。
        h + ... 将这个新生成的“增量矩阵”叠加到旧的记忆矩阵上
        此时 h_size = [1, 8, 128, 128]

        b_k : [K]
        b_v : [V]

        b_k[:,None] : [K,1]
        b_v[None,:] : [K,1]
        outer produce
        [K,1][1,V]=[K,V]
        """
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        """
        torch.einsum 是 Einstein Summation（爱因斯坦求和约定）
        b: Batch size (1)
        h: Head num (8)
        d: $K$ 维度，也就是特征长度 (128)
        m: $V$ 维度，也就是输出长度 (128)
        bq = [1, 8, 128]
        h_size = [1, 8, 128, 128]
        b_q（一个 128 维的向量）。你的笔记本是 h（一个 $128 \times 128$ 的映射矩阵）。
        # 如果不用 einsum，你会这么写：
        # 1. 给 b_q 补一个维度变成 [1, 8, 1, 128]
        # 2. 跟 h [1, 8, 128, 128] 做矩阵乘法
        # 3. 得到 [1, 8, 1, 128]，再把那个 1 删掉
        o[:, :, i] = torch.matmul(b_q.unsqueeze(-2), h).squeeze(-2)
        $o = q^\top H$ 数学上

        再从数学上看， 忽略batch 和head num就是前两维 'd,dm->m'

        b_q : [d]   = [K]
        h   : [d,m] = [K,V]
        看的出是对K维度就和 输出是[V]
        $o=q^Th$
        """
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)
       # if i == 0 or i == T - 1:
            # Step i=0: b_q.shape=torch.Size([1, 8, 128]), b_k.shape=torch.Size([1, 8, 128]), b_v.shape=torch.Size([1, 8, 128]), h.shape=torch.Size([1, 8, 128, 128]), o[:,:,i].shape=torch.Size([1, 8, 128])

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    # [naive_recurrent_gated_delta_rule] return: o.shape=torch.Size([1, 8000, 8, 128]), h.shape=torch.Size([1, 8, 128, 128])
    return o, h
```

### Chunk形式 

从前面看出，虽然linear attention把复杂度降低了，Recurrent 是线性的，但是难并行，因此提出了 chunkwise parallel，它将输入输出划分为若干个大小为的 chunks (沿着token len维度分块，比如token len为8000， chunk为64 就是125个chunk)，并根据前一个块的最终状态以及当前块的 K  Q  V 来计算输出。

先直接上结论

**memory**

$$\begin{equation}
\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]} } + \left( \widetilde{\mathbf{U}_{[t]} } - \overleftarrow{\mathbf{W}_{[t]} } \mathbf{S}{[t]}^\top \right)^\top \overrightarrow{\mathbf{K}_{[t]} } \in \mathbb{R}^{d_v \times d_k}
\end{equation}$$

**attention**

$$\begin{equation}
\mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}_{[t]} } \mathbf{S}_{[t]}^\top + \left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{\Gamma} \right) \left( \widetilde{\mathbf{U}_{[t]} } - \overleftarrow{\mathbf{W}_{[t]} } \mathbf{S}_{[t]}^\top \right) \in \mathbb{R}^{C \times d_v}
\end{equation}
$$

其中 solve $`\Delta{v}`$ system

$$\begin{equation}
\widetilde{\mathbf{U}_{[t]} } = \left[ \mathbf{I} + \text{strictLower}\left( \text{diag}(\beta_{[t]}) (\mathbf{\Gamma}_{[t]} \odot \mathbf{K}_{[t]} \mathbf{K}^\top_{[t]}) \right) \right]^{-1} \text{diag}(\beta_{[t]}) \mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}
\end{equation}
$$

衰减因子矩阵

$$\begin{equation}
\mathbf{\Gamma}_{[t]} = 
\begin{cases}
\frac{\gamma^r_{[t]} }{\gamma^i_{[t]} }, & i \leq r \\
0, & i > r
\end{cases} \in \mathbb{R}^{C \times C}
\end{equation}
$$

key correction

$$\begin{equation}
\overleftarrow{\mathbf{W}_{[t]} } = \text{diag}(\gamma^i_{[t]}) \mathbf{W}_{[t]} \in \mathbb{R}^{C \times d_k}
\end{equation}
$$
$$
\begin{equation}
\mathbf{W}_{[t]} = \left[ \mathbf{I} + \text{strictLower} \left( \text{diag}(\beta_{[t]}) (\mathbf{K}_{[t]} \mathbf{K}^\top_{[t]}) \right) \right]^{-1} \text{diag}(\beta_{[t]}) \mathbf{K}_{[t]} \in \mathbb{R}^{C \times d_k}
\end{equation}
$$

memory decay

$$
\begin{equation}
\overrightarrow{\mathbf{S}_{[t]} } = \gamma_{[t]}^C \mathbf{S}_{[t]} \in \mathbb{R}^{d_v \times d_k}
\end{equation}
$$
$$
\begin{equation}
\overrightarrow{\mathbf{K}_{[t]} } = \text{diag}\left(\frac{\gamma^C_{[t]} }{\gamma^i_{[t]} }\right) \mathbf{K}_{[t]} \in \mathbb{R}^{C \times d_k}
\end{equation}
$$
$$
\begin{equation}
\overleftarrow{\mathbf{Q}_{[t]} } = \text{diag}(\gamma^i_{[t]}) \mathbf{Q}_{[t]} \in \mathbb{R}^{C \times d_k}
\end{equation}
$$

其中

$$
\begin{equation}
\gamma^j_{[t]} = \prod_{j=tC+1}^{tC+j} \alpha_j
\end{equation}
$$


它和全局 $`\gamma`$ 的关系是 $`\gamma_{tC+j} / \gamma_{tC}`$，本质上是从 chunk 内起点到当前位置的相对累乘.

### 解释attention

$$\begin{equation}
\mathbf{O}_{[t]} = \underbrace{\overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t]}^\top}_{\text{历史 memory 的投影}} + \underbrace{(\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{\Gamma}_{[t]}) (\widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t]}^\top)}_{\text{当前 step delta 对输出的贡献}}\end{equation}
$$

#### $`\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \Gamma_{[t]}`$

* $`\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \in \mathbb{R}^{C \times C}`$ — query-key 相似度

* $`\mathbf{\Gamma}_{[t]} \in \mathbb{R}^{C \times C}`$ — mask / lower-triangular 比例，保证只用过去信息


$$
\Gamma_{[t], i,j} =
\begin{cases}
\gamma^j_{[t]} / \gamma^i_{[t]}, & i \le j \\
0, & i > j
\end{cases}
$$

* 结果矩阵仍是 $`C \times C`$，对应 PyTorch 中 `g`、`chunk_local_cumsum` 累积后的 gating。

---

#### $`\widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t]}^\top`$

* $`\widetilde{\mathbf{U}_{[t]}} \in \mathbb{R}^{C \times d_v}`$ — 新的 value 贡献，等价 PyTorch 中 `b_v` 经过 beta/gate 和投影后的值



$$
\widetilde{\mathbf{U}_{[t]}} = \left[ I + \text{strictLower}( \text{diag}(\beta) (\Gamma \odot K K^\top) ) \right]^{-1} \text{diag}(\beta) V
$$

* $`\overleftarrow{\mathbf{W}_{[t]}} = \text{diag}(\gamma^i_{[t]}) W_{[t]} \in \mathbb{R}^{C \times d_k}`$

$$W_{[t]} = [I + \text{strictLower}(diag(\beta) K K^\top)]^{-1} diag(\beta) K$$

* 于是 $`\overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t]}^\top`$ 就是“历史 memory 对当前增量的投影”

### 附加： chunk wise形式推导
我们从**memory 更新**和**attention**输出两个式子出发：

$$\begin{equation}
S_t =
S_{t-1}(\alpha_t(I-\beta_t k_t k_t^\top))
+
\beta_t v_t k_t^\top  \end{equation}
$$
$$
\begin{equation}
o_t = S_t q_t\end{equation}
$$

其中

$$\begin{equation}
S_t\in\mathbb{R}^{d_v\times d_k},\quad
k_t,q_t\in\mathbb{R}^{d_k},\quad
v_t\in\mathbb{R}^{d_v}
\end{equation}
$$

* 先把式子写成delta rule形式

$$\begin{equation}
S_t
=\alpha_t S_{t-1}-\alpha_t S_{t-1}\beta_t k_tk_t^\top+
\beta_t v_tk_t^\top
\end{equation}
$$

整理：

$$
\begin{equation}
S_t=
\alpha_t S_{t-1}
+
\beta_t (v_t-\alpha_t S_{t-1}k_t)k_t^\top
\end{equation}
$$

定义

$$
\begin{equation}
\Delta v_t=\beta_t(v_t-\alpha_t S_{t-1}k_t)
\end{equation}
$$

得到

$$
\begin{equation}
S_t=
\alpha_t S_{t-1}
+
\Delta v_t k_t^\top
\end{equation}
$$

delta rule形式

* 展开一个chunk

把chunk大小设为$`C`$ ,沿着token len 分成多个chunk，在每个chunk内 token就是这样

$$
\begin{equation}
tC,tC+1,\dots,tC+C-1
\end{equation}
$$

定义chunk 开始时的 memory。

$$
\begin{equation}
S_{[t]}=S_{tC}
\end{equation}
$$

**第一个 token**

$$
\begin{equation}
S_{tC+1}=
\alpha_1 S_{[t]} + \Delta v_1 k_1^\top
\end{equation}
$$

**第二个 token**

$$
\begin{equation}
S_{tC+2}=
\alpha_2 S_{tC+1} + \Delta v_2 k_2^\top
\end{equation}
$$

代入：

$$
\begin{equation}
S_{tC+2}=\alpha_2\alpha_1 S_{[t]}
+
\alpha_2\Delta v_1k_1^\top
+
\Delta v_2k_2^\top
\end{equation}
$$

推广到chunk 末尾

$$
\begin{equation}
S_{tC+C}=
\gamma^C S_{[t]}
+
\sum_{i=1}^{C}
\gamma^{C-i}\Delta v_i k_i^\top
\end{equation}
$$

右边第一项

$$
\begin{equation}
\overrightarrow{S_{[t]}}
=\gamma^C_{[t]} S_{[t]}
\end{equation}
$$

其中

$$\begin{equation}
\gamma^j_{[t]} = \prod_{j=tC+1}^{tC+j} \alpha_j\end{equation}$$

再看右边第二项，把它写成矩阵形式

定义

$$\begin{equation}
\overrightarrow{K_{[t]}}=
\text{diag}\left(\frac{\gamma^C_{[t]} }{\gamma^i_{[t]} }\right)K_{[t]}
\end{equation}
$$

其中

$$\begin{equation}
K_{[t]} = 
\begin{bmatrix}
k_1 \\
k_2 \\
\vdots \\
k_C
\end{bmatrix}
\end{equation}
$$
$$\begin{equation}
\Delta V_{[t]} = 
\begin{bmatrix}
\Delta v_1 \\
\Delta v_2 \\
\vdots \\
\Delta v_C
\end{bmatrix}
\end{equation}
$$

于是

$$\begin{equation}
S_{[t+1]}=
\overrightarrow{S_{[t]}}
+
\Delta V_{[t]}^\top
\overrightarrow{K_{[t]}}
\end{equation}
$$

这已经和最终公式$`\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]} } + \left( \widetilde{\mathbf{U}_{[t]} } - \overleftarrow{\mathbf{W}_{[t]} } \mathbf{S}_{[t]}^\top \right)^\top \overrightarrow{\mathbf{K}_{[t]} } `$ 非常接近。就差$`\Delta V_{[t]}`$继续推导。

之前的式子

$$
\begin{equation}
\Delta v_t=\beta_t(v_t-\alpha_t S_{t-1}k_t)
\end{equation}
$$

且

$$
\begin{equation}
S_{tC+i}=\gamma^iS_{[t]}+\sum_{j<=i}\frac{\gamma^i}{\gamma^j}{\Delta{v_j}k^T_j}
\end{equation}
$$

其中$`S_{[t]}`$是上一个chunk结束时的memory,$`S_{t-1}`$代表 token recurrence,（32式），所以$`S_{t-1}`$,所以



$$
\begin{equation}
S_{i-1} = \left( \prod_{r=1}^{i-1} \alpha_r \right) S_{[t]} + \sum_{j < i} \left( \prod_{r=j+1}^{i-1} \alpha_r \right) \Delta v_j k_j^\top
\end{equation}
$$

chunk内


$$
\begin{equation}
S_{i-1} = \gamma^{i-1} S_{[t]} + \sum_{j < i} \frac{\gamma^{i-1}}{\gamma^j} \Delta v_j k_j^\top
\end{equation}
$$

代入$`S_{t-1}k_t`$：

所以

$$
\Delta v_i = \beta_i \left( v_i - \gamma^{i-1} S_{[t]} k_i - \sum_{j < i} \frac{\gamma^{i-1}}{\gamma^j} (k_j^\top k_i) \Delta v_j \right)
$$

写成矩阵形式

把i=1..C写成矩阵 (chunk内)

$$
K\in\mathbb{R}^{C\times{d_k}}
$$

$$
V\in\mathbb{R}^{C\times{d_v}}
$$

$$
{\Delta}V\in\mathbb{R}^{C\times{d_v}}
$$

把之前的
$`
\begin{equation}
\Delta v_i
=\beta_i
\left(
v_i-
\gamma^{i-1}S_{[t]}k_i-
\sum_{j<i}{\frac{\gamma^{i-1}}{\gamma^j}}(k_j^\top k_i){\Delta v_j}
\right)
\end{equation}
`$
式写成矩阵

```math
\Delta v_i + \beta_i \sum_{j < i} \frac{\gamma^{i-1}}{\gamma^j} (k_j^\top k_i) \Delta v_j = \beta_i v_i - \beta_i \gamma^{i-1} S_{[t]} k_i
```

由于求和项$` \beta_i \sum_{j < i} \frac{\gamma^{i-1}}{\gamma^j}  (k_j^\top k_i) `$只在$`j<i `$存在

```math
\mathbf{L}_{ij} = 
\begin{cases}
\beta_i \frac{\gamma^{i-1} }{\gamma_{j}}  (k_j^\top k_i) , & j < i \\
0, & j >= i
\end{cases}
```

```math
L = \text{strictLower} (\text{diag}(B) (\Gamma \odot K K^\top) )
```

所以

```math
(1+L){\Delta}V=\beta{V}-BS^\top_{[t]}
```

其中

```math
L=\text{strictLower}(\text{diag}(\beta)(\Gamma\odot KK^\top))
```

且
```math
B_i = \beta_i \gamma^{i-1} k_i
```

我们定义

$$
\begin{equation}
\widetilde U=
(I+L)^{-1}\text{diag}(\beta)V
\end{equation}
$$

这就正好得到：

$$
\begin{equation}
\widetilde U_{[t]}=
\left[
I+
\text{strictLower}
(\text{diag}(\beta)(\Gamma\odot KK^\top))
\right]^{-1}
\text{diag}(\beta)V
\end{equation}
$$

现在再加$`\Delta {v}`$式子的$`-\gamma^{i-1}S_{[t]}k_i`$

$$
\begin{equation}
WS^T_{[t]}
\end{equation}
$$

其中论文的表达如下

$$
\begin{equation}
W_{[t]}=
\left[
I+
\text{strictLower}
(\text{diag}(\beta)( KK^\top))
\right]^{-1}
\text{diag}(\beta)K
\end{equation}
$$

这里严格的推导是$` W = (1+L)^{-1} \text{diag}(\beta \gamma^{i-1}) K`$ 上面论文的表达似乎是 rescaling 得到的。

最终的memory更新

$$
\begin{equation}
\Delta V_{[t]}
= \widetilde U_{[t]}-
\overleftarrow W_{[t]}S_{[t]}^\top
\end{equation}
$$

代回

$$
\begin{equation}
S_{[t+1]}=
\overrightarrow{S_{[t]}}
+
(\widetilde U_{[t]}-\overleftarrow W_{[t]}S_{[t]}^\top)^\top
\overrightarrow K_{[t]}
\end{equation}
$$

这就最终的memory更新公式。

**attention 输出**

token 输出

$$
\begin{equation}
o_i=S_i q_i
\end{equation}
$$

把

$$
\begin{equation}
S_{tC+i}=\gamma^iS_{[t]}+\sum_{j<=i}\frac{\gamma^i}{\gamma^j}{\Delta{v_j}k^T_j}
\end{equation}
$$

代入：

$$
\begin{equation}
o_i=q_i^\top\gamma^{i}S_{[t]}
+
\sum_{j\le i}(q_i^\top k_j){\frac{\gamma^i}{\gamma^j}}\Delta v_j
\end{equation}
$$

上式第一项

$$\gamma^{i}{q_i^\top}S_{[t]}$$
$$\begin{equation}
\text{diag}{(\gamma^{i})}Q_{[t]}S^T_{[t]}
\end{equation}
$$

定义

$$
\begin{equation}
\overleftarrow{Q}_{[t]}  = \text{diag}(\gamma^i_{[t]}) Q_{[t]} 
\end{equation}
$$

所以第一项为

$$
\begin{equation}
\overleftarrow{Q}_{[t]}S^T_{[t]}
\end{equation}
$$

59式的第二项

$$\sum_{j\le i}(q_i^\top k_j){\frac{\gamma^i}{\gamma^j}}{\Delta}v_j$$

先算QK^T
然后乘衰减矩阵

$$
\begin{equation}
\mathbf{\Gamma}_{[t]} = 
\begin{cases}
\frac{\gamma^r_{[t]} }{\gamma^i_{[t]} }, & i \leq r \\
0, & i > r
\end{cases} \in \mathbb{R}^{C \times C}
\end{equation}
$$

于是

$$
QK^T\odot \Gamma
$$

于是第二项为 

$$({QK^T \odot {\Gamma})\Delta V}$$

得到attention

$$
\begin{equation}
O_{[t]}=\overleftarrow Q_{[t]}S_{[t]}^\top
+
(Q_{[t]}K_{[t]}^\top\odot\Gamma)\Delta V_{[t]}
\end{equation}
$$

再代入55式 
也就是$`\Delta V_{[t]}
=\widetilde U_{[t]}-\overleftarrow W_{[t]}S_{[t]}^\top
`$

得到

$$
\begin{equation}
O_{[t]}=
\overleftarrow Q_{[t]}S_{[t]}^\top
+
(QK^\top\odot\Gamma)
(\widetilde U_{[t]}-\overleftarrow W_{[t]}S_{[t]}^\top)
\end{equation}
$$

### ChunkWise GDN的torch实现

```python
def chunk_gated_delta_rule_ref_nvlab(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor = None,
):
    """
    Pure PyTorch chunk reference from NVlabs GatedDeltaNet (same math as their chunk.py).
    Inputs: (B, T, H, D) in FLA convention. g in log space. Returns o (B, T, H, V).
    https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py
    """
    BT = chunk_size
    # BT(chunk_size)=64
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    # after transpose (B,H,T,D):
    # q.shape=torch.Size([1, 8, 8000, 128]), 
    # k.shape=torch.Size([1, 8, 8000, 128]), 
    # v.shape=torch.Size([1, 8, 8000, 128]), 
    # beta.shape=torch.Size([1, 8, 8000])
    # g.shape=torch.Size([1, 8, 8000])
    T_orig = q.shape[-2] #  T_orig=8000,
    pad_len = (BT - (T_orig % BT)) % BT # pad_len =0
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
        print(f"[chunk_gated_delta_rule_ref_nvlab] after pad: q.shape={q.shape}")

    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    # b=1, h=8, l=8000, d_k=128, d_v=128, l//BT=125
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % BT == 0, f"seq length {l} must be multiple of chunk_size {BT}"

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    """
    mask.shape=(64, 64)
    torch.triu是创建上三角
    如果BT = 4
    [[ True,  True,  True,  True],  # 这一行代表时刻 0
    [False,  True,  True,  True],  # 这一行代表时刻 1
    [False, False,  True,  True],  # 这一行代表时刻 2
    [False, False, False,  True]]  # 这一行代表时刻 3
    这个矩阵后续 在mask_fill那个地方 做Strictly Lower Triangular
    """
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    # after rearrange(c=BT=64): q.shape=torch.Size([1, 8, 125, 64, 128]), decay.shape=torch.Size([1, 8, 125, 64])
    # .cumsum(-1)是在对decay最后一个维度做前缀和
    """
    此时 decay 的形状是 [1, 8, 125, 64]。
    decay.unsqueeze(-1): 形状变为 [1, 8, 125, 64, 1]（变成列向量视图）。
    decay.unsqueeze(-2): 形状变为 [1, 8, 125, 1, 64]（变成行向量视图）。相减 (-): 触发广播机制。PyTorch 会创建一个 64x64 的矩阵，其中位置 (i, j) 的值正好是 decay[i] - decay[j]。.exp(): 得到最终的衰减矩阵。结果：L_mask[i, j] 存储的就是从时刻 $j$ 到时刻 $i$ 的累积衰减系数。
    """
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).exp()
    # L_mask.shape=torch.Size([1, 8, 125, 64, 64])
    # 上三角包括对角线 全部填充为0
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, BT):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(BT, dtype=torch.float, device=q.device)
    # attn.shape=torch.Size([1, 8, 125, 64, 64]) (last two dims are 64,64)
    k_cumsum = attn @ v
    attn = -((k_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)
    for i in range(1, BT):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(BT, dtype=torch.float, device=q.device)
    k_cumdecay = attn @ k_beta
    u = v = k_cumsum # 这个u后面没用到不管 把v更新了
    # k_cumsum/k_cumdecay shape: torch.Size([1, 8, 125, 64, 128]), v.shape=torch.Size([1, 8, 125, 64, 128])

    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.to(torch.float32)
        # initial_state set: S.shape=torch.Size([1, 8, 128, 128])
    else:
        print(f"[chunk_gated_delta_rule_ref_nvlab] S.shape={S.shape} (zeros)")
    o = torch.zeros_like(v)
    mask_o = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    num_chunks = l // BT
    # mum_chunks=l//BT=125, mask_o.shape=(64,64)
    for i in range(0, num_chunks):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask_o, 0)
        v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn_i @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        # if i == 0 or i == num_chunks - 1:
        #     print(f"[chunk_gated_delta_rule_ref_nvlab] chunk i={i}: q_i.shape={q_i.shape}, attn_i.shape={attn_i.shape}, v_new.shape={v_new.shape}, S.shape={S.shape}")
            #  chunk i=0: q_i.shape=torch.Size([1, 8, 64, 128]), attn_i.shape=torch.Size([1, 8, 64, 64]), v_new.shape=torch.Size([1, 8, 64, 128]), S.shape=torch.Size([1, 8, 128, 128])

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T_orig]
    o = o.transpose(1, 2).contiguous()
    # return o.shape=torch.Size([1, 8000, 8, 128])
    return o
```

# Ref

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [【论文解读】Gated Delta Network](https://wkq9411.github.io/2026-01-18/Paper-Gated-Delta-Network.html)
- [Question on chunkwise Gated DeltaNet notation: should M include decay (Gamma)?](https://github.com/NVlabs/GatedDeltaNet/issues/15)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)

