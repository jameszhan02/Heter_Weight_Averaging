# Union 维度扩展的数学直觉与逐步推导

> 这是一份本地中文学习笔记，已被 `.gitignore` 忽略，不属于正式仓库文档。
>
> 目标：从矩阵 shape 的基本含义开始，理解论文 2607.18026 中 Union-style expansion 为什么需要关注 attention head、GQA、SwiGLU、归一化和残差连接。

## 0. 我们究竟在解决什么问题？

假设有两个同系列但规模不同的 Transformer：

- small model 的参数记作 $\theta_s$；
- large model 的参数记作 $\theta_t$；
- small model 的 hidden size、MLP width 和层数都可能更小。

因为两个模型的参数 shape 不同，不能直接计算：

$$
(1-\lambda)\theta_s+\lambda\theta_t.
$$

论文先定义一个扩展操作 $\mathcal E$，把 small model 放进 large model 的参数空间：

$$
\widetilde\theta_s=\mathcal E(\theta_s).
$$

然后才进行插值：

$$
\theta^{\cup}(\lambda)
=(1-\lambda)\widetilde\theta_s+\lambda\theta_t.
$$

这里有两个不同层次的问题：

1. **Shape 问题**：如何把一个较小 tensor 变成目标 shape？
2. **Function 问题**：扩展后，模型对输入的计算是否仍尽量接近原 small model？

简单 zero padding 主要解决第一个问题。论文中的 head mapping、GQA mapping、SwiGLU mapping、normalization correction 和 residual identity 是在努力解决第二个问题。

---

## 1. 先读懂线性层的 shape

### 1.1 向量和矩阵

设输入向量是：

$$
x\in\mathbb R^{d_{\mathrm{in}}}.
$$

线性层输出是：

$$
y=Wx,
$$

其中：

$$
W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\qquad
y\in\mathbb R^{d_{\mathrm{out}}}.
$$

因此，对 PyTorch 的 `Linear.weight`：

- rows 对应输出坐标；
- columns 对应输入坐标；
- shape 是 `[out_features, in_features]`。

这是后面判断每个 axis 语义的基础。

### 1.2 一个很小的数值例子

设：

$$
W=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
\qquad
x=
\begin{bmatrix}
5\\
6
\end{bmatrix}.
$$

那么：

$$
y=Wx=
\begin{bmatrix}
1\times5+2\times6\\
3\times5+4\times6
\end{bmatrix}
=
\begin{bmatrix}
17\\
39
\end{bmatrix}.
$$

如果 input dimension 扩大，应该增加 columns；如果 output dimension 扩大，应该增加 rows。这就是为什么不能只看“这是一个二维 tensor”，还要知道两个 axis 各自代表什么。

---

## 2. Slot mapping：把 source 坐标放到 target 的哪些位置？

论文使用确定性的 slot map：

$$
\pi_{n\rightarrow m}(i)
=
\left\lfloor\frac{im}{n}\right\rfloor,
\qquad i=0,\ldots,n-1,
$$

其中：

- $n$ 是 source slot 数量；
- $m$ 是 target slot 数量；
- $m\ge n$；
- $i$ 是 source slot index；
- $\pi(i)$ 是对应的 target slot index。

### 2.1 例子：4 个 slot 扩到 8 个

$$
\begin{aligned}
\pi(0)&=\lfloor0\times8/4\rfloor=0,\\
\pi(1)&=\lfloor1\times8/4\rfloor=2,\\
\pi(2)&=\lfloor2\times8/4\rfloor=4,\\
\pi(3)&=\lfloor3\times8/4\rfloor=6.
\end{aligned}
$$

所以：

```text
source: S0 S1 S2 S3
target: S0  0 S1  0 S2  0 S3  0
index:   0  1  2  3  4  5  6  7
```

### 2.2 例子：3 个 slot 扩到 5 个

$$
\pi(0)=0,\qquad
\pi(1)=1,\qquad
\pi(2)=3.
$$

所以 source 不一定会被等间隔放置，但映射是确定的，并且在 $m\ge n$ 时不会把两个 source slots 放进同一个 target slot。

### 2.3 二维权重的扩展

设：

$$
W\in\mathbb R^{a_s\times b_s}.
$$

目标 shape 是：

$$
a_t\times b_t.
$$

论文的通用规则是：

$$
\mathcal E(W)_{
\pi_{a_s\rightarrow a_t}(i),
\pi_{b_s\rightarrow b_t}(j)
}
=cW_{ij},
$$

其他位置为零。

普通线性权重通常取 $c=1$。这相当于：

1. 先创建一个 target-shape 的零矩阵；
2. 确定 source 每一行应该进入哪一个 target row；
3. 确定 source 每一列应该进入哪一个 target column；
4. 把原值放到行列映射的交点。

但 Transformer 中并不是所有坐标都应该逐个映射。attention head 需要作为一个完整 block 映射。

---

## 3. Attention 为什么要以完整 head 为单位？

### 3.1 一个 attention head 做什么？

对 hidden state $x$，先计算：

$$
q=W_qx,\qquad
k=W_kx,\qquad
v=W_vx.
$$

一个 head 的 attention 核心是：

$$
\operatorname{Attention}(q,k,v)
=
\operatorname{softmax}
\left(
\frac{qk^\top}{\sqrt{d_h}}
\right)v,
$$

其中 $d_h$ 是 head dimension。

假设：

$$
h=H d_h,
$$

其中 $H$ 是 query head 数量。投影结果会从一个长度为 $Hd_h$ 的向量 reshape 成 $H$ 个长度为 $d_h$ 的 head。

例如 $H=4,d_h=128$：

```text
coordinates   0:128   -> head 0
coordinates 128:256   -> head 1
coordinates 256:384   -> head 2
coordinates 384:512   -> head 3
```

### 3.2 为什么不能逐坐标打散？

一个 head 内的所有 $d_h$ 个坐标共同形成一个 dot product：

$$
qk^\top=\sum_{r=1}^{d_h}q_rk_r.
$$

如果把同一个 source head 的一半坐标放到 target head 0，另一半放到 target head 3，那么 target 模型 reshape 之后，这两半不会再参与同一个 head 的 dot product。原计算结构因此被破坏。

此外，RoPE 通常还会对 head 内坐标按固定方式配对旋转。随意拆散坐标可能进一步破坏这些配对。

所以，当 head dimension 固定时，论文把一个完整 head 当作一个 slot：

```text
small heads: Q0 Q1 Q2 Q3
large slots: Q0  0 Q1  0 Q2  0 Q3  0
```

每个 `Q0` 实际代表连续的 $d_h$ 个坐标，而不是一个标量。

---

## 4. GQA 为什么需要分别映射 Q heads 和 KV heads？

### 4.1 MHA 与 GQA 的区别

普通 Multi-Head Attention 常见的是：

$$
H=K,
$$

即 query heads 数量等于 key-value heads 数量。

Grouped-Query Attention 使用：

$$
H>K.
$$

多个 query heads 会共享一个 KV head。

例如：

```text
Query heads: Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
KV heads:    K0          K1

Q0-Q3 share K0
Q4-Q7 share K1
```

每个 KV head 服务的 query head 数量是：

$$
g=H/K.
$$

### 4.2 四个 attention 权重的 axis 语义

按照 PyTorch `[out_features, in_features]`：

$$
W_q\in\mathbb R^{(Hd_h)\times h},
$$

$$
W_k,W_v\in\mathbb R^{(Kd_h)\times h},
$$

$$
W_o\in\mathbb R^{h\times(Hd_h)}.
$$

对应关系：

| Weight | Row axis | Column axis |
|---|---|---|
| $W_q$ | query-head output | hidden input |
| $W_k$ | KV-head output | hidden input |
| $W_v$ | KV-head output | hidden input |
| $W_o$ | hidden output | concatenated query-head input |

因此：

- $W_q$ 的 rows 按 query heads 映射；
- $W_k,W_v$ 的 rows 按 KV heads 映射；
- 它们的 columns 按 hidden coordinates 映射；
- $W_o$ 的 columns 按 query heads 映射；
- $W_o$ 的 rows 按 hidden coordinates 映射。

### 4.3 一个保持 GQA 分组的例子

Small model：

$$
H_s=4,\qquad K_s=2,\qquad H_s/K_s=2.
$$

Target model：

$$
H_t=8,\qquad K_t=4,\qquad H_t/K_t=2.
$$

分别映射：

```text
small Q: Q0 Q1 Q2 Q3
target Q: Q0  0 Q1  0 Q2  0 Q3  0

small K/V: K0 K1
target K/V: K0  0 K1  0
```

因为 source 和 target 的 group ratio 都是 2，query 与 KV 的分组关系可以自然对应。

如果两个模型的 head dimension、GQA ratio 或模块语义无法对应，普通 padding 并不能自动解决问题。这是方法的支持边界，而不是一个 shape trick 能掩盖的问题。

---

## 5. SwiGLU 的三个矩阵为什么要按不同方向扩展？

### 5.1 SwiGLU 的计算顺序

设 hidden size 是 $h$，intermediate size 是 $m$。SwiGLU MLP 可以写成：

$$
g=W_{\mathrm{gate}}x,
$$

$$
u=W_{\mathrm{up}}x,
$$

$$
z=\operatorname{SiLU}(g)\odot u,
$$

$$
y=W_{\mathrm{down}}z.
$$

其中 $\odot$ 是逐元素乘法。

各个 shape 是：

$$
x\in\mathbb R^h,
$$

$$
W_{\mathrm{gate}},W_{\mathrm{up}}
\in\mathbb R^{m\times h},
$$

$$
g,u,z\in\mathbb R^m,
$$

$$
W_{\mathrm{down}}
\in\mathbb R^{h\times m},
$$

$$
y\in\mathbb R^h.
$$

所以数据流是：

```text
hidden h
   |
gate/up projections
   |
intermediate m
   |
element-wise SwiGLU
   |
down projection
   |
hidden h
```

### 5.2 每个 axis 的语义

| Weight | Shape | Rows | Columns |
|---|---:|---|---|
| `gate_proj` | `[m, h]` | intermediate output | hidden input |
| `up_proj` | `[m, h]` | intermediate output | hidden input |
| `down_proj` | `[h, m]` | hidden output | intermediate input |

因此扩展时：

- `gate_proj` 和 `up_proj`：rows 用 intermediate slot map，columns 用 hidden slot map；
- `down_proj`：rows 用 hidden slot map，columns 用 intermediate slot map。

### 5.3 为什么这样做可以保留已有 MLP 子网络？

用 $P_h$ 表示“把 small hidden vector 放入 target hidden slots”的映射，用 $P_m$ 表示“把 small intermediate vector放入 target intermediate slots”的映射。

扩展后的输入是：

$$
x'=P_hx.
$$

理想化地写，扩展后的 gate 矩阵为：

$$
W'_{\mathrm{gate}}
=P_mW_{\mathrm{gate}}P_h^\top.
$$

于是：

$$
W'_{\mathrm{gate}}x'
=P_mW_{\mathrm{gate}}P_h^\top P_hx.
$$

因为 slot mapping 只是把 source 坐标放入互不冲突的 target slots，所以：

$$
P_h^\top P_h=I.
$$

因此：

$$
W'_{\mathrm{gate}}x'
=P_mW_{\mathrm{gate}}x.
$$

也就是说，source gate 的结果只是被放到了 target intermediate slots，数值本身没有改变。`up_proj` 同理。

SiLU 和逐元素乘法都只作用于每个坐标，因此已映射的 source slots 仍可执行原来的门控计算，未使用的 slots 保持为零。

对 down projection：

$$
W'_{\mathrm{down}}
=P_hW_{\mathrm{down}}P_m^\top.
$$

将映射后的 intermediate vector 投影回来：

$$
W'_{\mathrm{down}}P_mz
=P_hW_{\mathrm{down}}P_m^\top P_mz
=P_hW_{\mathrm{down}}z.
$$

所以原 small MLP 的输出被放回 target hidden slots。

这一推导说明：

> gate/up/down 的 axis 必须按照它们在数据流中的语义成对映射，才能让原来的 MLP 子网络闭合。

如果只把三个矩阵都复制到“左上角”，某些特殊 shape 下可能碰巧工作，但那不是从模块语义推导出来的通用规则。

---

## 6. 新 attention/MLP 坐标为什么置零？

扩展产生了 small model 原本没有的参数位置。如果这些位置使用随机初始化，新路径会立即产生未经训练的输出。

抽象地说，扩展后的分支可以分成：

$$
G_{\mathrm{expanded}}(x)
=G_{\mathrm{copied}}(x)+G_{\mathrm{new}}(x).
$$

我们希望扩展刚完成时：

$$
G_{\mathrm{new}}(x)=0.
$$

于是：

$$
G_{\mathrm{expanded}}(x)
=G_{\mathrm{copied}}(x).
$$

对 SwiGLU，新增坐标可能经过非线性和乘法：

$$
\operatorname{SiLU}(g_{\mathrm{new}})
\odot u_{\mathrm{new}}.
$$

随机值可能被门控结构放大或改变方向。将新增 gate/up/down 坐标置零，可以让新增 nonlinear branches 在 expansion 后不贡献输出。

对 attention，新 Q/K/V 坐标如果随机初始化，可能产生高方差 logits：

$$
q_{\mathrm{new}}k_{\mathrm{new}}^\top/\sqrt{d_h}.
$$

把新坐标置零，可以避免尚未训练的 attention pattern 立即干扰原模型。

注意：这些位置在后续 interpolation 中会获得 large checkpoint 的值，因此不会永远是零。

---

## 7. RMSNorm 宽度变化为什么可能需要缩放补偿？

这是最容易感觉“突然开始数学化”的部分，可以分步看。

### 7.1 Small model 的 RMS

忽略很小的 $\epsilon$，small hidden vector 的 RMS 是：

$$
\operatorname{RMS}_s(x)
=
\sqrt{
\frac{1}{h_s}
\sum_{i=1}^{h_s}x_i^2
}.
$$

RMSNorm 的核心输出近似为：

$$
y_i
=
\gamma_i
\frac{x_i}{\operatorname{RMS}_s(x)}.
$$

### 7.2 Zero padding 后，分母改变了

现在把 $x$ 放进 $h_t$ 维空间，其余坐标为零。平方和没有改变：

$$
\sum_{i=1}^{h_t}(x'_i)^2
=
\sum_{i=1}^{h_s}x_i^2.
$$

但平均时的除数从 $h_s$ 变成了 $h_t$：

$$
\operatorname{RMS}_t(x')
=
\sqrt{
\frac{1}{h_t}
\sum_{i=1}^{h_s}x_i^2
}.
$$

将它与原 RMS 比较：

$$
\operatorname{RMS}_t(x')
=
\sqrt{\frac{h_s}{h_t}}
\operatorname{RMS}_s(x).
$$

因为 $h_t>h_s$，所以 target RMS 更小。

### 7.3 归一化后的有效坐标会被放大

对已有的非零坐标：

$$
\frac{x_i}{\operatorname{RMS}_t(x')}
=
\sqrt{\frac{h_t}{h_s}}
\frac{x_i}{\operatorname{RMS}_s(x)}.
$$

也就是说，只做 zero padding 会让有效坐标在 RMSNorm 后被额外放大：

$$
\sqrt{h_t/h_s}.
$$

为了抵消它，可以把复制过来的 normalization scale 乘以：

$$
\sqrt{h_s/h_t}.
$$

两者相乘：

$$
\sqrt{h_s/h_t}
\sqrt{h_t/h_s}
=1.
$$

这就是论文提到的宽度变化 normalization correction 的来源。

### 7.4 为什么说“可能”使用这个修正？

上面的推导忽略了 $\epsilon$，并假设：

- source 有效坐标只是被嵌入；
- 新坐标为零；
- 使用 RMSNorm；
- target 的归一化定义与 source 一致。

LayerNorm 还会减均值。加入大量零坐标后，均值也会改变，因此不能直接把 RMSNorm 的推导原封不动套过去。

所以真正实现时必须先确认模型使用 RMSNorm 还是 LayerNorm，不能只根据参数名猜测。

---

## 8. 新 normalization scale 为什么设为 1？

归一化通常包含一个可学习 scale $\gamma$：

$$
\operatorname{Norm}(x)
=
\gamma\odot\widehat x.
$$

如果设置：

$$
\gamma=1,
$$

则：

$$
\gamma\odot\widehat x=\widehat x.
$$

它没有额外缩放 normalized signal，所以 1 是乘法意义上的 neutral value。

如果设置：

$$
\gamma=0,
$$

则整个 normalized signal 被压成零。0 对乘法不是 identity，而是 annihilator。

因此要记住：

```text
additive neutral value:       0
multiplicative neutral value: 1
```

线性分支希望新增输出为零，所以新增 branch weights 使用 0。Normalization scale 是乘法因子，希望不额外改变信号，所以使用 1。

---

## 9. 额外 Transformer block 为什么是 identity？

### 9.1 残差连接是关键

一个简化的 pre-norm residual block 是：

$$
h_{\ell+1}
=
h_\ell
+G_\ell\big(N_\ell(h_\ell);W_\ell\big).
$$

其中：

- $h_\ell$ 是输入；
- $N_\ell$ 是 normalization；
- $G_\ell$ 是 attention 或 MLP branch；
- $+h_\ell$ 是 residual path。

如果新层的 branch weights 让：

$$
G_\ell\big(N_\ell(h_\ell);W_\ell^{\mathrm{new}}\big)=0,
$$

那么：

$$
h_{\ell+1}=h_\ell+0=h_\ell.
$$

这就是 identity function。

### 9.2 Identity block 不等于 identity matrix

一个常见误解是：

> 为了让额外层成为 identity，应该把 attention/MLP 权重设成单位矩阵。

这里不对。Transformer block 已经有一条直接传递输入的 residual path：

```text
                 +----------------------+
                 |                      |
input h -> Norm -> branch -> zero ------+--> h + 0 = h
                 |                      |
                 +---- residual h ------+
```

因此只需要让 residual branch 输出零，不需要 branch 自己计算 identity。

如果 branch 自己输出 $h$，残差相加反而可能得到：

$$
h+h=2h,
$$

这不是 identity。

### 9.3 一个真实 pre-norm block 有两个 residual sublayers

通常可以写成：

$$
u
=x+\operatorname{Attention}(N_1(x)),
$$

$$
y
=u+\operatorname{MLP}(N_2(u)).
$$

如果 attention branch 和 MLP branch 都输出零：

$$
u=x+0=x,
$$

$$
y=u+0=x.
$$

所以整个新增 block 才是 identity。

只关闭其中一个 branch 不够。例如 attention 为零但 MLP 非零时：

$$
y=x+\operatorname{MLP}(N_2(x)),
$$

它仍然会改变输入。

### 9.4 Norm scale 设为 1 与 branch 设为 0 是否矛盾？

不矛盾。

当 branch 完全输出零时，norm 的输出不会通过 branch 影响 residual stream。但把 norm scale 设为 1 可以保持它处于健康的 neutral 状态。

在与 large checkpoint 插值后，branch weights 会逐渐非零。此时 normalization 也需要提供正常信号。如果一开始把 norm scale 设为零，插值后的新增层可能同时受到异常的 norm 缩放和 branch 缩放。

---

## 10. Interpolation 后为什么新增路径会逐渐激活？

Union 合并是：

$$
\theta^{\cup}(\lambda)
=(1-\lambda)\mathcal E(\theta_s)
+\lambda\theta_t.
$$

考虑一个在 expanded small 中为零、在 large model 中为 $W_t$ 的新增 branch weight：

$$
W_{\mathrm{merged}}
=(1-\lambda)\cdot0+\lambda W_t
=\lambda W_t.
$$

因此：

- $\lambda=0$：新增 branch 关闭；
- 很小的 $\lambda$：新增 branch 只受到较小的 large-model 参数注入；
- $\lambda=1$：恢复 large model 的该参数。

但是要特别注意：

> 参数随 $\lambda$ 线性变化，不代表模型输出也随 $\lambda$ 线性变化。

Transformer 包含：

- QK 点积；
- softmax；
- SiLU；
- element-wise multiplication；
- normalization；
- 多层组合。

这些运算都是非线性的。因此 near-balanced interpolation 可能发生严重干扰，这也是论文为什么只在端点附近使用较小比例。

---

## 11. “保持函数”到底是严格保证还是近似目标？

需要分情况。

### 可以严格说明的局部结构

- slot placement 本身是确定性的；
- 新参数位置为零；
- residual branch 输出严格为零时，新增 pre-norm block 严格是 identity；
- 理想化线性层和 SwiGLU 子空间在一致映射下可以保留已有计算。

### 可能只近似成立的整体行为

- normalization 会受到 hidden width 变化影响；
- attention reshape、RoPE 和 GQA 分组必须真正兼容；
- embedding、LM head 和 tied weights 必须遵循 target architecture；
- 浮点 dtype 和数值误差可能造成差异；
- 不同模型即便同系列，也不一定有完全对齐的内部表示；
- interpolation 后是多个非线性模块共同变化，不存在一般性的功能保持保证。

因此论文使用的是“deterministic expansion largely preserves the source function”，而不是声称对任意模型都能数学上完全等价。

---

## 12. 用一条数据流把所有规则串起来

```text
small hidden vector x
        |
        | P_h: map complete hidden/head blocks into target slots
        v
expanded hidden vector x'
        |
        +--> Q rows: query-head block mapping
        |
        +--> K/V rows: KV-head block mapping
        |
        +--> O columns: corresponding query-head mapping
        |
        +--> gate/up rows: intermediate mapping
        |
        +--> down columns: matching intermediate mapping
        |
        +--> unused parameters: zero
        |
        +--> new norm scales: one
        |
        +--> additional blocks: residual input + zero branch
        v
expanded small-model computation in target parameter space
```

---

## 13. 学习检查题

不需要一次全部回答。能用自己的话解释，就说明已经开始真正理解。

### 基础题

1. PyTorch `Linear.weight` 为什么是 `[output, input]`？
2. 当 input dimension 增大时，为什么增加的是 columns？
3. 当 output dimension 增大时，为什么增加的是 rows？

### Attention 题

4. 为什么不能把一个 attention head 的坐标拆到多个 target heads？
5. 为什么 `q_proj.weight` 的 rows 表示 query heads？
6. 为什么 `o_proj.weight` 的 columns 才表示 query-head 输入？
7. GQA 中为什么 Q heads 和 KV heads 必须分别映射？

### MLP 题

8. `gate_proj` 的 `[m,h]` 中，两个 axis 各代表什么？
9. `down_proj` 为什么是 `[h,m]`？
10. 如果 `down_proj` 的 intermediate axis 与 `gate_proj` 映射不一致，会发生什么？

### Normalization 题

11. zero padding 为什么会改变 RMSNorm 的分母？
12. 为什么乘以 $\sqrt{h_s/h_t}$ 可以抵消有效坐标的放大？
13. 为什么 normalization scale 的 neutral value 是 1 而不是 0？

### Residual 题

14. 为什么 branch 输出为零时，residual block 是 identity？
15. 为什么把 branch 设置成 identity matrix 反而可能得到 $2h$？
16. 为什么 attention 和 MLP 两个 residual branches 都必须处理？

---

## 14. 当前最值得先掌握的三件事

第一次学习不需要立即记住所有公式。先确保可以解释：

1. **Axis semantics**：矩阵的 rows 和 columns 在模块中分别代表什么。
2. **Residual identity**：identity 来自 `input + zero branch`，不是来自单位矩阵。
3. **Normalization width effect**：数值虽然被原样复制，但归一化所使用的维度数量改变了，所以函数仍可能改变。

掌握这三点后，再去看真实 Qwen checkpoint 中 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj` 和 `down_proj` 的 shape，会比直接读公式容易很多。

## 参考位置

- Paper Section 3.2: Union-Style Merging.
- Paper Appendix B.1: Union-Style Expand.
- Paper equations 13-19 and pseudocode 1.
- Paper URL: <https://arxiv.org/abs/2607.18026>
