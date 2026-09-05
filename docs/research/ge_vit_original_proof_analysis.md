# GE-ViT：关键公式摘录与证明审查

日期：2026-09-05。对象：Xu 等，*E(2)-Equivariant Vision Transformer*，UAI 2023。

## 1. 原文定位与摘录边界

- [正式出版页](https://proceedings.mlr.press/v216/xu23b.html)
- [正文 PDF](https://proceedings.mlr.press/v216/xu23b/xu23b.pdf)：§4.1–4.3，§5.1–5.3，印刷页 2359–2361。
- [附录 PDF](https://proceedings.mlr.press/v216/xu23b/xu23b-supp.pdf)：A 的错误定位、B 的证明，尤其 PDF 第 2–4 页。

这里只摘录必要的数学式并保留编号；不整段复制原文证明。后续推导是重新组织的数学验证和本项目分析，不冒充作者原句，也不把分析意见算作原文定理。已对照正文第 6 页、附录第 3 页的渲染图核对乘法顺序。

### 关键数学式

原文式 (13)：

$
\Phi[\rho_1(g)f]=\rho_2(g)[\Phi[f]].
$

原文式 (15)，在该文所分析的 GSA 位置编码约定下：

$
\rho((i,\tilde h),(j,\hat h))
=\rho^P(x(j)-x(i),\tilde h^{-1}\hat h).
$

原文式 (18)，GE 的替换项：

$
\rho((i,\tilde h),(j,\hat h))
=\rho^P(x(j)-x(i),\tilde h\hat h^{-1}\tilde h).
$

正文式 (18) 后定义的作用：

$
\mathcal L_h[\rho]((i,\tilde h),(j,\hat h))
=\rho^P\!\left(h^{-1}(x(j)-x(i)),
h^{-1}(\tilde h\hat h^{-1}\tilde h)\right).
$

注意：外部输出朝向 $h$、被求和的输入朝向 $\tilde h$、key 朝向 $\hat h$ 不是同一个变量；原文还用相似字母标识 attention head。下面改记为 $h,u,v$，省略 head，以避免混淆。

## 2. 自行重推：修正项为什么有效

设空间位置为 $p,q$，整体变换为 $g=(t,a)$。定义特征作用：

$
(L_gF)(p,u)=F(a^{-1}(p-t),a^{-1}u).
$

把用于第 $h$ 个输出朝向的位置特征写成

$
K_h(p,u;q,v)=\psi(h^{-1}(q-p),h^{-1}uv^{-1}u).
\tag{A1}
$

做换元 $p=t+ap',q=t+aq',u=au',v=av'$。两个分量分别化简为：

$
h^{-1}(q-p)=h^{-1}a(q'-p')=(a^{-1}h)^{-1}(q'-p'),
$

$
(au')(av')^{-1}(au')
=au'v'^{-1}a^{-1}au'=a(u'v'^{-1}u').
$

因此

$
K_h(t+ap',au';t+aq',av')=K_{a^{-1}h}(p',u';q',v').
\tag{A2}
$

这一步只用了逆元、结合律和线性群作用；**没有使用横轴与纵轴正交，也没有使用旋转群交换律**。

反过来，$(au')^{-1}(av')=u'^{-1}v'$，不会多出前面的 $a$。若同时保留原文那种“再在第二分量左乘 $h^{-1}$”的作用，就不能与第一分量一起换成 $(a^{-1}h)^{-1}$。这说明的是**特定位置编码与特定作用的配对问题**，不是“一切使用相对方向 $u^{-1}v$ 的网络都不等变”。

## 3. 从位置特征到完整 attention 的独立证明

设 Q、K、V 是在所有位置和群槽共享的通道映射。定义

$
\ell_h[F](p,u;q,v)
=\langle QF(p,u),K(F(q,v)+K_h(p,u;q,v))\rangle,
$

$
\alpha_h[F](p,u;q,v)
=\frac{e^{\ell_h[F](p,u;q,v)}}
{\sum_{(z,w)\in\mathcal N(p,u)}e^{\ell_h[F](p,u;z,w)}}.
$

与正文式 (16)–(17) 的求和结构对应，写单头输出为

$
T[F](p,h)=\sum_{u\in H}\sum_{(q,v)\in\mathcal N(p,u)}
\alpha_h[F](p,u;q,v)VF(q,v).
\tag{A3}
$

这里省略共享输出线性层；softmax 是在固定 $(p,u,h)$ 的 key 邻域上归一化，**不是把外层的 u 求和也混入 softmax**。

要求邻域满足 $\mathcal N(g\cdot(p,u))=g\cdot\mathcal N(p,u)$。式 (A2) 保证换元前后每个 logit 相同；邻域是双射，分母也相同，于是 alpha 相同。将 key 求和与 u 求和一起重编号：

$
T[L_gF](p,h)=T[F](a^{-1}(p-t),a^{-1}h)=L_gT[F](p,h).
\tag{A4}
$

逐头都成立且输出映射共享时，多头拼接也成立。

## 4. 证明需要写出的边界条件（本分析）

1. **邻域协变与测度不变是两件事。** 附录 B 后半部分的重编号不能仅靠 unimodular 一词保证。固定不随旋转变换的任意邻域、截断窗口或 mask 都需要另外核对。
2. **有限像素格不是连续平面。** 对固定 square grid，90° 可以成为索引置换；一般角度需要插值。有限矩形边界、采样起点和 stride 也会限制严格成立的变换。
3. **参数共享必须符合表示类型。** 若方向是一个可置换的群轴，逐槽共享的 MLP/LN 可以与置换交换；若把方向压成连续二维向量，则普通逐通道 GELU/LN/任意线性层不自动满足旋转交换关系。
4. **等变与分类不变不同。** 对空间和群轴作对称池化才得到相应不变读出；单独选固定朝向不行。
5. **多尺度并不自动意味着缩放等变。** 固定若干尺度标签可以在旋转中保持不变；要证明缩放还要处理采样格、尺度集合的闭合和面积权重。

## 5. 当前本地 GE 对比实现与原文不是逐式同一模型

检查对象：`model/gevit_tiny.py`，本次同步代码 `6847f53`。

- `C4LiftingPatchEmbed` 用四次 `rot90` 的共享卷积核，显式保留四个朝向槽。
- `GEViTLocalAttention` 用 query 朝向的逆旋转处理空间 offset，方向项使用 `(key_orientation-query_orientation) % 4`。
- 因而位置描述接近
  $(R_{-u}(q-p),v-u)$，而不是逐字实现式 (A1) 的外部 h 与 $2u-v-h$ 结构。
- 文件开头已经声明这是 DeiT 尺寸的 adaptation。应继续将其称为 **GE-ViT-inspired / p4 local adaptation**，不能把原文式 (18) 的完整证明未经核对地套到当前实现。
- 当前实现自身可沿“群相对坐标”的路线另证，见另一份文档；还需检查窗口、分层下采样和 readout 的边界行为。

这不等于否定其已测性能，也不等于断言其不等变；只是划清“原论文公式”“项目改编算法”和“尚未完成的整体等变验证”。

## 6. 建议的论文措辞

可以说：本项目研究将共享几何探测与局部群结构引入 ViT；对满足格点和表示约束的离散子模块可给出等变证明。

暂不能说：现有 SHARE-ViT 已严格 E(2)-equivariant；或者将直角坐标改成 Hex 三坐标后整个 backbone 自动严格等变。

配套：[Hex 三向坐标与迁移证明](hex_gevit_transfer_proof.md)。
