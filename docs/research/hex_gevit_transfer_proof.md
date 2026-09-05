# Hex 三向坐标：GE 迁移的构造、证明与限制

日期：2026-09-05。性质：研究设计与独立推导，**不是现有 SHARE-ViT 的整体等变认证，也不是已经训练过的新模型**。

## 0. 结论先行

有漂亮的迁移方法。最简路线不是将 x/y 三角函数逐项改写，而是：

1. 用受约束的三向整数坐标表示 Hex **中心格点**；
2. 用坐标的带符号循环置换表示 60° 旋转；
3. 用群相对坐标构造 attention 的位置特征；
4. 将等变证明化成一次换元或一句群乘法恒等式。

这条路线首先保证的是 $\Lambda\rtimes C_6$（p6 型）离散等变；加入反射可扩到 $\Lambda\rtimes D_6$。这里 D6 有 12 个元素，但**不是每 30° 旋转一次的 C12**。

Hex 卷积与 p6/p6m 并非新概念，见 [HexaConv，ICLR 2018](https://ehoogeboom.github.io/publication/hexaconv/)；群 attention 的背景见 [Romero 与 Cordonnier，ICLR 2021](https://arxiv.org/abs/2010.00977)。下面具体的符号、证明展开和项目迁移建议由本分析重新组织，不宣称这些基本恒等式本身是论文贡献。

## 1. 三个坐标，但只有两个自由度

取

$
\Lambda=\{(q,r,s)\in\mathbb Z^3:q+r+s=0\}.
$

这表示三角格点集，即六边形 Voronoi 单元的中心；不是度为 3 的蜂巢顶点图。定义平面嵌入

$
\iota(q,r,s)=\left(\frac{\sqrt3}{2}q,\frac{r-s}{2}\right).
\tag{H1}
$

六个最近邻差向量是 $(1,-1,0)$ 的全部排列。在此标度下距离均为 1。

$
\|\iota(c)\|^2=\tfrac12(q^2+r^2+s^2),\qquad
d_{hex}(c,0)=\max(|q|,|r|,|s|).
\tag{H2}
$

若只存 $(q,r)$，则 $s=-q-r$，对应

$
\iota(q,r)=\left(\tfrac{\sqrt3}{2}q,\tfrac12q+r\right),\quad
G=\begin{pmatrix}1&1/2\\1/2&1\end{pmatrix}.
$

物理内积是 $c^TGd$，不是裸坐标的 $q_cq_d+r_cr_d$。第三坐标用于对称表达，不应作为额外空间维度计入模型。

## 2. 60° 旋转只需置换和取负

定义

$
R(q,r,s)=(-r,-s,-q),\qquad S(q,r,s)=(q,s,r).
\tag{H3}
$

直接代入 H1 得

$
\iota(Rc)=R_{\pi/3}\iota(c),\quad
\iota(Sc)=\operatorname{diag}(1,-1)\iota(c).
$

而且

$
R^2(q,r,s)=(s,q,r),\ R^3c=-c,\ R^6=I,
\quad S^2=I,\quad SRS=R^{-1}.
\tag{H4}
$

所以 C6 的变换是精确索引置换，不需要在 Hex 格上插值。轴向两坐标中

$
R=\begin{pmatrix}0&-1\\1&1\end{pmatrix},
\qquad R^TGR=G.
\tag{H5}
$

**这比每次构造浮点 sin/cos 旋转矩阵更干净；但收益主要在几何索引与实现常数，不会自动减少 attention 的全部计算量。**

## 3. 格点平移与旋转组成半直积

用 $g=(t,a)$，$t\in\Lambda,a\in\mathbb Z_6$ 表示

$
g\cdot p=t+R^ap.
$

群律与逆元为

$
(t,a)(u,b)=(t+R^au,a+b),\qquad
(t,a)^{-1}=(-R^{-a}t,-a).
\tag{H6}
$

朝向加减均模 6。规则表示特征保留位置与朝向槽：

$
F:\Lambda\times\mathbb Z_6\to\mathbb R^C,
\quad (L_{(t,a)}F)(p,h)=F(R^{-a}(p-t),h-a).
\tag{H7}
$

同一通道在群槽间只做置换，这与“192 个数任意经过普通 ViT”是不同的表示约束。

## 4. 路线 A：保留 GE 原文修正结构

与[原文分析文档](ge_vit_original_proof_analysis.md)的 A1 对照，将群元素写成 C6 指数：

$
K_h(p,u;q,v)=\psi(R^{-h}(q-p),\,[2u-v-h]_6).
\tag{H8}
$

这就是 $h^{-1}uv^{-1}u$ 在循环群中的加法形式，不是 $v-u$。对整体变换 $(t,a)$，令 $p=t+R^ap'$、$q=t+R^aq'$、$u=u'+a$、$v=v'+a$：

$
R^{-h}(q-p)=R^{-(h-a)}(q'-p'),
$

$
2(u'+a)-(v'+a)-h=2u'-v'-(h-a).
$

故

$
K_h(g\cdot(p',u');g\cdot(q',v'))=K_{h-a}(p',u';q',v').
\tag{H9}
$

在共享 Q/K/V、协变邻域和完整群求和下，应用另一文档的 A3–A4 换元即可得到
$T[L_gF]=L_gT[F]$。三向坐标并不改变证明，只把空间旋转替成整数置换。

若扩展 D6，请恢复 $h^{-1}uv^{-1}u$ 的乘法，**不可继续使用 $2u-v-h$**：含反射的群不交换。

## 5. 路线 B：更短的群相对坐标证明（优先建议）

不逐式复刻 GE 的外部 h 与 u 求和，而使用 query 自身朝向作为局部参考系。令 $A=(p,u), B=(q,v)$ 为两个群 token：

$
\xi(A,B)=A^{-1}B=(R^{-u}(q-p),[v-u]_6).
\tag{H10}
$

最关键的一行是

$
\xi(gA,gB)=(gA)^{-1}(gB)=A^{-1}B=\xi(A,B).
\tag{H11}
$

### 命题：基于 H10 的局部 attention 等变

假设邻域满足 $\mathcal N(gA)=g\mathcal N(A)$，Q/K/V 与位置映射 ψ 在所有群 token 共享。定义

$
\ell_F(A,B)=\langle QF(A),K(F(B)+\psi(\xi(A,B)))\rangle,
$

$
\alpha_F(A,B)=\frac{\exp\ell_F(A,B)}
{\sum_{C\in\mathcal N(A)}\exp\ell_F(A,C)},\qquad
(TF)(A)=\sum_{B\in\mathcal N(A)}\alpha_F(A,B)VF(B).
\tag{H12}
$

由 $(L_gF)(gA)=F(A)$ 与 H11，得到
$\ell_{L_gF}(gA,gB)=\ell_F(A,B)$。邻域双射保证 softmax 分母相同；对求和换元 $B'=gB$：

$
(TL_gF)(gA)=\sum_{B\in\mathcal N(A)}\alpha_F(A,B)VF(B)
=(TF)(A)=(L_gTF)(gA).
\tag{H13}
$

证毕。多头、共享输出映射、残差、逐群槽共享的通道 MLP 同样与群槽置换交换。

也可把位置特征做成标量 bias $b(A^{-1}B)$ 加到 QK 上，证明不变。ψ 可以读取整组三坐标；**无需为 q/r/s 三个分量强行共享同一 MLP**，因为输入已经是左作用不变的局部描述。若直接对世界坐标分别做不共享映射，则没有此保证。

路线 B 的证明对非交换群也成立。它是本分析建议的群相对 attention，不应写成“GE 原文式 (18) 的同一公式”。当前 `model/gevit_tiny.py` 的 query-frame offset 与方向差，与 B 路线更接近。

## 6. 从图像提升到群特征，也要有证明

先在原生 Hex 标量图 $f:\Lambda\to\mathbb R^{C_{in}}$ 上定义

$
(\mathcal L f)(p,h)=\sum_{\delta\in\mathcal D}
W(R^{-h}\delta)f(p+\delta),
\tag{H14}
$

其中 $\mathcal D$ 是 R 不变的有限邻域，W 是共享模板。对 $f_g(p)=f(R^{-a}(p-t))$，令 $\delta=R^a\delta'$：

$
\mathcal L f_g(p,h)
=\sum_{\delta'\in\mathcal D}W(R^{-(h-a)}\delta')
 f(R^{-a}(p-t)+\delta')
=\mathcal L f(R^{-a}(p-t),h-a).
\tag{H15}
$

这给出输入层与后续群 attention 的同一种作用，而不是只证明中间 attention。

但从普通 RGB 方格图采样到 Hex 图的算子 P 还需要满足
$P\mathcal R_a=L_aP$。60° 通常不是方格像素置换，所以对一般离散图像这个关系只有近似；必须把“Hex 内部精确”等变与“RGB 端到端近似”等变分别报告。

## 7. 与当前 SHARE 组件的对应

### 7.1 半圆六姿态不是完整 C6 群轨道

当前 half6 的角为 $0°,30°,60°,90°,120°,150°$。加 60° 后会出现 180°、210°，不在原集合内。除非额外规定并满足对跖约束，不能把它们简单 modulo 180° 当成等价响应。

- 最干净的 C6 实验用 $0°,60°,120°,180°,240°,300°$。
- 若保留 30° 密度，可用全 12 朝向。对 60° 旋转它们按槽位 +2 置换，分成偶数/奇数两个轨道；这仍只证明 C6，不证明格点上严格 C12。
- 无向轴满足 $\theta\sim\theta+\pi$ 时，自然的矩编码是 $(\cos2\theta,\sin2\theta)$。当前一阶 cos/sin 在跨半圆边界时变号，不能既保留一阶向量，又忽略反向响应的符号与数值约束。

### 7.2 null-softmax 可以保留

若真实方向 logits 满足 $z_d(L_gf,gp)=z_{d-a}(f,p)$，null logit 在 g 下不变，则完整方向集合上的 null-softmax 满足相同置换。null 是单独的固定槽，不需复制六份。

对 full C6 概率，定义

$
z_m(p)=\sum_{d=0}^5p_d(p)e^{im d\pi/3}.
$

换元得到 $z_m(L_gf,gp)=e^{im a\pi/3}z_m(f,p)$。因此 cos/sin 圆周矩本身可以严格等变，前提是**完整轨道与正确的输入响应变换**。

### 7.3 Look bias 也有独立的充分条件

对每个探针 m 的完整方向概率与模板，设

$
B_f(p,q)=\frac1M\sum_{m,s,d}p_{m,s,d}(f,p)
T_m\big(R^{-d}(q-p)/s\big).
\tag{H16}
$

若概率方向协变、尺度 s 在旋转下不变、模板旋转/采样与格点作用一致，则换元 $d=d'+a$ 给出
$B_{L_gf}(gp,gq)=B_f(p,q)$。任意 M 的平均、Image/Feature 两支相加、共享 G 层探针，都不破坏该恒等式，条件是各支的输入特征本身已经按声明的表示变换。

**仅有协变 bias 不足以证明整个 attention：QK 和 V 的表示也必须兼容。**

### 7.4 普通 ViT 仍是结构边界

现有 paired token 被任意线性层、逐坐标非线性及普通绝对 PE 处理后，不再必然按二维旋转变换。对多个一阶二维向量的线性映射，C6 的一种兼容形式是
$W=A\otimes I+B\otimes J$，其中 $J=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$；一般无约束 W 不满足交换关系。若要求同类型向量的反射等变，B 项还需受限（标准向量到向量时为零）。

可选路线是完整群槽 + 共享逐槽 MLP，或显式 irrep/向量网络；不能仅替换位置坐标后沿用任意 192 维映射并宣称严格证明。

### 7.5 三向特征表示是另一个选择，不等于三向位置坐标

二维向量可冗余嵌入三维零和面；在该表示中，60° 仍是带符号置换。令 $\Pi=I-\mathbf1\mathbf1^T/3$，则 $\Pi\tanh(z)$ 与这种作用交换，因为 tanh 是奇函数且 Π 与 R/S 交换。逐坐标 GELU/ReLU 不满足符号交换；tanh 后不投影也未必保持零和约束。

这给出一种便于证明的非线性备选，但三坐标只承载两个自由度，存储增加而非减少，且仅保留特定表示类型；不是已经验证准确率更好的方案。

## 8. 邻域、边界与尺度：严格性的前提

$
\mathcal D_k=\{(q,r,s):q+r+s=0,\max(|q|,|r|,|s|)\le k\}
$

是 C6/D6 不变邻域；恰在第 k 圈有 $6k$ 个点，含中心总计 $1+3k(k+1)$。因此两圈共 19 点（去中心 18），无需人为将外圈 12 点当成“每 30° 的等角圆采样”：点数与欧氏角度均匀性不是同一件事。

- 无限格或与群作用兼容的周期域可证明格点平移；有限中心六边形只天然保留绕中心旋转/反射，不保留任意平移。
- 方形裁剪掩码通常不是 60° 不变域；mask 也必须变换，或只比较共同有效内部区域。
- 在 $k\Lambda$ 上下采样可保留该子格的 C6 作用，但平移子群缩小；采样原点与池化邻域必须一致。
- 多个固定半径分支在旋转证明中可以并存。放缩通常不是有限格与尺度集合的双射，因此多尺度不等于尺度等变。
- 普通随机 dropout 在单次不同掩码下不保证点态等变；证明可针对 eval，或训练时使用配对变换的掩码。

## 9. 建议的最小实验，不先重写整套模型

1. **纯代数检查**：R6=I、反射关系、欧氏度量、群逆元、位置描述协变。
2. **原生 Hex 合成图**：先证明 lifting + 一个局部 attention 的 60° 等变，禁用绝对 PE 和非协变边界。
3. **RGB→Hex 单独测误差**：把采样误差从 backbone 的误差中拆出来。
4. **固定计算预算**：显式群轴若每槽仍 192 维，激活会扩大约 6 倍；若总预算固定 192，每槽是 32，不能直接沿用“每槽 3 个等宽 head”，需要重新分配。
5. **和当前 SHARE 并行比较**：严格 C6 分支、现有 half6d3r、以及相近预算的 GE-inspired p4；准确率和等变误差分别报告。

建议先实现 **Hex lifting + H10/H12 群相对局部 attention**，不要一开始就证明整套现有双 Look。之后可用 H16 增加严格兼容的 Look。这是机制迁移，不是简单把 p4 的常数 4 换成 6。

## 10. 数值核对与可复现性

配套脚本：`verify_hex_gevit.py`。它在小型周期轴向格上检查坐标恒等式、GE 修正结构、群相对 attention 与完整 C6 圆周矩，并展示半圆槽集合不闭合。周期域只用于隔离边界、验证代数；不代表真实 ImageNet 图像也有周期边界。

解析证明承担一般结论；有限数值测试只用于发现符号、逆元、索引与转向错误，不能代替证明，也不验证当前训练模型端到端等变。

本次实际运行（NumPy float64，周期格 3×3，所有 9 个平移与 6 个旋转）：

| 检查 | 结果 |
| --- | --- |
| cube/axial 嵌入、度量、C6/D6 群恒等式、环点数 | 通过 |
| 路线 A，保留 GE 修正位置结构 | 最大绝对误差 4.441×10⁻¹⁶ |
| 路线 B，群相对位置 attention | 最大绝对误差 2.498×10⁻¹⁶ |
| 完整 C6 的 null-softmax、圆周矩变换 | 通过 |
| 当前半圆六角集合对 +60° 的闭合性 | 不闭合（预期反例） |

这两条 attention 数值检查使用全 key 集合来隔离群代数；局部协变邻域的成立由前文条件证明，真实局部实现仍需单独测试 mask。
