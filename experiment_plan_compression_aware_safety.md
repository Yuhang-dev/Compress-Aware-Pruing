# Experiment Plan — Compression-Aware Safety Entanglement

> **执行者注意 (READ FIRST):** 本文件是交给编码代理 (Codex) 执行的完整实验计划。请**严格按 Phase 顺序执行**,每个 Phase 末尾有 **GATE(验证门)**:到达 GATE 后**停下,产出指定结果文件,等待人工确认**再继续。不要跳阶段。不要为了"做完"而牺牲正确性。遇到不确定的接口/数据集路径,先验证再用,不要凭记忆硬编码 HuggingFace repo id。

---

## 0. 项目目标与核心假设

**目标:** 提出一种**训练阶段**方法,使对齐 LLM 的**安全机制能在事后被标准压缩器(Wanda / SparseGPT / 量化)压缩后依然保留**,且在 **iso-utility(同稀疏度、同困惑度/MMLU)**下安全保持率显著高于基线。

**威胁模型(重要):** **良性压缩** —— 部署者为效率而压缩,**不是**对手主动攻击。因此本项目**不做**对抗鲁棒性/抗篡改评测,只衡量"压缩后安全是否还在"。

**核心假设(Phase 2 要验证的 go/no-go):** 标准剪枝器只看良性校准数据的重要性($\text{Wanda}=|W|\cdot\|X\|$;$\text{SparseGPT}\propto w^2/[H^{-1}]$),安全权重因在良性数据上激活范数 $\|X\|\approx0$ 而被优先剪除。若在微调时把"写出拒绝方向"的能量**搬进剪枝器必然保留的高重要性/高奇异子空间**,标准剪枝就会顺带保住安全。

**贡献定位(决定取舍,务必遵守):** 头牌贡献是**概念**——*Compression-Aware Safety Entanglement*(把安全绑进压缩器必保子空间)。主引擎是 Method B(剪枝在环训练),Method A 是其机理分析与轻量变体。**Method B 的"剪枝感知训练"范式本身不是 novelty**(PAT, arXiv 2408.14721 已有),所以**实现上必须守住与 PAT 的边界**:(i) 不学可训练掩码;(ii) 用部署者现成剪枝器的判据算掩码;(iii) 产出**仍稠密、对任意标准压缩鲁棒**的模型,而非一个特定剪枝模型;(iv) 目标是安全而非效用。这些边界直接影响代码设计。

---

## 1. 环境与依赖

```bash
# Python 3.10+, CUDA GPU (单卡 ≥24GB 可跑 7B bf16 推理与 LoRA 训练;full-FT 建议 ≥40GB 或开梯度检查点)
pip install "torch>=2.2" transformers accelerate datasets peft
pip install vllm                 # 评测期快速生成
pip install lm-eval              # MMLU/HellaSwag/GSM8K 标准评测 (EleutherAI lm-evaluation-harness)
pip install bitsandbytes         # NF4 量化 baseline
pip install auto-gptq autoawq    # GPTQ / AWQ 量化 (二选一即可,先用其中一个)
pip install wandb pandas matplotlib pyyaml
```

剪枝实现:**优先复用开源实现**,不要自己重写 SparseGPT。
- Wanda + SparseGPT:克隆 `locuslab/wanda`(含两者),或 `IST-DASLab/sparsegpt`。
- Magnitude:平凡,自己实现(按 $|W|$ 全局/逐层取阈)。
- 若复用有困难,Wanda 可自实现(逐输出行按 $|W_{ij}|\cdot\|X_j\|$ 取 top-$(1-s)$,约 50 行);SparseGPT 必须复用。

**第一步先验证依赖与 GPU,产出 `env_report.txt`(torch/cuda 版本、GPU 显存、各包版本),再继续。**

---

## 2. 模型与数据集

### 2.1 模型(HF;若路径变动,先搜索确认现行 repo id)
- **主模型:** `meta-llama/Llama-2-7b-chat-hf`(refusal-direction / brittleness 文献的标准载体)。
- **泛化模型(Phase 5):** `meta-llama/Llama-3.1-8B-Instruct`。
- **安全判别模型(judge):** `meta-llama/Llama-Guard-3-8B`。

### 2.2 数据集
**有害提示(ASR↓ 评测 + 训练用拒绝监督):**
- AdvBench、HarmBench、StrongREJECT、JailbreakBench、HEx-PHI(HEx-PHI 需申请;拿不到则跳过,不阻塞)。
- **必须切分 train/eval 不相交**:训练用的有害提示与评测有害提示**严格分开**(防泄漏)。建议:AdvBench 划一部分 + 额外有害指令做训练监督;HarmBench/StrongREJECT/JailbreakBench 全部留作评测(held-out)。

**过度拒绝(误拒↓ 评测):** XSTest、OR-Bench(用 toxic/hard split)。

**通用能力(iso-utility):** WikiText-2(困惑度)、MMLU、GSM8K、HellaSwag、(可选)MT-Bench 胜率。

**剪枝校准集:** C4 或 WikiText,采样 **128 条、序列长 2048**(Wanda/SparseGPT 标准设置)。**这是良性数据,部署者用它剪枝,不要混入有害数据**(混入有害数据本身是一个 baseline,见 §9)。

**拒绝监督数据构造(训练用):** 把训练划分的有害提示配上规范拒绝回复 $y^{\text{ref}}$。可用模板化拒绝或现成安全 SFT 集(如 BeaverTails / safe-RLHF 的拒绝样本)。记录构造脚本到 `data/build_refusal_sft.py`。

**产出 `data/manifest.json`:列出每个数据集的最终 HF 路径、split、样本数、用途(train/eval-harm/eval-refusal/eval-utility/calib),供人工核对切分无泄漏。**

---

## 3. 仓库结构

```
casafety/
  configs/                  # 全部实验用 YAML 配置(数据/模型/方法/超参)
    base.yaml
    method_a2.yaml
    method_b.yaml
    ...
  data/
    manifest.json
    build_refusal_sft.py
  src/
    vpref.py                # 拒绝方向抽取 + 校验
    pruners.py             # Wanda / SparseGPT / magnitude 掩码 + 量化封装
    masked_linear.py       # MaskedLinearSTE 包装 (Method B 用)
    losses.py              # L_refuse / L_utility / L_entangle(A1,A2,C)
    methods/
      method_a.py
      method_b.py
      method_c.py
      method_d.py
    train.py               # 统一训练入口(读 config)
    eval_safety.py         # ASR(关键词 + LlamaGuard3)
    eval_refusal.py        # XSTest / OR-Bench 误拒
    eval_utility.py        # PPL / MMLU / GSM8K / HellaSwag(调 lm-eval)
    compress_and_eval.py   # 给一个稠密模型,跨配置压缩后跑全套评测,输出一行行结果
  results/                  # 所有结果表(CSV)+ 图 + 检查点指针
  scripts/                  # 一键脚本(每 Phase 一个)
  README.md
```

设计原则:**配置驱动**(方法/超参/压缩配置都从 YAML 读),**评测与训练解耦**(`compress_and_eval.py` 对任意稠密 checkpoint 都能跑),**所有数值结果落 CSV**(便于汇总成主表)。

---

## 4. 共用组件规格(先实现,A/B/C/D 都依赖)

记号:$\hat r_\ell$=第 $\ell$ 层残差流单位拒绝方向;写出矩阵 $W_{\text{out}}\in\{\texttt{o\_proj},\texttt{down\_proj}\}$;$\|X_j\|$=第 $j$ 输入特征在校准集上的 $L_2$ 激活范数;$\pi_\theta,\pi_{\text{ref}}$=当前/冻结原始模型。

### 4.1 拒绝方向抽取 `vpref.py`(difference-of-means,Arditi et al.)
$$
r_\ell=\frac{1}{|\mathcal D_{\text{harm}}|}\!\sum_{x\in\mathcal D_{\text{harm}}}\!a^{(\ell)}(x)-\frac{1}{|\mathcal D_{\text{benign}}|}\!\sum_{x\in\mathcal D_{\text{benign}}}\!a^{(\ell)}(x),\quad \hat r_\ell=r_\ell/\|r_\ell\|.
$$
- 在**末 token**位置取残差激活;有害/无害各采 ~256 条(用 train 划分,不用 eval)。
- **校验(必须做):** 逐层做激活干预——在某层把 $\hat r_\ell$ **加到**无害提示残差上应**诱发拒绝**,**从**有害提示残差中**减去**应**抑制拒绝**。选诱发/抑制效果最强的中后层作为目标层 $\ell^\*$。产出 `results/vpref_validation.csv`(每层的诱发率/抑制率)。
- 输出:`artifacts/vpref/{model}_layer{ell}.pt`。

### 4.2 `L_refuse`(保拒绝)`losses.py`
主形式(NLL,稳健):
$$
\mathcal L_{\text{refuse}}=-\mathbb E_{x\sim\mathcal D_{\text{harm}}}\sum_t\log\pi_\theta(y^{\text{ref}}_t\mid x,y^{\text{ref}}_{<t}).
$$
变体(方向投影,做消融):有害输入在 $\ell^\*$ 层残差对 $\hat r$ 的投影更大、无害更小(带 margin)。

### 4.3 `L_utility`(防漂移,iso-utility 保险)`losses.py`
$$
\mathcal L_{\text{utility}}=\mathbb E_{x\sim\mathcal D_{\text{benign}}}\,D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$
良性数据用 C4/通用指令样本。冻结 $\pi_{\text{ref}}$ 副本算 teacher logits。

### 4.4 剪枝器 `pruners.py`
统一接口:`compute_mask(layer_weight, calib_act_stats, sparsity, method) -> binary_mask`。
- **Wanda:** 逐输出行,保留 top-$(1-s)$ 的 $|W_{ij}|\cdot\|X_j\|$。
- **SparseGPT:** 复用开源实现(含补偿更新 $\delta=-\frac{w_q}{[H^{-1}]_{qq}}H^{-1}_{:,q}$)。
- **Magnitude:** 逐层保留 top-$(1-s)$ 的 $|W_{ij}|$。
- **2:4 半结构化:** Wanda/SparseGPT 都支持,作为一种 sparsity 配置。
- **量化封装:** NF4(bitsandbytes,最简)+ GPTQ 或 AWQ(其一)。统一接口 `quantize(model, method) -> model`。
- 校准激活统计 `calib_act_stats`(各层 $\|X_j\|$、SparseGPT 的 $H$)在固定良性校准集上计算,缓存到 `artifacts/calib/`。

---

## 5. 方法规格

### 5.1 Method A —— 子空间对齐(轻量,Phase 2 主角)
设 $a=W_{\text{out}}^\top\hat r\in\mathbb R^{m}$($a_j$=输入特征 $j$ 对写拒绝方向的贡献)。

**A1(激活加权能量集中):** $\|X_j\|$ 取 stop-grad。
$$
\mathcal L^{\text{A1}}_{\text{ent}}=-\frac{\sum_j\|X_j\|\,a_j^2}{\sum_j a_j^2}.
$$
**A2(顶奇异子空间投影,默认主用):** 构造 $\tilde W=W_{\text{out}}\,\mathrm{diag}(\|X\|)$,SVD 取 top-$k$ 右奇异子空间 $V_k$,$P_{\text{keep}}=V_kV_k^\top$;$\tilde a=\mathrm{diag}(\|X\|)\,a$:
$$
\mathcal L^{\text{A2}}_{\text{ent}}=\frac{\|(I-P_{\text{keep}})\,\tilde a\|^2}{\|\tilde a\|^2}.
$$
- 对所有目标层的 $W_{\text{out}}$ 求和。$k$ 作超参(如保留能量 90% 对应的秩)。
- $\|X\|$、$P_{\text{keep}}$ 随权重漂移,**每 N 步重算一次**(EM 式)。
- 总损失 $\mathcal L=\mathcal L_{\text{refuse}}+\lambda_u\mathcal L_{\text{utility}}+\lambda_e\mathcal L^{\text{A2}}_{\text{ent}}$。

### 5.2 Method B —— 剪枝在环训练(★ 主算法,Phase 3)
**STE 实现(关键,Codex 注意):** 用直通估计让被剪权重也拿梯度。PyTorch 技巧:
```python
# W_pruned 前向=掩码后,反向=直通(等价于 M=I)
W_pruned = W + (W * M - W).detach()
```
用 `MaskedLinearSTE` 包装目标 `nn.Linear`(替换其 weight 计算为上式后做 `F.linear`)。掩码 `M` 用 `pruners.compute_mask(...)` 算,**`M` detached**,不学。

**训练循环(伪代码):**
```
precompute calib_act_stats on fixed benign calib set   # 部署者视角的校准
every N steps: recompute calib_act_stats (weights drift)
for step in range(T):
    cfg = sample(COMPRESS_CONFIGS)        # 见下,每步随机一个压缩配置
    if cfg is pruning:
        M = compute_mask(W, calib_act_stats, cfg.sparsity, cfg.pruner)
        set MaskedLinearSTE masks = M     # 前向用 W⊙M, 反向直通到稠密 W
    elif cfg is quant:
        apply fake-quant to target layers # 模拟量化,反向直通
    L = L_refuse(harmful_batch) + λ_u * L_utility(benign_batch)
    L.backward(); optimizer.step()        # 梯度落到稠密 W
return dense pruning-robust W             # 注意:产物是稠密模型,不是剪枝模型
```
**`COMPRESS_CONFIGS` 采样空间(B 的差异化来源):** 稀疏度 $s\in\{0.25,0.5,0.6,\text{2:4}\}$ × 剪枝器 $\in\{\text{Wanda},\text{SparseGPT}\}$,外加一档 fake-quant(NF4)。**每步随机采样一个**,使鲁棒性跨配置泛化。$s=0$(不压缩)也以一定概率采到,保住未压缩时的安全/能力。
- Wanda 掩码便宜(一次前向算 $\|X\|$),可每步或每少数步重算;SparseGPT 重,每 N 步重算并缓存。
- **防作弊:** 模型可能靠压低 utility 换安全 → $\lambda_u\mathcal L_{\text{utility}}$ + iso-utility 评测双重把关。

### 5.3 Method C —— 幅值/重要性再分配(便宜 baseline,Phase 4)
选择步:$I(W)=\mathbb E_{x\sim\mathcal D_{\text{harm}}}|W\odot\nabla_W\mathcal L_{\text{refuse}}(x)|$,取 top-$q\%$ 为安全关键集 $\mathcal S$。
强化步(hinge,把 $\mathcal S$ 的 Wanda 得分顶过本层第 $s$ 分位阈 $\tau_\ell$):
$$
\mathcal L^{\text{C}}_{\text{ent}}=\sum_{(i,j)\in\mathcal S}\big[\tau_\ell-|W_{ij}|\cdot\|X_j\|\big]_+^2\;+\;(\text{KL 锁良性行为}).
$$
预期:对 Wanda 有效,对 SparseGPT 因补偿更新而偏弱(报告此对比)。

### 5.4 Method D —— 结构焊死(硬约束版,Phase 5/可选)
将 $W_{\text{out}}$ 重参数化为低秩主干 + 残差,硬约束 $\hat r\in\mathrm{span}(U_k)$(top-$k$ 左奇异子空间):
$$
W_{\text{out}}=U_k\Sigma_k V_k^\top+W_{\text{resid}},\qquad \text{约束}\ \|(I-U_kU_k^\top)\hat r\|^2\to0.
$$
实现:惩罚项或周期性把 $\hat r$ 正交插入 $U_k$ 基。**预期能力税最重**,务必在 matched-$k$ 下量化。优先级低,时间不够可只做小规模或留作未来工作。

---

## 6. 评测协议(三轴,所有方法共用)

`compress_and_eval.py`:输入一个稠密 checkpoint,对 `COMPRESS_CONFIGS` 中每个配置压缩后,跑下面三轴,每个配置输出 CSV 一行。

**轴1 有害性 ASR↓:** AdvBench / HarmBench / StrongREJECT / JailbreakBench(+HEx-PHI 若有)。每条有害提示生成 ≤512 token,**两套判定**:(a) 拒绝关键词匹配 ASR;(b) **Llama-Guard-3-8B 判定** harmful 比例(主指标)。用 vLLM 生成。
**轴2 误拒↓:** XSTest(250 安全请求)、OR-Bench;报告 false-refusal rate。**这一轴必报**,否则安全数字被"靠变笨/滥拒"解释掉。
**轴3 通用能力(iso-utility 卡尺):** WikiText-2 困惑度、MMLU、GSM8K、HellaSwag(走 `lm-eval`),可选 MT-Bench。

**iso-utility 读法:** 同一压缩配置下,先确认各方法的 PPL/MMLU 落在同一区间(±0.5% / ±可接受 PPL),**再**比 ASR 与误拒。主表每行=一个压缩配置,列分三轴,对比各方法。

**统计:** 训练与评测各 **3 个随机种子**,报均值±标准差。

---

## 7. 基线(必须全部跑,用于守住 novelty/solidity)

1. **Dense-Aligned(上界):** 原对齐模型不压缩 —— 安全上界、能力上界。
2. **Pruned-Aligned(问题基线):** 原对齐模型直接被各配置压缩 —— 预期安全崩、能力基本保。**这是要打败的对象。**
3. **Harmful-Calibration:** 把有害数据混进剪枝校准集后再剪 —— 便宜的"安全感知剪枝"对照(审稿人必问)。
4. **Post-hoc Safety-SFT:** 先压缩再补一轮安全 SFT —— 另一便宜对照,对比"逐配置重对齐"的成本。
5. **PAT(motivating baseline,重要):** 跑 `kriskrisliu/PAT_Pruning-Aware-Tuning`,证明它保住效用却**丢失安全**——把与 PAT 的范式重叠转化为论证支点。
6. **(可选)HSR / 安全神经元恢复:** prune-time/restoration 类 SOTA 对照。

---

## 8. 分阶段执行计划(按序,带 GATE)

### Phase 0 — 环境 & 复现问题
1. 装环境,产出 `env_report.txt`。
2. 拉模型与全部数据集,产出 `data/manifest.json`,**人工核对 train/eval 切分无泄漏**。
3. 实现 `pruners.py` + 三轴评测脚本。
4. 跑 **Pruned-Aligned 基线**:主模型在 {Wanda,SparseGPT}×{50%} 下压缩,跑全套评测。
- **GATE-0:** 评测流水线端到端跑通;**复现"剪枝破坏安全"**(ASR 在 50% 稀疏下明显上升,而 PPL/MMLU 大致保住)。产出 `results/phase0_problem.csv` + 一张 ASR-vs-sparsity 图。**停,等人工确认问题已复现、流水线可信。**

### Phase 1 — 拒绝方向 & 机理诊断
1. 实现 `vpref.py`,抽 $\hat r_\ell$ 并做诱发/抑制校验,选 $\ell^\*$。
2. **机理诊断:** 验证安全关键权重在良性校准集上 $\|X\|$ 偏低(用 Method C 的 $I(W)$ 定位安全权重,统计其 $\|X\|$ 分位 vs 全体)。
- **GATE-1:** $\hat r_\ell$ 校验通过(诱发/抑制有效);机理诊断支持"安全权重良性激活低"。产出 `results/vpref_validation.csv` + `results/mechanism_diagnosis.csv`。**停,确认 V_pref 可用、机理成立。**

### Phase 2 — Method A2(★ 核心假设 go/no-go,最便宜)
1. 实现 `losses.py`(L_refuse/L_utility/A2)、`methods/method_a.py`、`train.py`。
2. **载体先用 LoRA**(只挂 `o_proj`/`down_proj`),训练 → **合并 LoRA 进 W** → 得稠密模型。
3. 用 `compress_and_eval.py` 在 {Wanda,SparseGPT}×{50%} 下评测,对比 Pruned-Aligned。
- **GATE-2(全项目的生死门):** A2 模型在 iso-utility(PPL/MMLU 持平 Pruned-Aligned)下,**剪枝后 ASR 明显低于 Pruned-Aligned**,且 XSTest 误拒未显著上升。
  - **若通过:** 核心假设成立,进 Phase 3。
  - **若不通过:** 先排查(LoRA 秩太低搬不动能量?→ 换"只对写出矩阵 full-FT";目标层选错?λ 不对?)。仍不行则**停下来汇报**,可能需退回方向 4 的诊断式路径(用分解证明输出级保护不足),不要硬推 B。
- 产出 `results/phase2_a2.csv` + 简短 `results/phase2_summary.md`。

### Phase 3 — Method B(主算法,跨配置鲁棒)
1. 实现 `masked_linear.py`(STE)、`methods/method_b.py`,接入 `COMPRESS_CONFIGS` 采样。
2. 训练(先 LoRA-on-write-matrices,信号弱再 full-FT-on-write-matrices)。
3. `compress_and_eval.py` 跑**完整压缩网格**:{Wanda,SparseGPT,magnitude}×{25%,50%,60%,2:4} + 1 档量化(NF4 或 GPTQ/AWQ)。
- **GATE-3:** B 在**多数配置**上 iso-utility 安全保持优于 A2 与 Pruned-Aligned,且**跨压缩范式**(剪枝 **和** 量化)都成立。产出 `results/phase3_b_grid.csv` + 跨配置热力图/曲线。**停,确认主结果成立。**

### Phase 4 — 基线 & 消融
1. 跑基线 3/4/5(Harmful-Calibration、Post-hoc Safety-SFT、PAT),可选 6。
2. **消融:**(a) 静态 $\hat r$ vs **可学安全方向**(训练期可训练、收敛后丢弃的探针,见 §10);(b) LoRA vs full-FT;(c) 目标矩阵/层范围;(d) $\lambda_e,\lambda_u,k$ 敏感性;(e) $\text{COMPRESS\_CONFIGS}$ 采样范围对泛化的影响;(f) Method A1 vs A2;(g) Method C(及其对 Wanda 强、对 SparseGPT 弱的对比)。
- **GATE-4:** 完整主表(各方法 × 各压缩配置 × 三轴)+ 消融表齐备。产出 `results/phase4_main_table.csv`、`results/ablations.csv`。**停,确认对照充分、PAT 确实丢安全。**

### Phase 5 — 泛化 & 收尾(+ 可选 D)
1. 在 `Llama-3.1-8B-Instruct` 上重跑 Phase 2–3 的关键设置,验证跨模型泛化。
2. 能力税曲线(matched-$k$ / matched-PPL 下安全 vs 能力)。
3. (可选)Method D 小规模实验。
4. 汇总所有 CSV → `results/FINAL_TABLES.md`(主表 + 消融 + 泛化 + 能力税),生成论文用图。
- **GATE-5:** 全部结果与图就绪,可写 paper。

---

## 9. 超参默认值(起点,允许调)

| 项 | 默认 |
|---|---|
| 训练步数 $T$ | 1–3k steps(LoRA);视收敛调 |
| LoRA | rank 32, alpha 64, target = {o_proj, down_proj}(必要时加 q/k/v) |
| 学习率 | LoRA 2e-4;full-FT(仅写出矩阵)2e-5 |
| batch / 累积 | 有效 batch 32–64 |
| $\lambda_u$(utility KL) | 1.0(按漂移调) |
| $\lambda_e$(entangle) | 0.1–1.0,扫 |
| A2 的 $k$ | 保留 90% 能量对应秩 |
| 统计重算周期 $N$ | 50–200 steps |
| 校准集 | 128×2048(C4) |
| 评测生成 | max_new_tokens 512, temperature 0(greedy) |
| 种子 | {0,1,2} |

---

## 10. 可选的"训练期可训练件"(仅此一种,且必须可丢弃)

**不要挂可训练掩码(会塌进 PAT)。** 唯一允许且不撞 PAT 的可训练件:把**安全方向/小安全子空间**做成可训练参数 $\hat r_\theta$,与纠缠损失联合优化(让模型自己找最该绑进高能子空间的方向);**训练完该参数不进推理图、直接丢弃**(类似 SAP 的安全探针:训练时插入、收敛后移除)。作为 Phase 4 的消融:静态 difference-of-means $\hat r$ vs 可学 $\hat r_\theta$。LoRA 仅作参数高效训练载体,**剪枝前必须合并进 $W$**,不要带未合并 adapter 去剪。

---

## 11. 正确性检查 / 必须防的坑(Codex 务必遵守)

1. **train/eval 有害提示严格不相交**——否则 ASR 结果无效。Phase 0 GATE 前人工核对。
2. **"保住权重 ≠ 保住行为"**——评测一律基于**剪枝后模型的真实生成**(经 LlamaGuard3 判),不要用权重保留率代替安全指标。
3. **iso-utility 是硬纪律**——比安全前先确认 PPL/MMLU 同档;主表必须三轴同行呈现。
4. **必报过度拒绝(XSTest/OR-Bench)**——防"靠滥拒刷安全"。
5. **STE 方向正确**——前向 $W\odot M$、反向直通到稠密 $W$;用 §5.2 的 `W + (W*M - W).detach()`,不要写成 `W*M`(那会让被剪权重永远拿不到梯度)。
6. **B 的产物是稠密模型**——评测时由 `compress_and_eval.py` 在外部施加压缩,不要在训练里把模型真的剪成稀疏后保存。
7. **LoRA 必须先合并再剪**。
8. **统计量漂移**——$\hat r$、$\|X\|$、$H$ 随权重变,按 $N$ 步重算并缓存。
9. **不学掩码 / 用现成剪枝器判据**——守住与 PAT 的边界。
10. **量化 baseline 别漏**——"跨压缩范式"是核心卖点,剪枝 **和** 量化都要有结果;注意 NF4 量化本就较保拒绝,需在此基础上证明纠缠仍有增益。
11. **HF 路径/接口先验证再用**,不要凭记忆硬编码;数据集字段名以实际为准。
12. **每个 GATE 必须停**,产出指定结果文件并等待人工确认,不要自动续跑下一 Phase。

---

## 12. 交付物清单

- `env_report.txt`、`data/manifest.json`
- `results/phase0_problem.csv` + ASR-vs-sparsity 图
- `results/vpref_validation.csv`、`results/mechanism_diagnosis.csv`
- `results/phase2_a2.csv` + `phase2_summary.md`(**go/no-go 结论**)
- `results/phase3_b_grid.csv` + 跨配置热力图
- `results/phase4_main_table.csv`、`results/ablations.csv`
- `results/FINAL_TABLES.md`(主表/消融/泛化/能力税)+ 论文用图
- 各 Phase 检查点指针与训练日志(wandb 可选)
