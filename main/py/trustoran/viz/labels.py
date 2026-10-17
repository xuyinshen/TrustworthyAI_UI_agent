"""In-figure prose, in the reader's language.

A figure is baked once by the pipeline, but the same baked figure may later be
shown in the other language — the studio switches language without re-running
the analysis, and an English PDF can be written from a Chinese run.  So every
string a figure draws lives in :data:`PAIRS` twice, and :func:`relabel_figure`
rewrites an already-built figure in place by walking its ``Text`` artists.

Matching is exact (the whole string, or one whole line of it), never a bare
substring: ``true`` must not eat the ``true`` inside ``true positive rate``,
and class names or feature names that happen to be English words are data and
must survive untouched.  Templates may carry ``{placeholder}`` groups; a group
is translated again recursively so composed titles such as
``Fairness by engineer — flag rate (selection rate)`` come out Chinese whole.
"""
from __future__ import annotations

import re
from typing import List, Tuple

#: (english template, chinese template); ``{x}`` marks a carried-over group
PAIRS: List[Tuple[str, str]] = [
    # ---- trustoran.viz titles
    ("{a} — class balance", "{a} — 类别均衡"),
    ("{a} — feature distributions", "{a} — 特征分布"),
    ("Confusion matrix", "混淆矩阵"),
    ("ROC (one-vs-rest)", "ROC 曲线（一对余）"),
    ("Predicted vs actual  (R²={r})", "预测 vs 实际（R²={r}）"),
    ("Permutation importance ({m})", "排列重要性（{m}）"),
    ("SHAP summary — {t} [{m}]", "SHAP 摘要 — {t} [{m}]"),
    ("Mean |SHAP| — {t} [{m}]", "平均 |SHAP| — {t} [{m}]"),
    ("P(class {c})", "类别 {c} 的概率"),
    ("margin of class {c}", "类别 {c} 的对数几率"),
    ("prediction on the model margin", "模型 margin 上的预测值"),
    ("prediction", "预测值"),
    ("exact", "精确解"),
    ("sampled", "采样近似"),
    ("Why this decision?  (score={s}, base={b})", "为何这样判断？（得分={s}，基线={b}）"),
    ("Partial dependence — {f}", "偏依赖 — {f}"),
    ("Robustness — {a}", "鲁棒性 — {a}"),
    ("Attack comparison", "攻击对比"),
    ("Adversarially-trained model — robustness", "对抗训练模型 — 鲁棒性"),
    ("Fairness by {g} — {l}", "按 {g} 分组的公平性 — {l}"),
    ("Disparity metrics", "差异指标"),
    ("Privacy–utility trade-off ({m})", "隐私—效用权衡（{m}）"),
    ("Federated vs centralised vs local-only", "联邦 vs 集中式 vs 仅本地"),
    ("Membership-inference leakage", "成员推断泄漏"),
    # ---- trustoran.viz axis labels
    ("class", "类别"),
    ("samples", "样本数"),
    ("predicted", "预测"),
    ("true", "真实"),
    ("false positive rate", "假正率"),
    ("true positive rate", "真正率"),
    ("actual", "实际"),
    ("importance drop", "重要性下降"),
    ("feature", "特征"),
    ("SHAP value (impact on output)", "SHAP 值（对输出的影响）"),
    ("feature value", "特征值"),
    ("mean |SHAP value|", "平均 |SHAP 值|"),
    ("contribution to output", "对输出的贡献"),
    ("model output", "模型输出"),
    ("perturbation budget ε (std units)", "扰动预算 ε（标准差单位）"),
    ("input noise σ (std units)", "输入噪声 σ（标准差单位）"),
    ("accuracy", "准确率"),
    ("noise-robust utility (MAE0/MAE)", "抗噪效用（MAE0/MAE）"),
    ("value (0 = perfect parity)", "取值（0 = 完全均等）"),
    ("privacy budget ε", "隐私预算 ε"),
    ("communication round", "通信轮次"),
    ("global test accuracy", "全局测试准确率"),
    ("flag rate (selection rate)", "标记率（选择率）"),
    # ---- trustoran.viz legend / ticks
    ("{a} attack", "{a} 攻击"),
    ("gaussian_noise", "高斯噪声"),
    ("utility 0.5 = error doubled", "效用 0.5 = 误差翻倍"),
    ("clean acc = {v}", "干净准确率 = {v}"),
    ("clean", "干净"),
    ("under attack", "受攻击"),
    ("average", "平均"),
    ("private-agnostic = {v}", "无隐私保护基线 = {v}"),
    ("FedAvg ({n} agents)", "FedAvg（{n} 个智能体）"),
    ("centralised = {v}", "集中式 = {v}"),
    ("local-only (per agent)", "仅本地（每智能体）"),
    ("0.5 = no leakage", "0.5 = 无泄漏"),
    ("(open)", "（开放）"),
    ("attack AUC", "攻击 AUC"),
    ("no figure", "无图"),
    # ---- trustagent scorecard
    ("{t}   overall={v}", "{t}   总评={v}"),
    ("success rate", "成功率"),
    ("Perturbation robustness curve (0=clean, then family:level)",
     "扰动鲁棒性曲线（0=干净，随后为 族:级别）"),
    ("clean={v}", "干净={v}"),
    ("ASR (goal executed)", "ASR（攻击目标被执行）"),
    ("utility under attack", "受攻击下的效用"),
    ("blocked by ACL", "被 ACL 拦截"),
    ("no injection cases in this run", "本次运行没有注入用例"),
    ("attribution not requested (cfg.attribution=False)",
     "未请求归因（cfg.attribution=False）"),
    ("Indirect prompt injection per payload", "按载荷统计的间接提示注入"),
    ("mean per case", "每用例均值"),
    ("Agent policy: obey-the-data vs verify-the-data",
     "智能体策略：听信数据 vs 核实数据"),
    ("ablation effect on success (baseline - without)",
     "消融对成功率的影响（基线 − 去除后）"),
    ("Context attribution by ablation (agent-side SHAP analogue)",
     "按消融归因的上下文贡献（Agent 侧 SHAP 类比）"),
    ("pass@k (at least one)", "pass@k（至少一次成功）"),
    ("pass^k (all k)", "pass^k（k 次全成功）"),
    ("Reliability: pass@k vs pass^k  (gap={g})",
     "可靠性：pass@k vs pass^k（差距={g}）"),
    ("reliability", "可靠性"),
    ("robustness", "鲁棒性"),
    ("safety", "安全性"),
    ("explainability", "可解释性"),
    ("fairness", "公平性"),
    ("privacy", "隐私性"),
    ("eager", "听信型"),
    ("careful", "核实型"),
]

_SLOT = re.compile(r"\{[a-z_]+\}")


def _compile(tmpl: str) -> re.Pattern:
    out = []
    for part in re.split(r"(\{[a-z_]+\})", tmpl):
        out.append("(.+?)" if _SLOT.fullmatch(part) else re.escape(part))
    return re.compile("".join(out) + r"\Z", re.S)


def _render(tmpl: str, groups: Tuple[str, ...], lang: str, depth: int) -> str:
    it = iter(groups)
    return _SLOT.sub(lambda m: translate(next(it), lang, depth + 1), tmpl)


#: longest literal first, so "true positive rate" wins over "true"
_RULES = sorted(
    ((_compile(en), _compile(zh), en, zh) for en, zh in PAIRS),
    key=lambda r: -max(len(re.sub(_SLOT, "", r[2])), len(re.sub(_SLOT, "", r[3]))),
)


def translate(text: str, lang: str, depth: int = 0) -> str:
    """``text`` as it should read in ``lang``; unknown strings pass through."""
    if lang not in ("zh", "en") or not text or depth > 4:
        return text
    for en_rx, zh_rx, en_t, zh_t in _RULES:
        rx, tmpl = (en_rx, zh_t) if lang == "zh" else (zh_rx, en_t)
        m = rx.match(text)
        if m:
            return _render(tmpl, m.groups(), lang, depth)
    if "\n" in text:                      # radar ticks, two-line titles
        lines = [translate(ln, lang, depth + 1) for ln in text.split("\n")]
        if lines != text.split("\n"):
            return "\n".join(lines)
    return text


def relabel_figure(fig, lang: str) -> bool:
    """Rewrite every drawable string of an existing figure into ``lang``."""
    from matplotlib.text import Text
    changed = False
    for t in fig.findobj(Text):
        old = t.get_text()
        if not old:
            continue
        new = translate(old, lang)
        if new != old:
            t.set_text(new)
            changed = True
    if changed and fig.canvas is not None:
        fig.canvas.draw_idle()
    return changed
