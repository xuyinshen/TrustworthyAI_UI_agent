"""Engine-level exit test for the exact-attribution default.

``_research/exact_shap_check.py`` proves the estimator itself (routing, exactness,
agreement with ``shap.TreeExplainer``, cost).  This script proves the change
survives the whole delivery path:

* ``TrustPipeline`` quick runs on an exact model (xgboost / lightgbm) and on a
  model without an exact path (MLP), and the explainability summary must carry
  ``shap_method`` + ``shap_efficiency_max_error`` and name the estimator used.
* The two SHAP figures must be titled with the method tag, and every drawable
  string in them must translate -- English -> Chinese, then back to English --
  so the studio can show a baked figure in either language.
* The PDF report is rendered in both styles' languages with the i18n probes
  armed (same instrumentation as ``report_mia_audit.py``), so a missing
  translation of the new summary fields surfaces here rather than leaking
  English into a Chinese report.
"""
import os, sys, json, warnings, traceback
warnings.filterwarnings("ignore", message="Glyph")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYTHONIOENCODING"] = "utf-8"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "QtUI"))

import trustoran_ui.i18n as I
from trustoran.pipeline import TrustPipeline
from trustoran.viz import labels as L

miss_text, miss_label = {}, {}
_orig_tr, _orig_lab = I.tr_in, I.label_in


def _tr_in(lang, key, **kw):
    out = _orig_tr(lang, key, **kw)
    if lang == "zh" and out == key and len(str(key)) > 3:
        miss_text[str(key)[:110]] = traceback.extract_stack()[-3].name
    return out


def _label_in(lang, key):
    out = _orig_lab(lang, key)
    k = str(key)
    if lang == "zh" and out == I._prettify(k) and not any("一" <= c <= "鿿" for c in out):
        miss_label[k] = traceback.extract_stack()[-3].name
    return out


I.tr_in = _tr_in
I.label_in = _label_in
import trustoran_ui.narrative as N
N.i18n = I
from trustoran_ui.pdf_report import ReportSpec, build_report, STYLES

OUT = os.path.join(HERE, "_exact_engine_out")
os.makedirs(OUT, exist_ok=True)

# dataset, model, expected method, expected english tag, expected zh tag, regression
CASES = [
    ("cci_a1sd_binary", "xgboost", "exact", "[exact]", "[精确解]", "类别 1 的对数几率"),
    ("vran_energy", "lightgbm", "exact", "[exact]", "[精确解]", "模型 margin 上的预测值"),
    ("cci_a1sd_binary", "mlp", "kernel", "[sampled]", "[采样近似]", "类别 1 的概率"),
]

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def title_of(fig):
    return fig.axes[0].get_title()


for dset, mdl, want_method, want_tag, want_zh_tag, zh_target in CASES:
    print("== %s / %s (expect method=%s)" % (dset, mdl, want_method), flush=True)
    rep = TrustPipeline(dset, model_name=mdl, quick=True, seed=42).run()
    ex = rep.pillars["explainability"]
    sh = ex.artifacts["shap"]
    summ = ex.summary

    check(sh.method == want_method, "ShapResult.method == %s (got %s)" % (want_method, sh.method))
    check(summ.get("shap_method") == want_method,
          "summary.shap_method == %s (got %r)" % (want_method, summ.get("shap_method")))
    if want_method == "exact":
        eff = summ.get("shap_efficiency_max_error")
        check(isinstance(eff, float) and eff < 1e-4,
              "summary.shap_efficiency_max_error finite and small (got %r)" % eff)
        check("exact TreeSHAP" in summ["method"], "summary.method names the exact path")
    else:
        check(summ.get("shap_efficiency_max_error") is None,
              "summary.shap_efficiency_max_error is None on the sampled path (got %r)"
              % summ.get("shap_efficiency_max_error"))
        check("KernelSHAP sampling" in summ["method"], "summary.method names the sampled path")
    check(json.dumps(summ, ensure_ascii=False) is not None, "summary is JSON-serialisable")

    # ---- figure titles: tag present, then bilingual round trip -----------
    for key, zh_prefix in (("shap_summary", "SHAP 摘要 — "), ("shap_bar", "平均 |SHAP| — ")):
        fig = ex.figures[key]
        en0 = title_of(fig)
        check(want_tag in en0, "%s title carries %s -- %r" % (key, want_tag, en0))
        L.relabel_figure(fig, "zh")
        zh = title_of(fig)
        check(zh == zh_prefix + zh_target + " " + want_zh_tag,
              "%s Chinese title fully translated -- %r" % (key, zh))
        L.relabel_figure(fig, "en")
        back = title_of(fig)
        check(back == en0, "%s round-trips back to the English title (%r -> %r)" % (key, en0, back))

    # ---- tables surface the target --------------------------------------
    tab = ex.tables["shap_mean_abs"]
    check(len(tab) == len(sh.feature_names), "%s/%s: shap_mean_abs has %d rows" % (
        dset, mdl, len(tab)))

    if (dset, mdl) == ("cci_a1sd_binary", "xgboost"):
        for style in STYLES:
            for lang in ("zh", "en"):
                spec = ReportSpec(style=style, language=lang,
                                  sections=ReportSpec(style=style).sections)
                p = build_report(spec, report=rep, agent=None, console=None,
                                 out_path=os.path.join(OUT, "%s_%s.pdf" % (style, lang)))
                print("   pdf %-10s %-3s -> %s (%.0f KB)" % (
                    style, lang, os.path.basename(p), os.path.getsize(p) / 1024), flush=True)

print("\n== untranslated sentences (%d) ==" % len(miss_text))
for k, where in sorted(miss_text.items()):
    print("  [%s] %s" % (where, k), flush=True)
print("== untranslated labels (%d) ==" % len(miss_label))
for k, where in sorted(miss_label.items()):
    print("  [%s] %s" % (where, k), flush=True)

print("\nfailures: %d" % len(FAILS), flush=True)
if FAILS:
    for m in FAILS:
        print(" - " + m, flush=True)
    sys.exit(1)
print("ALL CHECKS PASSED", flush=True)
