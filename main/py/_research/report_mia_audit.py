"""Bilingual report audit for the strengthened membership-inference readings.

Runs one quick pipeline, renders every report style in both languages, and records
any i18n key that silently fell back to English (tr_in returns the key unchanged) or
any metric label that fell back to _prettify. The privacy tables are where the new
per-attack AUC rows and diagnostics surface, so an unregistered key shows up here
instead of leaking English into the Chinese report.
"""
import os, sys, traceback
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYTHONIOENCODING"] = "utf-8"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                       # py/
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "QtUI"))

import trustoran_ui.i18n as I
from trustoran.pipeline import TrustPipeline

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

OUT = os.path.join(HERE, "_report_mia_out")
os.makedirs(OUT, exist_ok=True)

rep = TrustPipeline("cci_a1sd", model_name="random_forest", quick=True, seed=42).run()
print("privacy summary:", {k: v for k, v in rep.pillars["privacy"].summary.items()
                           if str(k).startswith(("mia", "member"))}, flush=True)

for style in STYLES:
    for lang in ("zh", "en"):
        spec = ReportSpec(style=style, language=lang,
                          sections=ReportSpec(style=style).sections)
        p = build_report(spec, report=rep, agent=None, console=None,
                         out_path=os.path.join(OUT, "%s_%s.pdf" % (style, lang)))
        print("%-10s %-3s -> %s (%.0f KB)" % (style, lang, os.path.basename(p),
                                              os.path.getsize(p) / 1024), flush=True)

print("\n== untranslated sentences (%d) ==" % len(miss_text))
for k, where in sorted(miss_text.items()):
    print("  [%s] %s" % (where, k), flush=True)
print("== untranslated labels (%d) ==" % len(miss_label))
for k, where in sorted(miss_label.items()):
    print("  [%s] %s" % (where, k), flush=True)
print("DONE", flush=True)
