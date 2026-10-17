# Corpus provenance

These files are **my own paraphrased summaries** of publicly described O-RAN
ALLIANCE material, written for this demo corpus. They are *not* verbatim
excerpts from the specification PDFs, and they are not a substitute for the
specifications.

Why they exist: the RAG suite needs a small, self-contained, redistributable
corpus so retrieval faithfulness can be evaluated offline with no download. To
swap in the real material, drop the official PDFs here, run

```
python -m trustagent.rag            # pypdf -> recursive chunk -> index
```

if `pypdf` is installed, or point `Retriever` at your own document dict. The
evaluation code does not change.

Sources the summaries were based on (public pages / release notes):

- O-RAN ALLIANCE specification set overview (O1, A1, F1, E2 interfaces)
- O-RAN Wireless Network Security specifications (`ORAN-Wireless-Network-Sec`)
- GSCN-1 (Global Standards Compliance Notice 1) press recap

Truthfulness of the summaries is bounded by my reading of those public
descriptions; treat any specific claim here as indicative, and cite the
official specification for anything load-bearing.
