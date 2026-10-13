# Verification and engineering proof

Verified locally on **2026-10-12**, Windows 11, Python 3.12.10, pytest 9.1.1 and Ruff 0.17.0.
Tesseract was not installed. These results describe this environment; the separate CI `ocr`
job installs Tesseract on Linux and uploads its own benchmark report.

## Reproduce

From the repository root, after installing `.[dev]` into `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m ruff check src tests scripts examples
.\.venv\Scripts\python.exe examples\quickstart.py
.\.venv\Scripts\python.exe examples\custom_schema.py
.\.venv\Scripts\python.exe -m privdoc bench --tier local --per-kind 2 --json bench-results.json
.\.venv\Scripts\python.exe -m pytest -q -ra -m tesseract
```

On Linux/macOS, use `.venv/bin/python`. The demo writes synthetic invoices to ignored `out/`;
the benchmark JSON is also ignored. No account, key or external download is needed.

| Check | Actual result |
|---|---|
| Complete pytest suite | **612 passed, 4 skipped**, exit 0; 56.57 s on this run |
| Ruff | **All checks passed**, exit 0 |
| Quickstart | Exit 0; typed invoice valid/complete, scan unread with `not_checked`, no data destinations |
| Custom-schema example | Exit 0; valid support-ticket JSON with eight found fields |
| Local benchmark | Exit 0; **30 documents, 42 pages**, no pipeline errors, $0 estimated model cost |
| Explicit real-Tesseract selection | **3 skipped, 613 deselected**, exit 0; no real OCR claim is made |

### Why four tests are skipped

- One test in [`test_cache.py`](../tests/test_cache.py) checks POSIX mode bits. Windows does
  not implement that permission model, so `POSIX permissions` is an intentional skip.
- Three cases in [`test_tesseract_integration.py`](../tests/test_tesseract_integration.py)
  require the real `tesseract` executable. It was absent from PATH. Unit tests fake its
  subprocess and TSV output; those passing tests do not replace real-binary integration.
- The Linux test job covers POSIX permissions. The Linux `ocr` job installs Tesseract and
  selects the real-binary cases. Its result must be read from GitHub Actions after a run.

## Benchmark result and interpretation

| Slice | Documents / pages | Correct present fields | Returned hallucinations | Invented absent fields |
|---|---|---|---|---|
| Typed text | 6 / 8 | **49/49 (100.0%)** | 0/49 | 0/9 |
| Mixed typed/scanned | 6 / 12 | **25/49 (51.0%)** | 0/25 | 0/9 |
| Scan, searchable scan, photo (each) | 6 / 8, 6 / 8, 6 / 6 | **0/49** | No values returned | 0/9 |
| All | 30 / 42 | **74/245 (30.2%)** | 0/74 | 0/45 |

Scanned pages were deliberately left unread without an OCR engine; `not_checked` preserves
that uncertainty. The 100% typed score applies only to these six generated documents, not
arbitrary invoices. The rule extractor copies page text, so zero hallucinations here is
structural, not measured LLM accuracy. No cloud provider, hosted model or real Tesseract was
used in this local benchmark. The [benchmark documentation](../README.md#benchmark) defines
the denominators and supplies larger and cloud-enabled commands.

## Proof map

| Design property | Implementation | Evidence to inspect |
|---|---|---|
| Privacy tier is checked at planning and send time | [`plan.py`](../src/privdoc/plan.py), [`privacy.py`](../src/privdoc/privacy.py) | [`test_plan.py`](../tests/test_plan.py), [`test_privacy.py`](../tests/test_privacy.py); [`conftest.py`](../tests/conftest.py) supplies socket/DNS tripwires |
| Page routing distinguishes text layers, scans and typed pages over backgrounds | [`document.py`](../src/privdoc/document.py), [`routing.py`](../src/privdoc/routing.py) | [`test_document.py`](../tests/test_document.py), [`test_pipeline.py`](../tests/test_pipeline.py) cover mixed and searchable scans plus letterhead/background regressions |
| Unsupported schema features fail instead of being silently ignored | [`schema.py`](../src/privdoc/schema.py), [`coerce.py`](../src/privdoc/coerce.py) | [`test_schema.py`](../tests/test_schema.py), [`test_coerce.py`](../tests/test_coerce.py) cover strict validation, nested optional values and coercion |
| Grounding and escalation preserve evidence and more trusted fields | [`grounding.py`](../src/privdoc/grounding.py), [`fields.py`](../src/privdoc/fields.py), [`escalation.py`](../src/privdoc/escalation.py) | [`test_grounding.py`](../tests/test_grounding.py), [`test_fields.py`](../tests/test_fields.py), [`test_escalation.py`](../tests/test_escalation.py) reject word fragments and exercise confidence/merge regressions |
| URL fetches pin checked addresses and enforce bounds on each redirect and socket operation | [`fetch.py`](../src/privdoc/fetch.py) | [`test_fetch.py`](../tests/test_fetch.py) uses fake DNS plus loopback HTTP servers for rebinding, encoded-address tricks, oversized chunked bodies and slow headers/chunks |
| Benchmark scoring is separate from pipeline grounding | [`bench.py`](../src/privdoc/bench.py), [`synth.py`](../src/privdoc/synth.py) | [`test_bench.py`](../tests/test_bench.py), [`test_synth.py`](../tests/test_synth.py) cover metric denominators, absent fields and generated ground truth |

## Review changes and boundaries

The README now leads with enforceable mechanisms, actual offline commands and a live CI link,
instead of a static passing-test badge. CI runs both examples explicitly and grants its token
read-only repository permissions. No runtime defect was found during this review.

Built-in local extraction rejects URL input and document egress. This is an application
boundary, not a sandbox for arbitrary custom Python code. An own-cloud destination is declared
by your configuration, not independently certified as belonging to your account. The cache
holds plaintext extraction results; Windows ACL enforcement was not tested. Model accuracy,
provider retention, real OCR quality and multi-machine performance remain outside these tests.
See [security notes](../README.md#security-notes) for operational constraints.

## Public-release audit

Reviewed the 77 tracked files and 115 unique historical text blobs reachable from the eight
existing commits. Enumeration used `git ls-files`, `git rev-list --objects --all` and
`git cat-file --batch`; commit attribution used
`git log --all --format="%h %an <%ae> | %s"`.

Pattern scans and source/example review found no credential markers, private workspace or
infrastructure identifiers, or non-example email addresses in file contents. Synthetic
documents are generated from fictional values; generated outputs and downloaded datasets
are ignored. Maintainer name and email remain in Git authorship metadata and will be public.
No release blocker was identified and no history was rewritten. A heuristic scan is not a
complete proof that all possible sensitive content has been excluded.
