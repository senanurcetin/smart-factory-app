# Smart Factory — Kaggle/UCI/NASA and BigQuery foundation

**Status:** Offline source inventory and SQL compilation, **not a cloud
connection, dataset upload, permission check, remote BigQuery dry run, or
validated Kaggle mirror**. No new live industrial telemetry or model
training takes place in this stage. See [issue #42](https://github.com/senanurcetin/smart-factory-app/issues/42).

## 1. Two distinct scientific sources

| Dataset | Canonical source | Purpose | Existing case-study source fingerprint |
| --- | --- | --- | --- |
| UCI AI4I 2020 | UCI DOI `10.24432/C5HS5C` | Static failure classification, 10,000 observations | `37a525b28d47dc3b5745e851b6ef2582ee5ff6d1a3112b8d2d56ca2f38b134c5` |
| NASA C-MAPSS FD001 | NASA Prognostics Center of Excellence / PHM08 | Sequential turbofan RUL regression, 20,631 training cycles | `35ec8c0ecc5d8497f18b6c11836cda7dc68471e322c7ac345677232199516d59` |

**Critical:** the two fingerprints above are from
`analysis._common.build_model_card`, which hashes
`pd.util.hash_pandas_object(df, index=True).values`. They are **not**
checksums of downloadable ZIP archives or CSVs. Do not copy them into
`archive_sha256`.

The versioned starting inventory is
[`docs/data/smart-factory-source-manifest.json`](data/smart-factory-source-manifest.json).
Both archive checksums, Kaggle mirror handles/versions and license review
are explicitly **pending**. Kaggle mirrors must be independently reconciled
with official UCI/NASA data, source license and split/grain before ingestion.
A Kaggle login by itself does not identify the right dataset.

## 2. Offline source checks (no network)

```bash
python -m analysis.source_registry
python -m unittest discover -s tests -p test_source_registry.py -v
```

This prints source status only. All records initially have
`archive_checksum_pinned=false` and `cloud_ingestion_authorized=false`.
It deliberately **does not download or upload any bytes**.

Once the *actual downloaded file bytes* are independently checked and its
64-character lowercase SHA-256 recorded in the manifest, verify a local
file:

```bash
python -m analysis.source_registry --verify uci_ai4i_2020 /path/to/verified-ai4i.zip
```

An unpinned, truncated, mismatched or symlinked archive fails closed.
This checks raw **file** integrity only: it does not prove row-level
equivalence with the trained benchmark's DataFrame, model-version accuracy,
or redistribution rights. Never commit an archive, auth token or secret.

## 3. BigQuery plan (offline only)

```bash
python -m analysis.bigquery_plan --project workintech-working --location EU
python -m unittest discover -s tests -p test_bigquery_plan.py -v
```

The named project is an *example of the proposed target*, not proof of
verified IAM, billing, quota, existence or regional compatibility. Set your
own verified project/location/prefix before use. The generator prints
schema/table/view SQL as JSON. It creates **zero** datasets, tables, jobs,
queries or cloud spending. Calling it a **BigQuery dry run** would be wrong:
a provider-side dry run has **not** occurred.

Proposed two-dataset naming layout:

- `smart_factory_raw`: `ai4i_records`, `fd001_cycles`, and
  `fd001_official_test_truth`, with explicit archive/source fingerprints
  and ingestion-run IDs.
- `smart_factory_mart`: `ai4i_model_scores`,
  `fd001_model_predictions`, read-only analytical views
  `ai4i_review_queue` and `fd001_official_evaluation`.

The AI4I review view attaches latest model scores only to matching
`source_sha256` and `udi`, and removes duplicate ingestion rows.
FD001 evaluation explicitly uses each held-out engine's **last** test
cycle, joins its separate official ground-truth RUL and a labelled
prediction with the same source hash and engine ID. No AI4I heuristic RUL
is substituted for NASA regression.

This initial SQL schema is a **proposal, not a migration**. The loader,
file-to-field normalization, official FD001 train/test/truth alignment,
schema drift controls, row counts and validation remain to be implemented
before cloud execution.

## 4. Cloud release and cost controls

The ChatGPT session's BigQuery provider plugin is unavailable
(`DISABLED_BY_ADMIN` at discovery) and a direct Kaggle tool is not
exposed. The connector cannot be assumed operational merely because
login or account connections exist elsewhere. Do not claim that data has
been read from, copied to, or validated inside those platforms.

For the next authorized stage:

1. Confirm exact BigQuery project, dataset region and GCP billing state.
   A read-only `nexus-bq-readonly` identity cannot load or mutate tables.
   If ingestion is approved, use a separately scoped **writer** identity,
   preferably through Workload Identity Federation/OIDC; never a committed
   service account JSON private key.
2. Independently pin Kaggle owner/dataset/version and verify licenses,
   archive hashes, official source compatibility and model-card row
   fingerprint semantics.
3. Add a bounded, idempotent *batch* loader with dry-run preflight, row-count
   assertions, canonical columns and unique source/version/run keys.
4. Require provider-side BigQuery dry runs, `maximum_bytes_billed`,
   per-query permissions, dataset-level write scope, job labels and budget
   monitoring **before** remote SQL execution. This code merely suggests a
   1 GiB maximum per future query; it does not enforce one in Google Cloud.
   The BigQuery free usage tier is not an absolute cost cap.
5. Build signed and timestamped versioned export snapshots for the Flask
   app; **do not query BigQuery per anonymous dashboard request**.
6. Run Ruff/Black, all offline tests, Chromium and container CI on the
   exact final PR head before merge, preserving the existing production
   release gates and user-approved NEXUS branding.

No automation, job scheduler, live plant connector, new cloud dataset, new
billing action or deployment is enabled by this documentation.
