# Hosting the leaderboard

The leaderboard is a static page (`results/leaderboard/index.html`, self-contained, with the
rank-stability plots embedded) plus Markdown tables, rebuilt by `scripts/make_leaderboard.py`
from the run directories under `results/models/`. Nothing in it needs a server, so the
recommended setup is a public GitHub repository with GitHub Pages and a submission
workflow by pull request.

## 1. Repository and page

* Push the repository to GitHub (code, configurations, docs, `results/**/*.json`, the small
  tables, `results/leaderboard/`, `results/figures/`; checkpoints, events, splits and the
  profile cache stay out, as `.gitignore` says).
* Enable GitHub Pages for the branch `main`, folder `/docs` (or a `gh-pages` branch), and
  copy `results/leaderboard/index.html` to `docs/leaderboard/index.html` in the release
  commit; the page is then served at `https://<org>.github.io/trafficloop/leaderboard/`.
  The workflow in `.github/workflows/leaderboard.yml` rebuilds the page on every push to
  `main` and uploads it as the Pages artefact, so the copy step is automatic.
* Give the release a DOI through Zenodo (GitHub integration: one click per release); the OpenACC leader
  profiles (`data/scenarios/openacc_profiles.json`, CC BY 4.0), the checkpoints without the kNN entries (with `NOTICE.md` and the Waymo Agreement inside, since the
  `waymo_hu` models are Distributed WOD Models) and the human platoon windows of P2h
  (`data/scenarios/human_profiles.json`; every window keeps the terms of its source, named in its `source`
  field: NGSIM chains CC BY-SA 4.0, AstaZero runs CC BY 4.0, TGSIM chains without a licence of this
  repository, see `docs/licences.md`, note 4) go into the same Zenodo
  record as data files, which keeps the repository small and the data citable. A copy of
  the events on Hugging Face Datasets is optional and convenient for `datasets.load_dataset`.

## 2. Submissions

An entry is a run directory (`docs/leaderboard.md`, section 1). A submitter

1. trains and evaluates with the benchmark's scripts (or their own code that writes the same
   files: `train.json` with the provenance block, `p1.json` ... `p4.json` with the
   `checklist` block, the per-event parquet tables),
2. opens a pull request that adds `results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/`
   (JSON and parquet only; the checkpoint goes to a release asset or Zenodo and its URL into
   `train.json` under `provenance.checkpoint_url`),
3. the workflow runs the tests and `scripts/make_leaderboard.py` as a check; a run whose
   checklist is incomplete is listed under "refused entries" with the reason,
4. a maintainer rebuilds the pages with all per-event tables, commits them and merges; the
   page that is published is the committed one. (The per-event tables of the dataset `highd`
   are not distributed, so a rebuild from the release assets alone has no intervals for it.)

Because the composite rank intervals depend on the whole set of entries, the leaderboard is
rebuilt as a whole on every merge; nothing is computed incrementally.

## 3. Alternatives

* **Codabench / EvalAI**: suited to hidden test sets and automatic scoring of uploaded
  predictions. The benchmark scores closed-loop behaviour of a model, not predictions, so a
  submission would have to be a model (code and weights) that the platform runs; this is
  possible with a Docker image but costs maintenance. Recommended only if the test parts are
  to be hidden later.
* **Hugging Face Space (static)**: the same HTML page, hosted next to the data; no gain over
  GitHub Pages except discoverability.
* **A self-hosted server**: not needed; every artefact is static.

## 4. What to publish with the page

`results/SUMMARY.md` (findings), `docs/checklist.md` (what an entry must state),
`docs/protocols.md` (what the numbers mean), the figures, and a `CITATION.cff` with the
paper and the Zenodo DOI.
