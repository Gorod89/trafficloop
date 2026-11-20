# Data licences

What each raw data source allows, and what TrafficLoop publishes for it. The statements were
checked on 2026-09-27 against the sources' own pages, APIs and licence files, and checked again on
2026-11-16 against the same sources: the terms of highD as sent with the access grant and as shown
on its page, the portal records of NGSIM and TGSIM, the JRC catalogue record of OpenACC and the
copyright notice that ships with its files, the Mendeley and figshare records, and
waymo.com/open/terms. Where a source changed its statement between the two dates, both statements
are given with their dates. The machine-readable counterpart, with file lists, sizes and check sums,
is `conf/sources.yaml` (`python scripts/download_data.py --list`). This page records what the
sources state; it is not legal advice.
"Derived events" are the car-following events TrafficLoop extracts: per-sample gap, speeds and
accelerations of a follower and its leader, i.e. trajectory data.

## Licence table

| Source | Licence | Licence URL | Access | Raw data: redistribution | Derived events: redistribution | TrafficLoop publishes | Citation |
|---|---|---|---|---|---|---|---|
| FollowNet event sets (figshare) | CC0 1.0 label on all 26 articles; the NGSIM I-80 set inherits NGSIM's ShareAlike terms (note 3) and the HighD sets the terms of highD (note 1) | https://creativecommons.org/publicdomain/zero/1.0/ | open, direct download | by the label; the upstream terms prevail (note 1) | by the label; the upstream terms prevail (note 1) | no events: every set is rebuilt locally from the figshare files (note 1); scores, also per event with the event identifiers, and trained models | Chen et al. 2023 [1], for the HighD sets also [7] |
| OpenACC (JRC) | CC BY 4.0 by the copyright notice in every campaign folder; the JRC catalogue names the European Commission reuse notice (note 9) | https://creativecommons.org/licenses/by/4.0/ | open, direct download | allowed with credit and changes indicated | allowed with credit and changes indicated | leader profiles and two manually driven platoon windows (release assets) | Makridis et al. 2021 [2] |
| NGSIM (data.transportation.gov) | CC BY-SA 3.0 Unported (note 3) | https://creativecommons.org/licenses/by-sa/3.0/legalcode | open, Socrata API, no account | allowed under the same licence | adaptations: allowed under CC BY-SA 3.0 or a later version (note 3) | human platoon windows with reconstructed NGSIM chains (release asset), as CC BY-SA 4.0 | US DOT FHWA [3] |
| Waymo car-following pairs, Hu et al. (Mendeley Data v3) | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ | open, direct download (note 2) | by the label, with credit; under the Waymo agreement only to registered users (note 2) | as for the raw data (note 2) | models trained on the pairs, as Distributed WOD Models with the notice, the Agreement and the attribution of note 2; scores, also per event with the identifiers of the pairs (segment number and local vehicle number of Hu et al.) in the per-event tables and the split files; no events | Hu et al. 2022 [4] |
| TGSIM (ITS DataHub) | no licence stated on 2026-11-16; U.S. public domain label on the portal until at least 2026-10-07 (note 4) | https://www.usa.gov/publicdomain/label/1.0/ | open, Socrata API, no account | not stated (note 4) | not stated (note 4) | near-miss episodes and human platoon windows with reconstructed TGSIM chains (release assets; note 4) | US DOT FHWA [5] |
| UTE, Ubiquitous Traffic Eyes | none: no licence file; the README asks for a citation | https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye | manual: Google Drive or Baidu links | unclear | unclear | code and check sums only | README [6] |
| highD (levelXdata), raw and through the FollowNet-highD events | non-commercial use; the dataset and modified versions are not to be distributed (note 10) | https://levelxdata.com/highd-dataset/ | by application | forbidden | forbidden (modified versions) | raw highD (dataset `highd`, used locally under its terms): extraction code and aggregate numbers only (JSON summaries, the leaderboard page), no identifiers, tables, split files or checkpoints; FollowNet-highD events (read from figshare, note 1): event check sums, scores, also per event with the event identifiers (row numbers of the FollowNet arrays) in the per-event tables and the split files, and trained models as abstract representations (non-commercial use, citation of Krajewski et al.; `NOTICE.md`); never the events of either | Krajewski et al. 2018 [7] |
| exiD (levelXdata) | non-commercial use; no redistribution | https://levelxdata.com/exid-dataset/ | by application | forbidden | forbidden | extraction code and event check sums, never the events | Moers et al. 2022 [8] |
| CitySim (UCF SST Lab) | terms of the signed request form: non-commercial research and teaching, no redistribution | https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset | by signed form, sent by e-mail | forbidden | forbidden (note 6) | code and check sums only | Zheng et al. [9] |
| pNEUMA (EPFL) | ambiguous: CC BY-NC 4.0 or CC BY 4.0 (note 7) | https://open-traffic.epfl.ch/index.php/downloads/ | open: Zenodo, or the EPFL form with a terms tick-box | unclear | not published until EPFL clarifies | code and check sums only | Barmpounakis and Geroliminis 2020 [10] |

## Notes

1. **FollowNet.** figshare reports CC0 for every article, but the authors' notebook asks users not
   to distribute the data without their permission, and the HighD and 30 s HighD sets are derived
   from highD, whose terms forbid distributing the dataset or modified versions of it (note 10); a
   CC0 label cannot grant more than the FollowNet authors hold. TrafficLoop therefore publishes no FollowNet-highD events
   (decision REL-1, 2026-10-26; the events were removed from the release assets of v0.1.0 to
   v0.3.0, which until then relied on the CC0 grant). They are rebuilt locally:
   `python scripts/download_data.py follownet --yes` fetches the figshare files with their check
   sums and `python scripts/m1.py dataset=follownet_highd "steps=[build]"` rebuilds both versions
   deterministically; the manifest of the events carries their check sums. Events extracted from
   raw highD are never published. Models trained on the FollowNet-highD events are abstract
   representations in the sense of the highD terms and are distributed for non-commercial use with
   the citation of highD (`NOTICE.md`, decision REL-3). The NGSIM I-80 set is derived from NGSIM: its
   CC0 label does not remove NGSIM's attribution and ShareAlike terms, which are kept for what is
   derived from the set (note 3). FollowNet's Waymo set falls under
   the Waymo agreement of note 2; the upstream terms of the Lyft and SPMD sets were not checked.
2. **Waymo pairs.** The Mendeley record carries the label CC BY 4.0 (page and DataCite record, read
   2026-11-16); its licence description warns that further permission may be needed for content of
   the dataset that belongs to a third party. The pairs are derived from the Waymo Open
   Dataset, whose Dataset License Agreement for Non-Commercial Use (March 2025, waymo.com/open/terms;
   checked 2026-10-27) counts every work made using the Dataset, models included, as Derivative IP:
   Section 1 requires an attribution in any Derivative IP, Section 2(b) allows distributing the
   Dataset and modifications of it only to those who registered at waymo.com/open and agreed to its
   terms (so neither the pairs nor events derived from them are redistributed here, whatever the
   CC BY 4.0 label of the Mendeley record says), Section 2(c) allows publishing trained
   models (Distributed WOD Models) for non-commercial distribution only, with a prominent notice that
   they are released under the Agreement and that downstream use is subject to it, a copy of the
   Agreement and the attribution, and Section 4 forbids use in vehicles, production systems or for
   commercial purposes. TrafficLoop complies in `NOTICE.md` (notice and attribution, repeated in the
   release notes and inside `checkpoints.tar.gz`) and `WAYMO_OPEN_DATASET_LICENSE.txt` (the copy);
   the `waymo_hu` runs and their checkpoints are distributed under those terms (decision REL-2). On
   2026-09-27 data.mendeley.com refused python-requests with a browser challenge (HTTP 403). The
   script then asks for a manual download; it does not work around the challenge.
3. **NGSIM.** On 2026-11-16 the portal record
   (`https://data.transportation.gov/api/views/8ect-6jqj.json`) states one licence: `licenseId`
   `CC_30_BY_SA`, "Creative Commons Attribution | Share Alike 3.0 Unported", terms link
   `http://creativecommons.org/licenses/by-sa/3.0/legalcode`; the DCAT file of the portal and
   catalog.data.gov repeat it. Until October 2026 the record carried a second statement, the Common
   Core field `License` with `https://creativecommons.org/licenses/by-sa/4.0/`: seen on 2026-09-27,
   present in the archived copies of the page of 2026-09-19, 2026-10-06 and 2026-10-11
   (`web.archive.org/web/20261011121429/https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj`)
   and absent from the copy of 2026-10-22 on. The field set that the portal uses now (`Common
   Core 3`) has `Rights: public` and no licence. Section 4(a) of CC BY-SA 3.0 Unported allows
   distributing the work itself under that licence only; Section 4(b) allows distributing an
   adaptation under the same licence, a later version of it with the same licence elements, or a
   compatible licence. TrafficLoop publishes no NGSIM trajectory table and no NGSIM events; the
   per-event result tables and the split files hold identifiers of NGSIM vehicles and events with
   sample counts, error sums and (calibration tables) fitted parameters, no series. The file that holds
   processed NGSIM trajectories (the selected and reconstructed NGSIM chains of
   `human_profiles.json`, `docs/protocols.md`) is shared as an adaptation under CC BY-SA 4.0 with
   the NGSIM attribution. The checkpoints trained on NGSIM-derived events alone are offered under
   the same licence with the same attribution; whether a trained model is an adaptation of the data
   in the sense of the licence is not decided here (the label meets the ShareAlike condition if it
   is). (cf-stability's `SOURCES.json` records "public domain (US DOT)", which contradicts the
   portal.)
4. **TGSIM.** On 2026-11-16 none of the five portal records states a licence: no `licenseId`, no
   `license`, no licence in the DCAT file of the portal or on catalog.data.gov. The field set
   `Common Core 3` has `Rights: public`, and every record gives a recommended citation with a DOI.
   When the tables were downloaded (2026-09-28), the records carried the Common Core field
   `License` with the U.S. public domain label `http://www.usa.gov/publicdomain/label/1.0/` and
   `Public Access Level: public` (seen on 2026-09-27; archived copy of the I-90/I-94 stationary page
   of 2026-10-07,
   `web.archive.org/web/20261007200215/https://data.transportation.gov/Automobiles/Third-Generation-Simulation-Data-TGSIM-I-90-I-94-S/9uas-hf8b`).
   The archived copy of 2026-10-16 no longer has the field: the portal dropped the old field set
   without a statement in its place. The near-miss episodes and the TGSIM chains of the human
   platoon windows were published on 2026-10-12 and 2026-10-15, inside that interval, with the
   statement of the download date and the recommended citations; whether the label was still shown
   on those days cannot be told from the archive. The published files stay (decision REL-6 of the
   author, 2026-11-17) with this statement of both facts in place of the label. TrafficLoop names no
   licence of its own for TGSIM-derived files and publishes no further ones while the records state
   no licence.
5. **UTE.** No licence is stated anywhere (repository, READMEs, Drive folders), and the READMEs
   call the release sample data. Being publicly reachable does not grant rights, so nothing derived
   from it is published until the authors grant a licence.
6. **CitySim.** The GitHub repository is Apache-2.0, but that covers the repository contents only.
   The data fall under the request form, which allows derivative works only if the data cannot be
   recovered from them. Events contain the trajectories, so they are not publishable. The
   collaboration sites (Southwest Jiaotong University, Southeast University, PolyU) are
   distributed by those teams.
7. **pNEUMA.** The download page `https://open-traffic.epfl.ch/index.php/downloads/` makes two
   statements, quoted literally:
   - under the DOWNLOAD button: "pNEUMA dataset © 2020 by E. Barmpounakis and N. Geroliminis is
     licensed under CC BY-NC 4.0" (linked to `http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1`);
   - in the FAQ "How can I use the pNEUMA and pNEUMA Vision dataset?": "pNEUMA dataset © 2020
     by E. Barmpounakis and N. Geroliminis is licensed under CC BY 4.0" (linked to
     `https://creativecommons.org/licenses/by/4.0/legalcode`).

   Zenodo (`10.5281/zenodo.10491409`) says `cc-by-4.0`. The terms and conditions page names no CC
   licence; it requires crediting the article and the Zenodo DOI, and the acknowledgement "Data
   source: pNEUMA – open-traffic.epfl.ch". Until EPFL clarifies, TrafficLoop assumes the stricter
   CC BY-NC 4.0 and publishes no data derived from pNEUMA.
8. **exiD.** The format document says the map-based columns (`leadId`, `leadDHW`, `leadDV`,
   `laneletId`, ...) are part of `XX_tracks.csv` only if they belong to the purchased package.
   Whether the academic release has them was not checked.
9. **OpenACC.** Every campaign folder of the download holds the same `copyright.txt` (540 bytes,
   sha256 `f20723519a580fb35b3499870f0b1baea54f483195105416e547f0014f1610d2`, downloaded
   2026-09-27): © European Union, 1995-2026; any copyright or sui generis right on the dataset "is
   licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence", and reuse
   is allowed with appropriate credit and with changes indicated. The JRC catalogue record
   (`https://data.jrc.ec.europa.eu/dataset/9702c950-c80f-4d2f-982f-44d06ea0009f`, doi
   10.2905/JRC.KMH3D00, read 2026-11-16) gives the use conditions as "European Commission reuse
   notice" (reuse authorised with acknowledgement of the source, Decision 2011/833/EU; conditions
   may be set in individual copyright notices) and the access as without limitations; it does not
   name CC BY 4.0. On 2026-11-16 the download server did not answer, so the notice was not read
   again at the source; the local copy has the check sum of `conf/sources.yaml`.
10. **highD.** Terms sent with the access grant and shown on `https://levelxdata.com/highd-dataset/`
    (read 2026-11-16): the dataset is free for non-commercial use only; every work that uses it
    refers to it (item 2); the dataset and "modified versions" are not distributed, while derivative
    works may be distributed as far as they are "abstract representations" of it, such as models
    trained on it or annotations that do not include its data, and do not allow it to be recovered
    (item 3); neither the dataset nor "any derivative work" is used for commercial purposes (item 4).
    Events extracted from highD and the FollowNet-highD events are modified versions and are never
    published. For the FollowNet-highD events, trained models and scores (per event with the event
    identifiers) are published as abstract representations, for non-commercial use, with the
    citation; for raw highD only aggregate numbers are published, without identifiers. That per-event
    scores are abstract representations is our reading of item 3, not a statement of the terms.

## What this means for a benchmark release

- **A release has no single licence.** Data-derived files are published per source, each with the
  terms and the citation of its source: CC BY 4.0 for OpenACC (note 9), CC BY-SA 4.0 for the
  adaptations of NGSIM (note 3), the statement of note 4 for TGSIM. Of the Waymo pairs and the
  FollowNet sets no events, trajectories, series or samples are published; what is published are
  scores, also per event with the identifiers of the events and of the followers (the per-event
  tables, which also hold the fitted parameters of the calibrations, and the split files), and
  trained models. Of raw highD only aggregate numbers are published, without identifiers (notes 1,
  2 and 10).
  NGSIM's ShareAlike rules out releasing NGSIM-derived data under another licence; a file that
  collects windows of several sources is a collection in which every window keeps its own terms.
- **Human platoon windows (P2h).** `data/scenarios/human_profiles.json` holds NGSIM chains, TGSIM
  chains and the two manual AstaZero runs in one file; every window keeps the terms of its source,
  named in its `source` field. The NGSIM chains are adaptations released under CC BY-SA 4.0, as
  NGSIM's ShareAlike requires (note 3); the AstaZero runs are under CC BY 4.0, with credit and the
  changes indicated; the TGSIM chains carry no licence of this repository: their records had the
  public domain label when the tables were downloaded and state no licence since October 2026
  (note 4).
- **Credit and changes.** The CC BY and BY-SA sources require stating what was changed. The
  release notes describe the processing: extraction criteria, resampling to 10 Hz and
  reconstruction, with the config hash.
- **Checkpoints that contain data.** The checkpoint of a kNN entry is a random sample of its
  training data (normalised states and accelerations with the scaler), so the kNN checkpoints of
  every dataset are excluded from the release assets (decision REL-2); the entries are rebuilt in
  seconds with `scripts/train.py dataset=<name> model=knn`. Network weights and calibrated parameters
  are distributed, those trained on the Waymo pairs under the terms of note 2.
- **Restricted sources are reproducible, not redistributed.** For highD, exiD, CitySim, UTE
  and pNEUMA the release contains the extraction code, the
  exact configuration, the sha256 of every raw file (from `SOURCES.json`) and the sha256 of every
  derived event file. A user with their own licence rebuilds the events and checks them bit for bit.
- **Results must not leak the data.** For restricted sources, publish aggregate and per-event
  metrics only. Per-sample outputs, such as simulated trajectories driven by the recorded leader,
  contain the recorded data.
- **No raw data in the release or in git.** `data/raw/` stays in `.gitignore`; the download script
  and this page tell users where to get each source.
- **Before a release,** open every data-derived file and compare its terms with this page. Open
  items: the status of TGSIM (note 4), the upstream terms of FollowNet's Lyft and SPMD sets
  (note 1; the sets are not used), EPFL's answer on pNEUMA (note 7) and a licence from the UTE
  authors (note 5).

## References

References marked \* are given from the literature and were not re-checked against the
publisher on 2026-09-27.

1. Chen, X., Zhu, M., Chen, K. et al. FollowNet: A Comprehensive Benchmark for Car-Following
   Behavior Modeling. Sci Data 10, 828 (2023). https://doi.org/10.1038/s41597-023-02718-7. Data:
   Zhu, M., Chen, X., Chen, K., Wang, P., Lu, H., Zhong, H., et al. (2023). figshare.
   https://doi.org/10.6084/m9.figshare.c.6777810.v1
2. \* Makridis, M., Mattas, K., Anesiadou, A., Ciuffo, B. (2021). OpenACC. An open database of
   car-following experiments to study the properties of commercial ACC systems. Transportation
   Research Part C 125, 103047. https://doi.org/10.1016/j.trc.2021.103047. Data: European
   Commission, Joint Research Centre, Open ACC Database, https://doi.org/10.2905/JRC.KMH3D00;
   © European Union, licensed under CC BY 4.0 (`copyright.txt` of every campaign, note 9).
3. U.S. Department of Transportation Federal Highway Administration. Next Generation Simulation
   (NGSIM) Vehicle Trajectories and Supporting Data [Dataset]. Provided by ITS DataHub through
   Data.transportation.gov. https://doi.org/10.21949/1504477
4. \* Hu, X., Zheng, Z., Chen, D., Zhang, X., Sun, J. (2022). Processing, assessing, and enhancing
   the Waymo autonomous vehicle open dataset for driving behavior research. Transportation
   Research Part C 134, 103490. https://doi.org/10.1016/j.trc.2021.103490. Data: Mendeley Data, V3.
   https://doi.org/10.17632/wfn2c3437n.3
5. U.S. Department of Transportation Federal Highway Administration (2023, 2024). Third Generation
   Simulation Data (TGSIM) <data set> Trajectories [Dataset]. Provided by ITS DataHub through
   Data.transportation.gov. One entry per data set used, with the DOI given on its page:
   I-90/I-94 stationary https://doi.org/10.21949/1404218, I-90/I-94 moving
   https://doi.org/10.21949/1404212, I-294 L1 https://doi.org/10.21949/1404217, I-294 L2
   https://doi.org/10.21949/1404224, I-395 https://doi.org/10.21949/1404222.
6. Ubiquitous Traffic Eyes, https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye: cite what the
   repository README asks for.
7. \* Krajewski, R., Bock, J., Kloeker, L., Eckstein, L. (2018). The highD Dataset: A Drone Dataset
   of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated
   Driving Systems. 21st International Conference on Intelligent Transportation Systems (ITSC),
   2118-2125. https://doi.org/10.1109/ITSC.2018.8569552
8. \* Moers, T., Vater, L., Krajewski, R., Bock, J., Zlocki, A., Eckstein, L. (2022). The exiD
   Dataset: A Real-World Trajectory Dataset of Highly Interactive Highway Scenarios in Germany.
   IEEE Intelligent Vehicles Symposium (IV), 958-964. https://doi.org/10.1109/IV51971.2022.9827305
9. \* Zheng, O., Abdel-Aty, M., Yue, L., Abdelraouf, A., Wang, Z., Mahmoud, N. CitySim: A
   Drone-Based Vehicle Trajectory Dataset for Safety-Oriented Research and Digital Twins.
   Transportation Research Record. https://doi.org/10.1177/03611981231185768
10. Barmpounakis, E., Geroliminis, N. (2020). On the new era of urban traffic monitoring with
    massive drone data: The pNEUMA large-scale field experiment. Transportation Research Part C
    111, 50-71. https://doi.org/10.1016/j.trc.2019.11.023. Data: https://doi.org/10.5281/zenodo.10491409.
    Acknowledgement: "Data source: pNEUMA – open-traffic.epfl.ch".
