# Notices for the data and the trained models

The code of this repository is under the MIT licence (`LICENSE`). The result files, the release assets
and the trained models are derived from third-party data whose terms are summarised in
`docs/licences.md`; the notices below are the ones those terms require.

## Models trained on the Waymo car-following pairs

The entries trained on the dataset `waymo_hu` (every run under `results/models/waymo_hu/` and every
checkpoint of these runs in the release asset `checkpoints.tar.gz`) were developed using the Waymo Open
Dataset through the car-following pairs of Hu et al. They are "Distributed WOD Models" in the sense of
the Waymo Dataset License Agreement for Non-Commercial Use (March 2025), a copy of which is in
`WAYMO_OPEN_DATASET_LICENSE.txt`.

Notice required by Section 2(c) of that Agreement: these models are released under the Waymo Dataset
License Agreement for Non-Commercial Use, and any further downstream use or modification of them is
subject to the terms of that Agreement, including the non-commercial restrictions of its Section 4 (no
use in the operation of a vehicle, in production systems or for primarily commercial purposes).

Attribution required by Section 1 of that Agreement: this software, these models and the results
obtained with them were made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset
License Agreement for Non-Commercial Use, available at waymo.com/open/terms, and your access and use of
such work are governed by the terms and conditions therein.

## Data that is not redistributed

* highD and the FollowNet-highD events derived from it: the terms of highD forbid redistributing derived
  data. No events, trajectories or samples of them are in the repository or in the release assets; the
  events are rebuilt locally from the figshare files of the FollowNet release
  (`python scripts/download_data.py follownet --yes`, then
  `python scripts/m1.py dataset=follownet_highd "steps=[build]"`).
* kNN entries: the checkpoint of a kNN entry is a random sample of its training data, so no kNN
  checkpoint is in the release assets, for any dataset. The entries are rebuilt in seconds with
  `python scripts/train.py dataset=<name> model=knn` and reproduce their published scores.

## Other sources

* NGSIM (CC BY-SA 4.0): the human platoon windows `human_profiles.json` contain NGSIM chains and are
  released under CC BY-SA 4.0 as a whole; events and models derived from NGSIM carry the same attribution.
* OpenACC (CC BY 4.0, European Commission, Joint Research Centre): the leader profiles
  `openacc_profiles.json` and the events derived from the campaigns, with the changes stated in
  `docs/data_layout.md`.
* TGSIM (United States public domain): the near-miss episodes and the events.
* FollowNet NGSIM I-80 events (CC0 1.0, Chen et al.): read from the figshare release.

Citations of every source are in `docs/licences.md`.
