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

## Models trained on highD-derived events

The entries trained on the dataset `follownet_highd` (every run under `results/models/follownet_highd/`
and their checkpoints in `checkpoints.tar.gz`) were trained on car-following events that the FollowNet
release derived from the highD dataset. The terms of highD (levelxdata.com/highd-dataset) allow derivative
works only as abstract representations of the dataset, such as trained models that do not include its
data and do not allow it to be recovered, for non-commercial purposes only, with a reference to the
dataset. These models are therefore distributed for non-commercial use only (academic research,
teaching, scientific publication), may not be used to recover the highD data, and any work that uses
them must cite Krajewski, R., Bock, J., Kloeker, L., Eckstein, L. (2018), The highD Dataset: A Drone
Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated
Driving Systems, IEEE ITSC, doi 10.1109/ITSC.2018.8569552. The MIT licence of the code does not extend to
these models.

## Models trained on NGSIM-derived events

The entries trained on `ngsim` and on `follownet_ngsim` (the NGSIM I-80 event set of the FollowNet release,
which carries a CC0 label on figshare but is derived from NGSIM, so NGSIM's terms apply through the
derivation) were trained on events derived from NGSIM, which the portal publishes under CC BY-SA 3.0
Unported. Their checkpoints are shared under CC BY-SA 4.0, a later version of that licence, with attribution
to the U.S. Department of Transportation, Federal Highway Administration, Next Generation Simulation (NGSIM)
Vehicle Trajectories and Supporting Data (ITS DataHub, doi:10.21949/1504477). Whether a trained model is an
adaptation of the data in the sense of that licence is not decided here; the label meets its ShareAlike
condition if it is.
## Data that is not redistributed

* highD and the FollowNet-highD events derived from it: the terms of highD forbid distributing the dataset
  or modified versions of it. No events, trajectories or samples of them are in the repository or in the
  release assets; the
  events are rebuilt locally from the figshare files of the FollowNet release
  (`python scripts/download_data.py follownet --yes`, then
  `python scripts/m1.py dataset=follownet_highd "steps=[build]"`).
* Waymo car-following pairs (Hu et al.) and events derived from them: the Waymo agreement allows distributing
  the Dataset and modifications of it only to users registered at waymo.com/open, so none are in the
  repository or in the release assets; the result tables hold scores only.
* kNN entries: the checkpoint of a kNN entry is a random sample of its training data, so no kNN
  checkpoint is in the release assets, for any dataset. The entries are rebuilt in seconds with
  `python scripts/train.py dataset=<name> model=knn` and reproduce their published scores.

## Other sources

* NGSIM (CC BY-SA 3.0 Unported on data.transportation.gov; its Section 4(b) allows adaptations under a later
  version of the licence): the human platoon windows `human_profiles.json` contain reconstructed NGSIM chains
  and are released under CC BY-SA 4.0 as a whole; the models trained on NGSIM-derived events, the FollowNet
  NGSIM I-80 set included, carry the same attribution (section above).
* OpenACC (European Commission, Joint Research Centre, doi:10.2905/JRC.KMH3D00; © European Union, CC BY 4.0
  by the copyright notice that ships with the files): the leader profiles `openacc_profiles.json` and the two
  manually driven AstaZero windows of `human_profiles.json`, with the changes stated in `docs/data_layout.md`
  and `docs/protocols.md`.
* TGSIM (U.S. Department of Transportation, Federal Highway Administration, ITS DataHub): the near-miss
  episodes and the reconstructed TGSIM chains of `human_profiles.json`. The portal records carried the U.S.
  public domain label when the tables were downloaded (September 2026); since October 2026 they state no
  licence (`docs/licences.md`, note 4). Cite the data sets as their records recommend.
* FollowNet event sets (Chen et al., CC0 label on figshare): read from the figshare release; the NGSIM I-80
  set inherits NGSIM's ShareAlike terms and the HighD sets the terms of highD, as above.

Citations of every source are in `docs/licences.md`.
