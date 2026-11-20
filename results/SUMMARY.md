# TrafficLoop: what the benchmark found

Summary of milestones 1-5 and of the additions after them (protocols P1s and P2h, analyses
E1b, E3d and E5, the certified entry). Numbers are from `results/m4/REPORT.md` (driver
split, fold 0, seed 0, full training parts, reconstructed data; intervals are 95 % driver
bootstraps with 1000 replicates unless stated), `results/sweeps/` (E5) and
`results/leaderboard/`. Protocols: `docs/protocols.md`; experiments: `docs/experiments.md`;
every decision and deviation: `docs/decisions.md`.

## The benchmark in one paragraph

Six datasets with car-following events on the common 10 Hz schema (NGSIM, FollowNet-HighD,
OpenACC, Waymo, FollowNet-NGSIM, TGSIM; 55,000 events), each in a raw and a
physics-reconstructed version, with event-, driver-, site-, regime- and block-level splits
and a leakage check. Twenty-five reference entries: two baselines (persistence, constant
speed), three mechanistic models in two calibrations each (pooled, median of per-event),
sixteen learned models from kNN to diffusion, including closed-loop-trained variants, and a
certified hybrid (an IDM core plus a bounded residual network whose string stability is
certified a priori). The protocols: one-step prediction (P0), closed loop over 1 s windows
(P1s) and over 10 s, 30 s and the whole event (P1), platoons of 100 followers behind
recorded OpenACC leaders and a braking pulse (P2) and behind 98 platoon windows with human
followers from NGSIM, TGSIM and AstaZero (P2h), string stability by the analytic criterion
and a step-response gain test (P3), a rare-event suite of pulses, a full stop, cut-ins and
4,473 confirmed near-miss episodes from TGSIM (P4), and transfer between datasets (P5). A
composite traffic-level score ranks the models on the concave-growth error, the hysteresis
error, the unstable fraction and the platoon collisions.

A seventh dataset, raw highD (the benchmark's own extraction of the dataset for which the plan
registered H5.1, H5.3 and H5.4; 12,670 events, kept local, results as aggregate numbers only),
was added on 2026-11-17. Findings 1 to 14 below are stated for the six earlier datasets unless
they name raw highD; finding 16 gives what holds on it.

## Headline findings

1. **One-step accuracy is the wrong yardstick (H5.1, H5.3).** On NGSIM only the diffusion
   model beats the persistence baseline at one step (ratio 0.62 [0.57, 0.67]); every other
   model is worse, the MLP by 3 %, the LSTM by 30 %, the mechanistic models 19 to 75 times.
   On FollowNet-HighD nothing beats persistence (best: MLP, 2.5 times). The one-step
   ranking is *anti*-correlated with the traffic-level composite: Kendall tau -0.52
   [-0.60, -0.45] on NGSIM, -0.53 [-0.56, -0.44] on HighD and -0.41 to -0.64 on the four
   other datasets. The sign holds under every composite of the sensitivity analysis (realism
   inputs added, a Borda count, each input left out, doubled safety weights): all 84
   intervals of tau(P0, variant), 14 variants on six datasets, lie below zero (highest upper
   bound -0.013). The top three under P0 (NGSIM: diffusion, persistence, MLP; HighD:
   persistence, MLP, mlp_loop) are replaced under the composite by mechanistic models on
   NGSIM (Newell median, Newell, IDM median) and by the certified hybrid, DDPG and IDM on
   HighD.
2. **Closed-loop accuracy depends on the horizon, and on the dataset.** On NGSIM the
   one-step networks match the mechanistic models at 10 s (LSTM 3.07 m, IDM 3.36 m) and
   diverge afterwards (LSTM 81.6 m over the whole event, IDM 6.1 m); the closed-loop-trained
   variants lead at 10 s, 30 s and over the whole event (mlp_loop 2.48 / 5.64 m at 10 s /
   whole event). Kendall tau between the P0 and the whole-event ranking is -0.31 on NGSIM;
   between 10 s and the whole event 0.35. On HighD (15 s events, free flow) the learned
   models lead everywhere (gru_loop 3.14 m, seq2seq 3.16 m, mlp_loop 3.17 m, IDM 4.0 m), and
   five of the six best (all but seq2seq, which has no equilibrium) are string unstable at
   every equilibrium and collide in 1,076 to 1,794 of the 1,900 platoon followers: the
   whole-event ranking is anti-correlated with the composite there (tau -0.36).
3. **Trajectory imitation does not transfer to traffic-level behaviour.** Of the 17
   learned entries, 8 (FollowNet-NGSIM) to 14 (OpenACC) are string unstable at every
   equilibrium P3 finds (gains 1.06 to 28) and up to five more at 64 to 96 % of them
   (entries without an equilibrium, one or two per dataset, are not counted); DDPG is
   unstable at 6 % of the HighD equilibria and 42 % on NGSIM (0 to 65 % over the six
   datasets), the certified hybrid at none (finding 11). The stability-regularised MLP is
   unstable at 35 % of its NGSIM equilibria at the verdict tolerance, but with gains of at
   most 1.0001 (none in the 0.1 % band), and amplifies perturbations of 1 m/s at 92 % of
   them. In 100-vehicle platoons on the in-range OpenACC leaders the mechanistic models,
   DDPG and the certified hybrid never collide on NGSIM, while the other learned models
   collide in 272 to 998 of the 1,000 followers (seq2seq 40); on HighD only the pooled IDM,
   Gipps and the certified hybrid never collide (DDPG 433, the other four mechanistic
   calibrations 81 to 1,123), and the other learned models collide in 181 to 1,794 of the
   1,900 followers (seq2seq 62). In the synthetic rare-event scenarios the mechanistic
   models and the certified hybrid never collide on NGSIM, DDPG collides in 1 of the 14 and
   the other 15 NGSIM networks in 14 to 100 % (13 of them in 64 % or more; partly outside
   their speed range); in the TGSIM near-miss episodes the NGSIM networks collide in 8 to
   36 %, against 1.5 to 7 % for IDM and Gipps (NGSIM and HighD) and 3.9 % for the certified
   hybrid.
4. **Driver leakage inflates little; H5.2 is not confirmed (paired design).** The plan registered
   H5.2 as an inflation of at least 15 % of the one-step RMSE of the networks under the event-level
   split and below 5 % for the IDM; the first condition fails on every dataset (at most 5.4 %), so
   the hypothesis is not confirmed. Moving half of the events of
   every test driver into the training part changes the one-step error by about 5 % at most
   (NGSIM: MLP 5.4 %, IDM 2.8 %, LSTM 0.1 %; OpenACC, where every probe driver leaks about
   95 events: lstm_loop 5.4 %, networks on average 1.8 %), the 10 s closed-loop error by at
   most 4 % on NGSIM and 7 % on OpenACC, and the whole-event error notably only for the
   one-step MLP on NGSIM (25 %, 8 % on the affected events; the OpenACC changes of up to
   11 % are not significant after Holm). Differences are often significant after Holm but
   small; the unpaired event-against-driver comparison moves by the same order in both
   directions. Driver identity is not what these models memorise.
5. **Rankings by spacing and by speed agree (H5.4 not supported).** Kendall tau between the
   P1 rankings by spacing and by speed error is 0.83 to 0.87 at every horizon on NGSIM and
   HighD, 0.79 to 0.89 on all six datasets.
6. **Transfer is asymmetric.** Trained on NGSIM (congested), the networks degrade up to 13
   times over the whole event on HighD (GRU 13.0, gru_loop 13.3, diffusion 12.5, LSTM 11.7)
   and up to 51 times on OpenACC (gru_loop; LSTM 43), the mechanistic models 1.4 to 3.8
   times on HighD and up to 14 times on OpenACC. DDPG transfers everywhere within 20 % (mean
   degradation 1.01 over its 30 foreign pairs, worst 1.17), the certified hybrid within 65 %
   (mean 1.19, worst 1.65). Trained on HighD (free flow), every model holds up on NGSIM over
   10 s windows (0.5 to 1.6 times). Ratios below 1 on TGSIM and NGSIM mean that the native
   one-step networks diverge in closed loop on their own long events (TGSIM: LSTM 542 m, MLP
   444 m over the whole event; their closed-loop-trained variants 7 to 11 m), not that the
   foreign model is good.
7. **The data versions matter.** Raw NGSIM acceleration is noise (persistence 71 % of a
   constant-speed follower), raw TGSIM acceleration is unusable (99 % quantile 18 m/s^2);
   the reconstruction makes both consistent with the integration scheme. Near-miss triggers
   mined on raw tracks survive the reconstruction in 84 % of the TGSIM episodes but only
   38 % of the NGSIM ones (leader braking: 6 %).
8. **The closed-loop divergence sits in a few followers (E1b).** At 1 s the closed-loop
   ranking still follows the one-step ranking (Kendall tau with P0 0.85 to 0.95 on the six
   datasets, 0.84 to 0.92 with medians over windows); at 10 s it is 0.23 to 0.55 (medians
   0.11 to 0.63). Over the whole event it turns negative with the pooled RMSE on the four
   datasets with 30 s windows (NGSIM -0.31, FollowNet-NGSIM -0.16, OpenACC -0.23, TGSIM
   -0.40; HighD 0.50 and Waymo 0.09 on their short events) but is near zero or positive with
   medians over events (NGSIM -0.15, FollowNet-NGSIM -0.07, OpenACC 0.31, TGSIM 0.08). The
   divergence of the one-step networks in closed loop is concentrated in a few followers: on
   NGSIM five followers carry 99.0 and 99.6 % of the squared whole-event error of the LSTM
   and the GRU, whose median errors over events (4.7 and 4.8 m) are close to the IDM's
   (4.0 m).
9. **Tuning buys one-step accuracy, not stability (E5).** Sixteen configurations each of
   the LSTM and the MLP on NGSIM (width, depth of the MLP, history, learning rate, weight
   decay, batch size, the previous acceleration as input), selected on the validation part by
   the one-step error and by the closed-loop 10 s spacing error. For the LSTM both criteria
   pick the same configuration (256 units, 3 s history, previous acceleration as input):
   one-step error 0.0119 m/s^2 on the test part against 0.116 for the reference
   configuration, whole-event error 6.5 m against 81.7 m, but string unstable at every
   equilibrium, and 614 of the 1,000 in-range platoon followers collide. For the MLP the
   criteria disagree (Kendall tau between them over the 16 configurations -0.23; LSTM 0.18):
   the two selections have whole-event errors of 44.9 and 46.7 m against 9.8 m for the
   reference, and both are unstable at every equilibrium (424 and 599 collisions). With the
   previous acceleration as input the one-step error falls far below that of persistence
   (0.0077 to 0.0119 against 0.089 m/s^2): one-step accuracy is cheap.
10. **A human platoon reference changes little (P2h, E3d).** Behind the 98 windows of P2h
    with human followers (40 NGSIM and 56 TGSIM chains, two manual AstaZero runs) the
    composite ranks the entries much as with the ACC-driven OpenACC reference: Kendall tau
    0.64 to 0.77 between v1 and v1h and 0.60 to 0.83 between v2 and v2h over the six
    datasets, and the one-step ranking stays anti-correlated with it (tau(P0, v1h) -0.48 to
    -0.54, tau(P0, v2h) -0.27 to -0.57). Under the human reference the first three are
    mechanistic entries or the certified hybrid on five of the six datasets; on
    FollowNet-HighD DDPG (v1h) and GAIL (v2h) take the third place.
11. **The certified hybrid buys string stability with trajectory fidelity.**
    `residual_idm_certified` (decision M2-8, `docs/models.md`) is string stable at every one
    of its 26 equilibria on every dataset and never collides in the platoons (P2, P2h) or in
    the synthetic scenarios; its whole-event spacing error is 5 to 50 % above the IDM's
    (NGSIM 6.44 against 6.11 m, HighD 6.01 against 4.01, OpenACC 11.70 against 11.19, Waymo
    5.79 against 5.52, FollowNet-NGSIM 6.23 against 5.82, TGSIM 8.43 against 7.83) and its
    one-step ratio 17 to 125, like those of the mechanistic models. Composite rank 1 of 25
    on FollowNet-HighD, 3 on Waymo, 5 on TGSIM, 8 on OpenACC, 10 on NGSIM and 13 on
    FollowNet-NGSIM (v2: 2, 3, 4, 3, 9 and 7.5); mean transfer degradation 1.19, second to
    DDPG (1.01) among the non-baseline entries. The core with a stability margin wins the
    budget on every dataset: the pooled IDM core admits none on four datasets and too little
    for the residual to act on NGSIM (0.0018) and Waymo (0.12), against 0.15 to 0.34 for the
    margin core (lipschitz 0.45 to 1.02 at r_max = 0.3 m/s^2).

12. **The headline correlation is not produced by the construction of the composite (E3f, exploratory).** An entry
    without a value of an input takes the last rank on it; only the unstable fraction is ever missing (persistence
    and constant speed everywhere, one or two learned entries per dataset). Without the two baselines tau(P0,
    composite) is -0.38 to -0.68 (further from zero than with them on five of six datasets); with every entry scored
    on the inputs it has, -0.43 to -0.62; with both, -0.39 to -0.68; every interval excludes zero. Leaving out any
    single entry moves tau by at most 0.06. Over the three seeds and five folds of the main datasets it ranges from
    -0.43 to -0.56 (NGSIM) and -0.43 to -0.53 (FollowNet-HighD): spreads of 0.13 and 0.10, close to the widths of the
    bootstrap intervals (0.15 and 0.12), which resample data and not training runs; one run per dataset lies above
    the interval of the main run. In these seven runs the top three of NGSIM are always the Newell median, Newell
    and the IDM median; on FollowNet-HighD DDPG is first in five runs and the certified hybrid in two (the main run
    among them) and second in five. Leaving one input out of the four-input composite: without the concave-growth
    or the hysteresis error tau stays between -0.45 and -0.63; without the unstable fraction or the collision count
    it is weaker on every dataset (-0.35 to -0.60 on five; -0.23 and -0.16 on FollowNet-NGSIM, p = 0.11 and 0.27
    over the 25 entries): on FollowNet-NGSIM the correlation rests on the two safety inputs, on the other five no
    single input carries it. With a collision gate (entries that collide in any in-range platoon rank last) 3 to 9
    entries per dataset stay admissible and tau is -0.44 to -0.64. The collision input counts the followers whose
    own gap closes and lets a collided vehicle drive on, so followers that never react count one collision per
    profile (persistence, constant speed: 8 to 17 on 1,000 to 2,000 followers); with the followers at or behind
    the first collision, or the profiles with a collision, in its place the baselines fall by 0.5 to 4 places and
    tau is -0.34 to -0.57 and -0.33 to -0.60 (every interval excludes zero; DDPG moves ahead of the certified
    hybrid on FollowNet-HighD under the first count). The stability rule counts a model that passes a perturbation
    on unchanged as stable: the two Newell entries (gain 0.9995 at 0.1 Hz) have an unstable fraction of zero and
    lead the composite on NGSIM, OpenACC and TGSIM. With neutral stability counted as instability only these two
    entries change, tau weakens on every dataset (-0.28 on FollowNet-NGSIM, p = 0.055; -0.40 to -0.56 on the other
    five) and the leaders on the three datasets become the IDM median, Gipps, DDPG and the pooled IDM; a tolerance
    of 1e-3 instead of 1e-5 changes no leader. The correlation is mainly a
    difference between the families: inside the 17 learned entries it is negative on all six datasets but
    significant on three, inside the six mechanistic entries there is no evidence either way (E3e).
13. **The correlation over all entries is largely a contrast of training objectives (E3g, exploratory).** The
    six calibrated entries, never fitted to the one-step acceleration, hold a median rank of 22.5 of 25 under P0
    on every dataset and of 3.5 to 5.75 under the composite; the ten entries trained on the one-step error alone
    hold 7 to 9 and 17 to 20.5; DDPG, trained on a reward, sits with the calibrations (20 or 21 under P0, 2 to 8.5
    under the composite); a rollout or multi-step loss leaves the one-step places and improves the composite on
    NGSIM and FollowNet-NGSIM only. With the objective held fixed the evidence is weak: inside the ten entries
    with a one-step objective tau(P0, composite) is negative on all six datasets (-0.04 to -0.64), inside the nine
    others (calibrations, policies, certified hybrid) on five (-0.11 to -0.44; 0.03 on FollowNet-HighD), and after
    Holm's correction over the six datasets it is significant on none in either group (uncorrected: NGSIM and
    TGSIM in the first). Inside the learned family it stays significant on three datasets after the correction. The
    benchmark cannot separate the objective from the model family: no mechanistic entry is calibrated on the
    one-step error and no network is trained on the closed-loop error alone.

14. **The certified hybrid improves on the calibrated IDM only where that model is unstable (exploratory).** The
    pooled IDM is stable at every equilibrium on four datasets and unstable at 64 % (FollowNet-HighD) and 27 %
    (OpenACC); the hybrid is stable everywhere, has a higher whole-event error on all six datasets and a better
    composite on FollowNet-HighD alone (0.115 against 0.177). Training its bounded residual lowers the one-step
    validation error by 1.6 to 3.7 %: the residual is nearly inert. On raw highD (finding 16) the pooled IDM is
    unstable at 19 % of its equilibria and still ranks ahead of the hybrid (0.161 against 0.214, ranks 2.5 and 6).
15. **The stability of the certified hybrid is that of its core (exploratory, decision REV4-1).** The core of the
    hybrid, an IDM calibrated with the stability margin 0.2, scored as an entry of its own (`idm_margin`, no residual,
    no training; `results/exploratory/margin_core/comparison.csv`), is string stable at every equilibrium and
    collision-free in the in-range platoons on all seven datasets, takes the hybrid's composite and rank on six of
    them (half a place less on OpenACC) and has a slightly lower whole-event error on all seven; its one-step error
    is 1.7 to 3.7 % higher. Calibrating the IDM with a stability margin removes the instability at a cost in
    accuracy; the certified learned residual adds nothing measurable at the traffic level.
16. **The tests registered for highD give the same verdicts on raw highD (confirmatory; HD-1 to HD-5).** Own
    extraction: 12,670 events of 12,603 followers. Deciding run (driver split, fold 0, seed 0): no entry within 10 %
    of persistence and all 23 more than 30 % behind it (best: the MLP, 2.78 times); the three most accurate
    entries, one step ahead and over the whole event, all leave the top three of the composite; tau between the
    rankings by spacing and by speed error 0.85 over the whole event. Exploratory: tau(P0, composite) is -0.50
    [-0.57, -0.40]; over the seven training runs (three seeds, five folds) the statistics of the three tests stay
    the same (best ratio 2.67 to 2.93, three of three dropped, whole-event tau 0.83 to 0.90) and so they do under
    the split by blocks of recordings, which shares no vehicle between training and test (tau(P0, composite)
    -0.41). The leader of the composite is not robust there: the IDM in three of the seven runs, Newell in two, the
    IDM median and DDPG in one each (DDPG in the deciding run); the certified hybrid, at ranks 5.5 to 7, is the only
    entry that is string stable and collision-free in every run. H5.2 was not run on highD (0.53 % of the followers
    have two events; the contract required 5 %). On NGSIM one of the seven training runs (seed 1) has a whole-event
    tau of 0.75 for H5.4, below the threshold of refutation (`results/m4/e3f_run_statistics.csv`).
17. **Three further checks (E3h, exploratory, decision REV5-1).** (a) H5.1 on the ten entries with a
    one-step objective gives the verdict of the 23: on NGSIM one within 10 % of persistence, one more than 30 %
    ahead and six more than 30 % behind; on both highD sets all ten more than 30 % behind. (b) With the P2 inputs
    over all leader profiles instead of the in-range ones, all three most accurate entries still leave the top
    three on NGSIM, FollowNet-HighD and raw highD (tau -0.56, -0.53, -0.52); the top three change on NGSIM (DDPG,
    the certified hybrid, Newell) and raw highD (DDPG, IDM, IDM median). (c) Permuting the composite inside the six
    objective groups, which keeps only the contrast between the groups, leaves a mean tau of -0.33 to -0.48: 74 to
    83 % of the observed correlation on six datasets and all of it on Waymo; the observed tau is in the lower tail
    of this reference on TGSIM, NGSIM and FollowNet-HighD only (shares 0.012, 0.015, 0.037, uncorrected; none below
    0.05 after Holm's correction). The correlation over all entries is largely a contrast between kinds of entries
    (`results/m4/e3h_*.csv`).

## Registered hypotheses, verdicts by the registered criteria

| Hypothesis | Registered criterion | Result | Verdict |
|---|---|---|---|
| H5.1 | persistence within 10 % of the one-step models (at least three models on NGSIM and highD); refuted by a gap above 30 % | NGSIM: 1 of 23 within 10 %, 19 worse by more than 30 %, diffusion 38 % better (a gap above 30 % for 20 of 23); FollowNet-HighD, which stood for highD (D12), and raw highD (HD-2): none within 10 %, all 23 worse by more than 30 % | refuted when the criterion is read for the majority of the models (reading fixed after the results, REV3-4), in favour of persistence |
| H5.2 | event split inflates the networks by at least 15 % and the IDM by less than 5 %; refuted below 5 % for the networks | networks 1.1 % (NGSIM) and 1.8 % (OpenACC) on average; of the twelve single cases ten below 5 %, two at 5.4 % | refuted when the bound is read for the class mean (REV2-3) |
| H5.3 | at least 2 of the 3 best by trajectory RMSE leave the composite's top three; refuted when the top threes coincide | NGSIM, FollowNet-HighD and raw highD: 3 of 3 leave for the one-step and the whole-event error, in every bootstrap replicate at least two | confirmed |
| H5.4 | rankings by spacing and speed diverge, tau below 0.5; refuted above 0.8 | tau 0.83 to 0.87 on NGSIM and FollowNet-HighD; raw highD 0.85 [0.79, 0.88] over the whole event (0.89 at 10 s, 0.81 at 30 s) | refuted |

Only these four tests on the datasets of the registered round are confirmatory, for highD first on FollowNet-HighD
and then, under a contract written before the run (HD-1, HD-2), on raw highD; the other datasets, E1b, E3b to E3g,
E5, P2h and the certified entry were added after their results and are exploratory (decisions R8, R9, REV1-1 to
REV1-4, REV2-1 to REV2-4). The margin core of the certified entry (REV4-1) and, on raw highD, the spread over seeds
and folds and the run under the block split (HD-2, HD-5) are exploratory as well; each was specified before it was
run. The certified hybrid is the authors' own method, added after the other entries had been scored; its
unstable fraction is zero by construction.

## Composite leaderboard (top eight per dataset)

From `results/m4/e3_rankings.csv` (25 entries, sorted by the composite, ties by the
whole-event P1 error):

| NGSIM | composite | concave error* [m/s] | hysteresis error* | unstable fraction | collisions_100* |
|---|---|---|---|---|---|
| newell_median | 0.130 | 0.42 | 92 | 0.00 | 0 |
| newell | 0.172 | 0.42 | 98 | 0.00 | 0 |
| idm_median | 0.182 | 0.40 | 92 | 0.65 | 0 |
| gipps | 0.224 | 0.44 | 91 | 0.88 | 0 |
| ddpg | 0.266 | 0.47 | 105 | 0.42 | 0 |
| mlp_loop | 0.312 | 0.37 | 78 | 1.00 | 548 |
| idm | 0.339 | 0.55 | 111 | 0.00 | 0 |
| gru_loop | 0.344 | 0.45 | 87 | 0.74 | 603 |

| FollowNet-HighD | composite | concave error* [m/s] | hysteresis error* | unstable fraction | collisions_100* |
|---|---|---|---|---|---|
| residual_idm_certified | 0.115 | 0.66 | 358 | 0.00 | 0 |
| ddpg | 0.146 | 0.64 | 274 | 0.06 | 433 |
| idm | 0.177 | 0.75 | 332 | 0.64 | 0 |
| idm_median | 0.198 | 0.47 | 227 | 1.00 | 81 |
| newell | 0.312 | 0.74 | 640 | 0.00 | 713 |
| newell_median | 0.312 | 0.88 | 407 | 0.00 | 636 |
| gipps | 0.323 | 0.87 | 404 | 1.00 | 0 |
| gail | 0.385 | 0.86 | 297 | 1.00 | 816 |

`*` in-range profiles (leader speeds inside the training speed range). Composite rank
intervals (95 %, bootstrap over profiles, speeds and the three training seeds) from
`results/leaderboard/`, which ranks the same 25 entries): NGSIM newell_median 1 [1.0, 2.0], newell 2 [2.0, 3.5], idm_median 3 [1.0, 4.0], gipps 4 [3.0, 7.0], ddpg 5 [4.0, 10.0];
HighD residual_idm_certified 1 [1.0, 4.0], ddpg 2 [1.0, 5.0], idm 3 [1.0, 5.0]. First three on the
other datasets (25 entries, seed 0): OpenACC newell, idm_median, gipps; TGSIM newell,
gipps, ddpg; Waymo gipps, idm, residual_idm_certified; FollowNet-NGSIM gipps, perl, then
lstm_loop and idm_median tied. The full tables with intervals, the rank-stability plots
and the refused entries (none of the 660 run directories) are in `results/leaderboard/`.

## Caveats

* Whole-event closed-loop errors of the one-step networks on NGSIM are dominated by a few
  followers (finding 8); their intervals are wide (LSTM 81.6 m [10.2, 140.6]) and the
  bootstrap mean lies below the point value.
* OpenACC has 25 drivers: its E2 intervals are coarse and its whole-event numbers noisy.
* The synthetic rare-event suite and half of the AstaZero leader windows lie above the
  speed range of the congested NGSIM data; in-range views are reported next to the full
  ones and feed the composite.
* The empirical platoon reference of P2 consists of ACC-equipped followers (two manual
  AstaZero runs excepted). P2h has human followers, but its track windows last 60 to 185 s
  with 4 to 46 recorded followers: its 100-vehicle keys say how far a disturbance travels
  within the window rather than how it grows, and the growth exponent of a long human
  platoon is still not measured.
* The mechanistic models are calibrated on a 2,000-event sample with one objective; the
  networks use one training budget per family (decision M2-4). The tables are fold 0 and
  seed 0; seeds 1-2 and folds 1-4 of the driver split exist for every entry on NGSIM,
  FollowNet-HighD and raw highD (seed and fold spread of the leaderboard), the four other
  datasets have fold 0 and seed 0 only, and E1 reports five clean-arm folds for the eight-model subset.
* E5 tunes two networks on one dataset, one fold and one seed; its runs live in
  `results/sweeps/` and are never ranked.
* FollowNet-HighD events define spacing as `dhw` and have no driver ids (driver split =
  event split there); the NGSIM near-miss set is not used for scoring.
* Vehicles shared between training and evaluation. Under the driver split the test followers
  are new, but many test events are led by a vehicle that follows in a training event: 75 %
  (NGSIM), 80 % (OpenACC), 76 % (TGSIM), 27 % (Waymo), 50 % (raw highD); unknown for the two
  FollowNet sets, which carry no identities. The block split removes this on NGSIM, TGSIM,
  Waymo and raw highD, not on OpenACC (25 vehicles); only raw highD has a run under it (HD-5).
* The references of P2, P2h and P4 are cut from all recorded data, without regard to the
  split. P2: 19 leader windows with their recorded followers from OpenACC campaign files that
  also hold 34 % of the OpenACC training events (14 % overlap a window in time). P2h: in all
  40 NGSIM windows and 55 of the 56 TGSIM windows a follower of the chain also follows in a
  training event of that dataset (61 % and 66 % of the chain followers). P4: 54 % of the
  followers of the recorded TGSIM episodes follow in a TGSIM training event. Entries trained
  on OpenACC (P2 inputs of the composite), on NGSIM and TGSIM (human-referenced composites)
  and on TGSIM (recorded-episode collision rate) have therefore seen part of their reference;
  for entries trained elsewhere, and for the registered tests, these references are foreign.
* H5.4 was registered for the closed-loop protocol without a horizon. On raw highD the rule
  refutes at every horizon by the point estimate (0.89, 0.81, 0.85), while the lower bounds
  of the intervals at 30 s (0.75) and over the whole event (0.79) lie below 0.8.
* The contract of the run on raw highD was committed before any entry was calibrated,
  trained or scored on it, after a probe read of two recordings and an interrupted
  extraction (HD-6).

## Reproduction

```
python scripts/download_data.py follownet --yes  # FollowNet figshare files; the highD-derived events are rebuilt, never redistributed
python scripts/m1.py dataset=<name>            # events, splits, baselines, calibration
python scripts/build_human_profiles.py         # human platoon windows of p2h (data/scenarios/human_profiles.json)
python scripts/m2.py dataset=<name>            # reference models
python scripts/m4.py main|seeds|folds|paired|event|transfer|ablation
python scripts/m4.py main|seeds|folds|transfer models=residual_idm_certified   # the certified entry
python scripts/m4.py protocols datasets=openacc,waymo_hu,follownet_ngsim,tgsim  # every protocol of every entry there
python scripts/evaluate.py dataset=<name> model=<label> "protocols=[p1s,p2h]"   # the two newer protocols of one run
python scripts/sweep.py dataset=ngsim model=lstm n=16                          # E5 (and model=mlp)
python scripts/m3_report.py; python scripts/m4_report.py
python scripts/make_leaderboard.py; python scripts/make_report.py
```

## Comparison with the published FollowNet baselines

FollowNet (Chen et al., Scientific Data 2023) reports, on the same highD and NGSIM I-80
events that this benchmark re-uses, the spacing MSE and the collision rate of five
baselines on a random 70/15/15 event split, rolled over the whole 15 s event. Our rollouts
start after the 3 s warm-up and are scored on the test part of the driver split (on the
FollowNet events this is a random event split, since they carry no identities); the
per-event mean of the spacing MSE is the comparable quantity.

| model | highD, published | highD, ours | NGSIM I-80, published | NGSIM I-80, ours |
|---|---|---|---|---|
| IDM | 27.5 (0 ‰) | 16.0 (0 ‰) | 55.0 (0 ‰) | 35.2 (0 ‰) |
| Gipps / GHR | 23.8 (0 ‰, GHR) | 18.6 (0 ‰, Gipps) | 62.2 (0 ‰, GHR) | 33.6 (0 ‰, Gipps) |
| NN / MLP | 24.1 (39.9 ‰) | 18.6 (7.2 ‰) | 29.0 (3.5 ‰) | 87.6 (36.6 ‰) |
| LSTM | 22.9 (3.7 ‰) | 10.9 (0.8 ‰) | 23.9 (17.3 ‰) | 55.8 (2.6 ‰) |
| mlp_loop / lstm_loop | --- | 10.0 / 10.2 | --- | 29.3 / 31.0 |
| DDPG | 38.5 (0 ‰) | 134.2 (0.4 ‰) | 46.9 (0 ‰) | 187.1 (0 ‰) |

Orders of magnitude agree on highD; our calibrated IDM and our LSTM are better than the
published ones, our one-step MLP and LSTM are worse on the NGSIM I-80 events (383 test
events against their 1,930 events, and long events where one-step networks diverge), and
our DDPG keeps a larger spacing than the record because its reward (Zhu et al. 2020) does
not compare the simulated with the observed spacing, which the FollowNet DDPG does.
