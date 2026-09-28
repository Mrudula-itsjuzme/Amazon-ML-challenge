# Frozen model: five-S1 test inference

Actual fresh inference using the cached frozen 50/50 IDF + numeric LightGBM blend at threshold 0.55. Reused the existing label-free candidate retrieval over all 9,969,589 test S2/S3 targets.

- S1 entities: 5
- K100 candidate pairs: 500
- Predicted links: 9
- Predictions reproduce the prior frozen-model smoke output byte for byte.

The 0.9516 macro F0.5 was measured on a separate held-out 500-S1 confirmation cohort. These five test entities have no available ground truth, so their accuracy is unknown. This is a partial demonstration, not a full submission.
