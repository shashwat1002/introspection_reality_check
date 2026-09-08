# ICL Data (k=3, n=10)

This folder contains in-context learning (ICL) experiment data for two ROME-edited models.
All data is filtered to k=3 buckets and n=10 in-context examples per bucket.

## Top-level structure

- `gemma/` — data for the Gemma model
- `llama/` — data for the LLaMA model

## Path structure (identical for both models)

```
<model>/
└── <layer_range>/          # gemma: 46_60 | llama: 54_73
    └── seed_{0..4}/        # 5 random seeds for robustness
        └── k_mean_buckets/
            ├── bucket_assignments/
            │   └── all_settings_bucket_assignments_k3.csv
            ├── samples/
            │   ├── avg_dominance_diff_generation/
            │   │   └── all_settings_avg_dominance_diff_generation_k3_n10_examples.csv
            │   ├── avg_new_dominance_generation/
            │   │   └── all_settings_avg_new_dominance_generation_k3_n10_examples.csv
            │   └── avg_true_dominance_generation/
            │       └── all_settings_avg_true_dominance_generation_k3_n10_examples.csv
            └── generations/
                ├── all_settings_avg_dominance_diff_generation_k3_n10_examples__avg_dominance_diff_generation_bucket_k3.csv
                ├── all_settings_avg_new_dominance_generation_k3_n10_examples__avg_new_dominance_generation_bucket_k3.csv
                └── all_settings_avg_true_dominance_generation_k3_n10_examples__avg_true_dominance_generation_bucket_k3.csv
```

## Folder descriptions

### `bucket_assignments/`
Contains `all_settings_bucket_assignments_k3.csv`: assigns every case to one of 3 k-means buckets based on each score column.

### `samples/<score_column>/`
The in-context examples sampled from each bucket for a given score column. Each file contains 10 examples per bucket (n=10), used as ICL demonstrations.

Score columns:
- `avg_dominance_diff_generation` — difference between new and true answer dominance
- `avg_new_dominance_generation` — dominance of the edited (new) answer
- `avg_true_dominance_generation` — dominance of the original (true) answer

### `generations/`
Model generation results when prompted with the sampled ICL examples. Each file corresponds to one (score column, k, n) configuration and records the model's predicted bucket label alongside the ground-truth bucket label, enabling accuracy evaluation.