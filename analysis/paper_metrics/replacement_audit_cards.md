# Replacement Audit Cards

These cards are generated from quality-gated same-family pairwise outputs. Percent fields are percentages over the shared item set unless otherwise stated.

## Thresholds

- High churn: >= 25.0%
- High regression mass: >= 8.0%
- High normalized regression burden: >= 10.0%
- Near parity: absolute accuracy delta <= 5.0%

## Cards

### Qwen3.5-2B -> Qwen2.5-3B-Instruct

- Status: `negative_delta_swap`
- Review priority: `exclude_or_reframe`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden; near_parity_churn; not_a_successful_replacement`
- Accuracy: 46.9% -> 44.2% (-2.7 points)
- Improvement/regression/churn: 14.4% / 17.1% / 31.5%
- Error persistence/correction rate: 72.9% / 27.1%
- Normalized regression burden: 32.2%
- Top improving categories: health:+0.086; other:+0.061; history:+0.051; engineering:+0.020; math:+0.000
- Top regressing categories: computer science:-0.132; philosophy:-0.103; psychology:-0.097; biology:-0.083; economics:-0.074

### Qwen3.5-0.8B -> Qwen3-4B-Instruct-2507

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 33.0% -> 48.2% (+15.2 points)
- Improvement/regression/churn: 25.0% / 9.8% / 34.8%
- Error persistence/correction rate: 62.7% / 37.3%
- Normalized regression burden: 14.6%
- Top improving categories: philosophy:+0.254; math:+0.245; other:+0.243; business:+0.218; physics:+0.182
- Top regressing categories: law:+0.011; biology:+0.022; engineering:+0.100; psychology:+0.107; health:+0.131

### Qwen2.5-0.5B-Instruct -> Qwen3.5-0.8B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 16.5% -> 33.0% (+16.6 points)
- Improvement/regression/churn: 25.4% / 8.8% / 34.2%
- Error persistence/correction rate: 69.6% / 30.4%
- Normalized regression burden: 10.5%
- Top improving categories: biology:+0.283; economics:+0.248; business:+0.229; psychology:+0.228; chemistry:+0.215
- Top regressing categories: history:+0.020; engineering:+0.056; law:+0.057; other:+0.061; philosophy:+0.087

### Qwen3-0.6B -> Qwen3.5-2B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_normalized_regression_burden`
- Accuracy: 28.0% -> 46.9% (+18.9 points)
- Improvement/regression/churn: 26.6% / 7.6% / 34.2%
- Error persistence/correction rate: 63.1% / 36.9%
- Normalized regression burden: 10.6%
- Top improving categories: biology:+0.378; economics:+0.317; psychology:+0.282; computer science:+0.245; philosophy:+0.238
- Top regressing categories: engineering:+0.028; history:+0.101; law:+0.129; math:+0.152; other:+0.161

### Qwen3-0.6B -> Qwen2.5-3B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 28.0% -> 44.2% (+16.2 points)
- Improvement/regression/churn: 24.7% / 8.5% / 33.2%
- Error persistence/correction rate: 65.7% / 34.3%
- Normalized regression burden: 11.8%
- Top improving categories: biology:+0.294; health:+0.258; economics:+0.243; other:+0.222; psychology:+0.184
- Top regressing categories: engineering:+0.048; law:+0.099; computer science:+0.113; chemistry:+0.126; philosophy:+0.135

### Qwen3.5-0.8B -> Qwen2.5-3B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 33.0% -> 44.2% (+11.2 points)
- Improvement/regression/churn: 21.9% / 10.7% / 32.6%
- Error persistence/correction rate: 67.3% / 32.7%
- Normalized regression burden: 16.0%
- Top improving categories: other:+0.191; health:+0.187; math:+0.170; business:+0.144; history:+0.141
- Top regressing categories: computer science:+0.009; law:+0.042; chemistry:+0.043; philosophy:+0.063; engineering:+0.076

### Qwen2.5-3B-Instruct -> Qwen3-4B-Instruct-2507

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden; near_parity_churn`
- Accuracy: 44.2% -> 48.2% (+4.0 points)
- Improvement/regression/churn: 18.3% / 14.2% / 32.5%
- Error persistence/correction rate: 67.3% / 32.7%
- Normalized regression burden: 25.5%
- Top improving categories: philosophy:+0.190; computer science:+0.132; chemistry:+0.109; math:+0.075; business:+0.074
- Top regressing categories: biology:-0.067; health:-0.056; law:-0.030; history:-0.010; psychology:+0.000

### Gemma-3-1B-it -> Gemma-2-2B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 17.5% -> 31.7% (+14.2 points)
- Improvement/regression/churn: 22.8% / 8.7% / 31.5%
- Error persistence/correction rate: 72.3% / 27.7%
- Normalized regression burden: 10.5%
- Top improving categories: biology:+0.217; economics:+0.198; chemistry:+0.185; physics:+0.185; business:+0.181
- Top regressing categories: history:+0.040; engineering:+0.048; other:+0.070; law:+0.076; philosophy:+0.087

### Qwen3.5-0.8B -> Qwen3.5-2B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 33.0% -> 46.9% (+13.9 points)
- Improvement/regression/churn: 22.2% / 8.3% / 30.5%
- Error persistence/correction rate: 66.9% / 33.1%
- Normalized regression burden: 12.4%
- Top improving categories: psychology:+0.204; economics:+0.203; biology:+0.172; math:+0.170; philosophy:+0.167
- Top regressing categories: engineering:+0.056; law:+0.072; history:+0.091; health:+0.101; chemistry:+0.116

### Qwen2.5-0.5B-Instruct -> Qwen3-0.6B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 16.5% -> 28.0% (+11.5 points)
- Improvement/regression/churn: 20.8% / 9.3% / 30.2%
- Error persistence/correction rate: 75.0% / 24.9%
- Normalized regression burden: 11.1%
- Top improving categories: business:+0.218; math:+0.206; physics:+0.182; psychology:+0.150; economics:+0.134
- Top regressing categories: law:+0.000; history:+0.010; philosophy:+0.016; other:+0.030; biology:+0.078

### Qwen3.5-2B -> Qwen3-4B-Instruct-2507

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden; near_parity_churn`
- Accuracy: 46.9% -> 48.2% (+1.3 points)
- Improvement/regression/churn: 15.2% / 13.9% / 29.1%
- Error persistence/correction rate: 71.3% / 28.7%
- Normalized regression burden: 26.2%
- Top improving categories: other:+0.113; philosophy:+0.087; math:+0.075; business:+0.059; engineering:+0.044
- Top regressing categories: biology:-0.150; psychology:-0.097; law:-0.061; economics:-0.040; computer science:+0.000

### Qwen3-0.6B -> Qwen3.5-0.8B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_churn; high_regression_mass; high_normalized_regression_burden`
- Accuracy: 28.0% -> 33.0% (+5.1 points)
- Improvement/regression/churn: 17.0% / 11.9% / 28.9%
- Error persistence/correction rate: 76.5% / 23.5%
- Normalized regression burden: 16.5%
- Top improving categories: biology:+0.206; economics:+0.114; computer science:+0.104; chemistry:+0.083; psychology:+0.078
- Top regressing categories: engineering:-0.028; math:-0.018; history:+0.010; business:+0.011; physics:+0.015

### Qwen2-0.5B-Instruct -> Qwen2.5-0.5B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_regression_mass; high_normalized_regression_burden; near_parity_churn`
- Accuracy: 13.0% -> 16.5% (+3.5 points)
- Improvement/regression/churn: 12.6% / 9.1% / 21.7%
- Error persistence/correction rate: 85.6% / 14.4%
- Normalized regression burden: 10.5%
- Top improving categories: business:+0.090; biology:+0.089; other:+0.087; math:+0.051; chemistry:+0.040
- Top regressing categories: health:-0.045; law:-0.030; philosophy:+0.024; psychology:+0.024; engineering:+0.028

### Gemma-3-270M-Instruct -> Gemma-3-1B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_required`
- Risk flags: `high_regression_mass; near_parity_churn`
- Accuracy: 10.4% -> 11.8% (+1.3 points)
- Improvement/regression/churn: 9.7% / 8.3% / 18.0%
- Error persistence/correction rate: 89.2% / 10.8%
- Normalized regression burden: 9.3%
- Top improving categories: economics:+0.064; biology:+0.061; psychology:+0.049; engineering:+0.044; philosophy:+0.016
- Top regressing categories: computer science:-0.038; history:-0.030; business:-0.027; chemistry:-0.003; math:-0.003

### Qwen2-0.5B-Instruct -> Qwen3-4B-Instruct-2507

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 13.0% -> 48.2% (+35.2 points)
- Improvement/regression/churn: 41.2% / 5.9% / 47.1%
- Error persistence/correction rate: 52.7% / 47.3%
- Normalized regression burden: 6.8%
- Top improving categories: business:+0.537; math:+0.484; economics:+0.441; physics:+0.414; chemistry:+0.407
- Top regressing categories: law:+0.038; history:+0.182; engineering:+0.185; health:+0.283; psychology:+0.359

### Qwen2-0.5B-Instruct -> Qwen3.5-2B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 13.0% -> 46.9% (+33.9 points)
- Improvement/regression/churn: 39.8% / 5.9% / 45.7%
- Error persistence/correction rate: 54.3% / 45.7%
- Normalized regression burden: 6.8%
- Top improving categories: biology:+0.544; economics:+0.480; business:+0.479; psychology:+0.456; math:+0.409
- Top regressing categories: law:+0.099; engineering:+0.141; history:+0.141; health:+0.253; philosophy:+0.278

### Qwen2.5-0.5B-Instruct -> Qwen3-4B-Instruct-2507

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 16.5% -> 48.2% (+31.8 points)
- Improvement/regression/churn: 38.3% / 6.5% / 44.9%
- Error persistence/correction rate: 54.1% / 45.9%
- Normalized regression burden: 7.8%
- Top improving categories: business:+0.447; math:+0.433; economics:+0.411; physics:+0.380; chemistry:+0.368
- Top regressing categories: law:+0.068; history:+0.152; engineering:+0.157; other:+0.304; biology:+0.306

### Qwen2-0.5B-Instruct -> Qwen2.5-3B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 13.0% -> 44.2% (+31.2 points)
- Improvement/regression/churn: 37.2% / 6.0% / 43.1%
- Error persistence/correction rate: 57.3% / 42.7%
- Normalized regression burden: 6.8%
- Top improving categories: business:+0.463; biology:+0.461; math:+0.409; economics:+0.406; psychology:+0.359
- Top regressing categories: law:+0.068; engineering:+0.161; philosophy:+0.175; history:+0.192; computer science:+0.245

### Qwen2.5-0.5B-Instruct -> Qwen3.5-2B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 16.5% -> 46.9% (+30.5 points)
- Improvement/regression/churn: 36.5% / 6.1% / 42.6%
- Error persistence/correction rate: 56.3% / 43.7%
- Normalized regression burden: 7.2%
- Top improving categories: biology:+0.456; economics:+0.450; psychology:+0.432; business:+0.388; physics:+0.361
- Top regressing categories: history:+0.111; engineering:+0.112; law:+0.129; other:+0.191; philosophy:+0.254

### Qwen2.5-0.5B-Instruct -> Qwen2.5-3B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 16.5% -> 44.2% (+27.8 points)
- Improvement/regression/churn: 34.0% / 6.3% / 40.3%
- Error persistence/correction rate: 59.3% / 40.7%
- Normalized regression burden: 7.5%
- Top improving categories: health:+0.384; economics:+0.376; business:+0.372; biology:+0.372; math:+0.358
- Top regressing categories: law:+0.099; engineering:+0.133; philosophy:+0.151; history:+0.162; computer science:+0.208

### Gemma-3-270M-Instruct -> Gemma-2-2B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 10.4% -> 31.7% (+21.2 points)
- Improvement/regression/churn: 28.4% / 7.1% / 35.5%
- Error persistence/correction rate: 68.3% / 31.7%
- Normalized regression burden: 8.0%
- Top improving categories: biology:+0.461; economics:+0.366; psychology:+0.350; chemistry:+0.222; physics:+0.219
- Top regressing categories: engineering:+0.080; history:+0.081; law:+0.118; philosophy:+0.127; computer science:+0.170

### Qwen3-0.6B -> Qwen3-4B-Instruct-2507

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 28.0% -> 48.2% (+20.2 points)
- Improvement/regression/churn: 27.4% / 7.1% / 34.5%
- Error persistence/correction rate: 62.0% / 38.0%
- Normalized regression burden: 9.9%
- Top improving categories: philosophy:+0.325; economics:+0.277; other:+0.274; computer science:+0.245; chemistry:+0.235
- Top regressing categories: law:+0.068; engineering:+0.072; history:+0.141; psychology:+0.184; physics:+0.198

### Qwen2-0.5B-Instruct -> Qwen3.5-0.8B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 13.0% -> 33.0% (+20.0 points)
- Improvement/regression/churn: 27.1% / 7.1% / 34.2%
- Error persistence/correction rate: 68.8% / 31.2%
- Normalized regression burden: 8.1%
- Top improving categories: biology:+0.372; business:+0.319; economics:+0.277; chemistry:+0.255; psychology:+0.252
- Top regressing categories: law:+0.027; history:+0.051; engineering:+0.084; philosophy:+0.111; other:+0.148

### Gemma-3-1B-Instruct -> Gemma-2-2B-Instruct

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 11.8% -> 31.7% (+19.9 points)
- Improvement/regression/churn: 26.9% / 6.9% / 33.8%
- Error persistence/correction rate: 69.6% / 30.4%
- Normalized regression burden: 7.9%
- Top improving categories: biology:+0.400; economics:+0.302; psychology:+0.301; chemistry:+0.225; business:+0.218
- Top regressing categories: engineering:+0.036; law:+0.110; history:+0.111; philosophy:+0.111; other:+0.165

### Qwen2-0.5B-Instruct -> Qwen3-0.6B

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_churn`
- Accuracy: 13.0% -> 28.0% (+15.0 points)
- Improvement/regression/churn: 22.6% / 7.6% / 30.2%
- Error persistence/correction rate: 74.1% / 25.9%
- Normalized regression burden: 8.7%
- Top improving categories: business:+0.309; math:+0.257; physics:+0.216; psychology:+0.175; chemistry:+0.172
- Top regressing categories: law:-0.030; philosophy:+0.040; history:+0.040; health:+0.081; engineering:+0.112

### Gemma-3-270M-Instruct -> Gemma-3-1B-it

- Status: `positive_delta_candidate`
- Review priority: `manual_review_recommended`
- Risk flags: `high_regression_mass`
- Accuracy: 10.4% -> 17.5% (+7.1 points)
- Improvement/regression/churn: 15.6% / 8.5% / 24.0%
- Error persistence/correction rate: 82.6% / 17.4%
- Normalized regression burden: 9.5%
- Top improving categories: biology:+0.244; psychology:+0.170; economics:+0.168; other:+0.109; computer science:+0.047
- Top regressing categories: business:+0.011; math:+0.030; engineering:+0.032; physics:+0.034; chemistry:+0.036

### Gemma-3-1B-Instruct -> Gemma-3-1B-it

- Status: `positive_delta_candidate`
- Review priority: `routine_monitoring`
- Risk flags: `no_default_flag`
- Accuracy: 11.8% -> 17.5% (+5.8 points)
- Improvement/regression/churn: 13.5% / 7.7% / 21.2%
- Error persistence/correction rate: 84.7% / 15.3%
- Normalized regression burden: 8.7%
- Top improving categories: biology:+0.183; psychology:+0.121; economics:+0.104; other:+0.096; computer science:+0.085
- Top regressing categories: engineering:-0.012; philosophy:+0.024; physics:+0.028; math:+0.033; law:+0.034
