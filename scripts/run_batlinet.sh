SCRIPT=./scripts/run_pipeline_with_n_seeds.sh
# Run all nn configs for 8 seeds
for folder in configs/ablation/diff_branch/batlinet; do
    find $folder -type f -exec $SCRIPT {} 8 \;
done