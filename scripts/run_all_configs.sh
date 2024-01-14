SCRIPT=./scripts/run_pipeline_with_n_seeds.sh
# Run all nn configs for 8 seeds
for folder in configs/ablation configs/baselines/nn_models; do
    find $folder -type f -exec $SCRIPT {} 8 \;
done

# Run all sklearn baselines for 1 seed
find configs/baselines/sklearn -type f -exec $SCRIPT {} 1 \;
