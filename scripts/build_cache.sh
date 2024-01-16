export PYTHONPATH=.:$PYTHONPATH
find ./configs -type f -exec \
    python scripts/pipeline.py {} \
        --workspace none \
        --build_cache_only True \
        --device cuda:0 \;
