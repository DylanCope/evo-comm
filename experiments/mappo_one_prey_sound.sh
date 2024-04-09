python train.py -cn mce_mappo --multirun \
    N_AGENT_SOUNDS=8 \
    TOTAL_TIMESTEPS=5000000 \
    PREY_AUDIBLE_RANGE=2 \
    N_PREY_SOUNDS=1 \
    SEED=0,1,2,3,4,5,6,7,8,9 \
    N_OVERLAPPING_SOUNDS=0,1