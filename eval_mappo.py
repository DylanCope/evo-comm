from jaxevocomm.train.mappo import MAPPO


experiment_dir = 'outputs/2024-03-28/13-07-31'
trainer, runner_state = MAPPO.restore(experiment_dir)
print('Successfully restored trainer and runner_state')

print('Running evaluations...')
_, metrics = trainer.rollout(runner_state, n_envs=128, n_steps=50)

print('Mean reward:', metrics['mean_total_reward'])

