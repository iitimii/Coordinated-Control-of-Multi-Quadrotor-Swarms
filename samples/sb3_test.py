import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback


env_id = "CartPole-v1"
env = make_vec_env(env_id, n_envs=2)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1, 
    n_steps=2048,
    batch_size=64,
    n_epochs=10, 
)

callback_no_improvement = StopTrainingOnNoModelImprovement(10, 90, verbose=1)

eval_callback = EvalCallback(
    eval_env=env,
    callback_after_eval=callback_no_improvement,
    eval_freq=1_000
)

time_steps = 100_000
model.learn(time_steps, callback=eval_callback, progress_bar=True)
model.save("ppo_cartpole")




env = gym.make(env_id, render_mode='human')
for i in range(5):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()