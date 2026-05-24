import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import os


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
    video_name_prefix: str = None,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, eval_episodes=1, is_eval=True, video_name_prefix=video_name_prefix)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    current_return = 0.0
    fire_idx = envs.unwrapped.envs[0].unwrapped.get_action_meanings().index('FIRE')

    # take a NOOP step to get initial lives count
    obs, _, _, _, infos = envs.step(np.array([0]))
    prev_lives = infos.get('lives', np.array([3]))[0]

    print(f'run name: {run_name}')

    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        current_return += float(rewards[0])
        lives = infos.get('lives', np.array([3]))[0]

        # life lost mid-game → press FIRE immediately to resume
        if lives < prev_lives and lives > 0:
            envs.step(np.array([fire_idx]))

        prev_lives = lives

        if lives == 0:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={current_return}")
            episodic_returns.append(current_return)
            current_return = 0.0
            obs, _ = envs.reset()
            # step once to get fresh lives count after reset
            obs, _, _, _, infos = envs.step(np.array([0]))
            prev_lives = infos.get('lives', np.array([3]))[0]
        else:
            obs = next_obs

    envs.close()
    return episodic_returns

def save_and_eval(label: str, eval_episodes: int, exp_name: str, run_name: str, q_network, make_env, device, env_id: str, Model: torch.nn.Module, end_e: float, capture_video: bool, global_step):
        ckpt_path = f"runs/{run_name}/{exp_name}.cleanrl_model_{label}"
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        torch.save(q_network.state_dict(), ckpt_path)
        print(f"\n--- Evaluation: {label} ---")
        returns = evaluate(
            ckpt_path,
            make_env,
            env_id,
            eval_episodes= eval_episodes,
            run_name=f"{run_name}",
            Model=Model,
            device=device,
            epsilon=end_e,
            capture_video=capture_video,
            video_name_prefix = label
        )
        avg = np.mean(returns)
        print(f"[{label}] avg={avg:.2f} | returns={[round(r,1) for r in returns]}", " | eval/avg_return", avg, global_step)
        return returns