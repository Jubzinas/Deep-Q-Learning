# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import sys
import random
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import ale_py
from torch.utils.tensorboard import SummaryWriter
import buffer
from dqn_eval import evaluate
import helper

gym.register_envs(ale_py)

from atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def smooth(y, w=10):
    y = np.array(y)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

def smoothed_steps(steps, w=10):
    steps = np.array(steps)
    if len(steps) < w:
        return steps
    return steps[w-1:]

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] #name of the experiment
    seed: int = 1 #seed of the experiment
    torch_deterministic: bool = True #if toggled, `torch.backends.cudnn.deterministic=False`
    cuda: bool = True #if toggled, cuda will be enabled by default
    track: bool = False #if toggled, this experiment will be tracked with Weights and Biases
    wandb_project_name: str = "cleanRL" #he wandb's project name
    wandb_entity: str = None #the entity (team) of wandb's project
    capture_video: bool = True #whether to capture videos of the agent performances (check out `videos` folder)
    save_model: bool = True #whether to save model into the `runs/{run_name}` folder
    upload_model: bool = False #whether to upload the saved model to huggingface
    hf_entity: str = "" #the user or org name of the model repository from the Hugging Face Hub
    log_frequency: int = 500 

    # Algorithm specific arguments
    env_id: str = "ALE/Breakout-v5" #the id of the environment
    total_timesteps: int = 1500000 #total timesteps of the experiments
    learning_rate: float = 1e-4 #the learning rate of the optimizer
    num_envs: int = 1 #the number of parallel game environments
    buffer_size: int = 300000 #the replay memory buffer size
    gamma: float = 0.99 #the discount factor gamma
    tau: float = 1.0 #the target network update rate
    target_network_frequency: int = 1000 #the timesteps it takes to update the target network
    batch_size: int = 32 #the batch size of sample from the reply memory
    start_e: float = 1 #the starting epsilon for exploration
    end_e: float = 0.01 #the ending epsilon for exploration
    exploration_fraction: float = 0.10 #the fraction of `total-timesteps` it takes from start-e to go end-e
    learning_starts: int = 80000 #timestep to start learning
    train_frequency: int = 4 #the frequency of training


def make_env(env_id, seed, idx, capture_video, run_name, eval_episodes=None, is_eval=False, video_name_prefix = None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            if eval_episodes is not None:
                mid, last = eval_episodes // 2, eval_episodes - 1
                trigger_set = {0, mid, last}
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda ep: ep in trigger_set, disable_logger=True, name_prefix=video_name_prefix)
            else:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda ep: ep == 0, disable_logger=True)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=1 if is_eval else 30)
        env = MaxAndSkipEnv(env, skip=4)
        if not is_eval:
            env = EpisodicLifeEnv(env)
            env = ClipRewardEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    def save_and_eval(label: str):
        ckpt_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model_{label}"
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        torch.save(q_network.state_dict(), ckpt_path)
        print(f"\n--- Evaluation: {label} ---")
        returns = evaluate(
            ckpt_path,
            make_env,
            args.env_id,
            eval_episodes=5,
            run_name=f"{run_name}",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
            capture_video=args.capture_video,
            video_name_prefix = label
        )
        avg = np.mean(returns)
        print(f"[{label}] avg={avg:.2f} | returns={[round(r,1) for r in returns]}")
        print("eval/avg_return", avg, global_step)
        return returns
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else 
                          "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print('using ', device, ' device')

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.total_timesteps, video_name_prefix = "before") for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = buffer.ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    mid_step = args.total_timesteps // 2
    eval_mid = False
    loss_history, q_history, step_history = [], [], []
    pbar = tqdm(range(args.total_timesteps), desc="Training", unit="step")
    for global_step in pbar:
        if global_step == 0:
           save_and_eval("before")
        if global_step >= mid_step and not eval_mid:
            eval_mid = True
            save_and_eval("mid")

        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    print("charts/episodic_return", info["episode"]["r"], global_step)
                    print("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % args.log_frequency == 0:
                    elapsed = time.time() - start_time
                    sps = int(global_step/elapsed)
                    eta_seconds = int((args.total_timesteps - global_step) / sps) if sps > 0 else 0
                    eta_min, eta_sec = divmod(eta_seconds, 60)
                    progress = 100 * global_step / args.total_timesteps
                    loss_val = loss.item()
                    q_val = old_val.mean().item()
                    loss_history.append(loss_val)
                    q_history.append(q_val)
                    step_history.append(global_step)
                    print(
                        f"[{progress:5.1f}%] step={global_step:>7} | "
                        f"loss={loss.item():.4f} | "
                        f"q_mean={old_val.mean().item():.3f} | "
                        f"eps={epsilon:.3f} | "
                        f"SPS={sps} | "
                        f"ETA={eta_min}m{eta_sec:02d}s"
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Training Curves — {args.env_id} ({args.total_timesteps:,} steps)", fontsize=13)

    # smooth helper
    def smooth(y, w=50):
        if len(y) < w:
            return y
        return np.convolve(y, np.ones(w)/w, mode='valid')

    #s = step_history[49:]  # align after smoothing
    #w = 10

    helper.plot_training_curves(loss_history, q_history, step_history, run_name, args.env_id, args.total_timesteps)
    if args.save_model:
        save_and_eval("end")
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from dqn_eval import evaluate

    envs.close()