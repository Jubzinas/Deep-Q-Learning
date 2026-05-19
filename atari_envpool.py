import os
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
import buffer

from dqn_eval import evaluate
from atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

gym.register_envs(ale_py)

import sys
if sys.platform == "darwin":
    raise SystemError("atari_envpool.py is for Linux/EC2 only. Use atari.py on Mac.")

import envpool

def smooth(y, w=10):
    y = np.array(y)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


def smoothed_steps(steps, w=10):
    steps = np.array(steps)
    if len(steps) < w:
        return steps
    return steps[w - 1:]


def plot_training_curves(loss_history, q_history, step_history, run_name, env_id, total_timesteps, w=10):
    if len(loss_history) == 0:
        print("No training data to plot — skipping.")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Training Curves — {env_id} ({total_timesteps:,} steps)", fontsize=13)
    ax1.plot(step_history, loss_history, alpha=0.2, color='steelblue')
    if len(loss_history) >= w:
        ax1.plot(smoothed_steps(step_history, w), smooth(loss_history, w),
                 color='steelblue', linewidth=2, label=f'TD Loss (smoothed w={w})')
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(step_history, q_history, alpha=0.2, color='coral')
    if len(q_history) >= w:
        ax2.plot(smoothed_steps(step_history, w), smooth(q_history, w),
                 color='coral', linewidth=2, label=f'Q-value mean (smoothed w={w})')
    ax2.set_ylabel("Q-value")
    ax2.set_xlabel("Step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plot_path = f"runs/{run_name}/training_curves.png"
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True
    save_model: bool = True
    log_frequency: int = 500

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"              # EnvPool uses this format (no ALE/ prefix)
    total_timesteps: int = 10000000
    learning_rate: float = 1e-4
    num_envs: int = 8                        # EnvPool handles many envs efficiently
    buffer_size: int = 200000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 256
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4


# ── EnvPool wrapper to make it compatible with the rest of the code ──────────
class EnvPoolWrapper:
    """
    Thin wrapper around an EnvPool environment to expose the same interface
    as gym.vector.SyncVectorEnv so the training loop doesn't need to change.
    """
    def __init__(self, env):
        self.env = env
        self.num_envs = env.config["num_envs"]
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

    def reset(self, seed=None):
        return self.env.reset(), {}

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return obs, rewards, terms, truncs, infos

    def close(self):
        self.env.close()


def make_envpool(env_id, num_envs, seed, episodic_life=True, reward_clip=True):
    """
    Create an EnvPool environment with the same preprocessing as the
    original make_env wrappers (noop reset, frame skip, max pooling,
    grayscale, resize 84x84, frame stack x4).
    """
    env = envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        seed=seed,
        episodic_life=episodic_life,    # EpisodicLifeEnv
        reward_clip=reward_clip,         # ClipRewardEnv
        stack_num=4,                     # FrameStackObservation
        img_height=84,                   # ResizeObservation
        img_width=84,                    # ResizeObservation
        gray_scale=True,                 # GrayscaleObservation
        noop_max=30,                     # NoopResetEnv
        frame_skip=4,                    # MaxAndSkipEnv
    )
    return EnvPoolWrapper(env)


def make_eval_envpool(env_id, num_envs, seed):
    """
    Eval version — no episodic life, no reward clipping, noop_max=1.
    """
    env = envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        seed=seed,
        episodic_life=False,    # full game episodes for eval
        reward_clip=False,      # real scores for eval
        stack_num=4,
        img_height=84,
        img_width=84,
        gray_scale=True,
        noop_max=1,             # deterministic start for eval
        frame_skip=4,
    )
    return EnvPoolWrapper(env)


# ── Q-Network (unchanged) ─────────────────────────────────────────────────────
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


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# ── make_env for eval + video (original Gymnasium stack) ─────────────────────
def make_env(env_id, seed, idx, capture_video, run_name, eval_episodes=None, is_eval=False, video_name_prefix=None):
    """
    Original Gymnasium-based make_env used exclusively for eval + video recording.
    Training uses EnvPool instead.
    """
    def thunk():
        gym_env_id = f"ALE/{env_id.replace('-v5', '')}-v5" if not env_id.startswith("ALE/") else env_id
        if capture_video and idx == 0:
            env = gym.make(gym_env_id, render_mode="rgb_array")
            if eval_episodes is not None:
                mid = eval_episodes // 2
                last = eval_episodes - 1
                trigger_set = {0, mid, last}
                video_folder = f"videos/{run_name}/{video_name_prefix}" if video_name_prefix else f"videos/{run_name}"
                os.makedirs(video_folder, exist_ok=True)
                env = gym.wrappers.RecordVideo(env, video_folder,
                                               episode_trigger=lambda ep: ep in trigger_set,
                                               disable_logger=True,
                                               name_prefix=video_name_prefix)
            else:
                video_folder = f"videos/{run_name}"
                os.makedirs(video_folder, exist_ok=True)
                env = gym.wrappers.RecordVideo(env, video_folder,
                                               episode_trigger=lambda ep: ep == 0,
                                               disable_logger=True)
        else:
            env = gym.make(gym_env_id)

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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    print(f"run name: {run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}")

    # ── Environment ──────────────────────────────────────────────────────────
    envs = make_envpool(args.env_id, args.num_envs, args.seed)
    print(f"obs space: {envs.single_observation_space}")
    print(f"action space: {envs.single_action_space}")

    # ── Networks ─────────────────────────────────────────────────────────────
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # ── Replay Buffer ─────────────────────────────────────────────────────────
    rb = buffer.ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    def save_and_eval(label):
        """
        Save checkpoint and evaluate using the original Gymnasium stack
        so RecordVideo works. Training uses EnvPool; eval uses make_env.
        """
        ckpt_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model_{label}"
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        torch.save(q_network.state_dict(), ckpt_path)
        print(f"\n--- Evaluation: {label} ---")
        returns = evaluate(
            ckpt_path,
            make_env,
            f"ALE/{args.env_id.replace('-v5', '')}-v5",   # convert back to ALE/ format
            eval_episodes=5,
            run_name=run_name,
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
            capture_video=args.capture_video,
            video_name_prefix=label,
        )
        avg = np.mean(returns)
        print(f"[{label}] avg={avg:.2f} | returns={[round(r, 1) for r in returns]}")
        return returns

    # ── Training loop ─────────────────────────────────────────────────────────
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

        epsilon = linear_schedule(args.start_e, args.end_e,
                                  args.exploration_fraction * args.total_timesteps,
                                  global_step)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # log episode returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # handle final observation for truncated episodes
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # training
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
                    sps = int(global_step / elapsed)
                    eta_seconds = int((args.total_timesteps - global_step) / sps) if sps > 0 else 0
                    eta_min, eta_sec = divmod(eta_seconds, 60)
                    progress = 100 * global_step / args.total_timesteps
                    loss_history.append(loss.item())
                    q_history.append(old_val.mean().item())
                    step_history.append(global_step)
                    print(
                        f"[{progress:5.1f}%] step={global_step:>7} | "
                        f"loss={loss.item():.4f} | "
                        f"q_mean={old_val.mean().item():.3f} | "
                        f"eps={epsilon:.3f} | "
                        f"SPS={sps} | "
                        f"ETA={eta_min}m{eta_sec:02d}s"
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for tp, qp in zip(target_network.parameters(), q_network.parameters()):
                    tp.data.copy_(args.tau * qp.data + (1.0 - args.tau) * tp.data)

    # ── End of training ───────────────────────────────────────────────────────
    plot_training_curves(loss_history, q_history, step_history,
                         run_name, args.env_id, args.total_timesteps)

    if args.save_model:
        save_and_eval("end")
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()