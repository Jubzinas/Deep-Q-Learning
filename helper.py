import numpy as np
import matplotlib.pyplot as plt
import os


def smooth(y, w=50):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

def smoothed_steps(steps, w=10):
    steps = np.array(steps)
    if len(steps) < w:
        return steps
    return steps[w-1:]

def plot_training_curves(loss_history, q_history, step_history, run_name, env_id, total_timesteps, w=10):
    """
    Plot and save training curves for loss and Q-value mean.

    :param loss_history:     list of loss values logged during training
    :param q_history:        list of mean Q-values logged during training
    :param step_history:     list of global steps corresponding to each log entry
    :param run_name:         run identifier used for the save path
    :param env_id:           environment name shown in the plot title
    :param total_timesteps:  total training steps shown in the plot title
    :param w:                smoothing window size (default 10)
    """
    if len(loss_history) == 0:
        print("No training data to plot — skipping training curves.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Training Curves — {env_id} ({total_timesteps:,} steps)", fontsize=13)

    # loss
    ax1.plot(step_history, loss_history, alpha=0.2, color='steelblue')
    if len(loss_history) >= w:
        ax1.plot(smoothed_steps(step_history, w), smooth(loss_history, w),
                 color='steelblue', linewidth=2, label=f'TD Loss (smoothed w={w})')
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-value mean
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