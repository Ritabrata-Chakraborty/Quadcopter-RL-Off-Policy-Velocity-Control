"""Distributed training loop (TD3 / DDPG / SAC) for quadcopter point navigation.

Launches Ray workers to collect episodes in parallel, trains actor and critic
networks centrally, and logs to TensorBoard / wandb.
"""

import gc
import os
import pickle
from typing import Optional

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

import parameter as P
from model import Actor, Critic, DDPGCritic, MultiCritic, MultiDDPGCritic, SACActor
from parameter import (
    ACTION_SIZE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    SAVE_BUFFER_EVERY,
    EXPERIMENT_DIR,
    EXPERIMENT_NAME,
    EXPERIMENT_TYPE,
    GAMMA,
    HIDDEN_SIZE,
    LOAD_MODEL,
    LR,
    MINIMUM_BUFFER_SIZE,
    NUM_META_AGENT,
    POLICY_UPDATE_FREQUENCY,
    REPLAY_SIZE,
    STATE_SIZE,
    SUMMARY_WINDOW,
    TAU,
    TENSORBOARD_DIR,
    TRAIN_PLOTS_DIR,
    USE_GPU,
    USE_GPU_GLOBAL,
    USE_MULTI_CRITIC,
    WANDB_ENABLED,
    WANDB_ENTITY,
    WANDB_PROJECT,
    get_config_dict,
)
from runner import RLRunner
from utils import PrioritizedReplayBuffer, ReplayBuffer, save_navigation_episode_outputs, soft_update


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_weights(actor: nn.Module, device: torch.device, local_device: torch.device) -> list:
    """Transfer actor weights to worker device."""
    if device != local_device:
        weights = actor.to(local_device).state_dict()
        actor.to(device)
    else:
        weights = actor.state_dict()
    return [weights]


def save_checkpoint(
    actor: nn.Module,
    critic: nn.Module,
    actor_target: Optional[nn.Module],
    critic_target: nn.Module,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    episode: int,
    total_steps: int,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
) -> None:
    print(f'Saving checkpoint (episode {episode})')
    checkpoint = {
        "actor_model": actor.state_dict(),
        "critic_model": critic.state_dict(),
        "actor_target_model": actor_target.state_dict() if actor_target is not None else None,
        "critic_target_model": critic_target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "episode": episode,
        "total_steps": total_steps,
        "experiment_type": EXPERIMENT_TYPE,
        "multi_critic": USE_MULTI_CRITIC,
    }
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if alpha_optimizer is not None:
        checkpoint["alpha_optimizer"] = alpha_optimizer.state_dict()
    named_path = os.path.join(CHECKPOINT_DIR, f'{episode}.pth')
    torch.save(checkpoint, named_path)
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'latest.pth'))


def log_metrics(
    training_data: list,
    curr_episode: int,
    total_steps: int,
    buffer_size: int,
    writer: SummaryWriter,
    wandb_run: Optional[object] = None,
) -> None:
    """Log averaged metrics to TensorBoard and wandb."""
    td = np.array(training_data)
    means = list(np.nanmean(td, axis=0))

    if USE_MULTI_CRITIC:
        critic_loss = means[0]
        actor_loss = means[1]
        num_critics = P.NUM_CRITICS
        individual_losses = means[2:2+num_critics]
        perf_start_idx = 2 + num_critics
        travel_dist, success_rate, total_reward, goal_distance, crash_rate, timeout_rate = \
            means[perf_start_idx:perf_start_idx+6]
    else:
        critic_loss, actor_loss, travel_dist, success_rate, total_reward, goal_distance, \
            crash_rate, timeout_rate = means

    writer.add_scalar('Losses/Critic Loss', critic_loss, curr_episode)
    writer.add_scalar('Losses/Actor Loss', actor_loss, curr_episode)

    if USE_MULTI_CRITIC:
        for i, loss in enumerate(individual_losses):
            writer.add_scalar(f'Losses/Critic_{i}_Loss', loss, curr_episode)

    writer.add_scalar('Perf/Reward', total_reward, curr_episode)
    writer.add_scalar('Perf/Travel Distance', travel_dist, curr_episode)
    writer.add_scalar('Perf/Success Rate', success_rate, curr_episode)
    writer.add_scalar('Perf/Crash Rate', crash_rate, curr_episode)
    writer.add_scalar('Perf/Timeout Rate', timeout_rate, curr_episode)
    writer.add_scalar('Perf/Goal Distance', goal_distance, curr_episode)
    writer.add_scalar('Perf/Buffer Size', buffer_size, curr_episode)

    if wandb_run:
        import wandb
        wandb_dict = {
            'Episode': curr_episode,
            'Buffer Size': buffer_size,
            'Critic Loss': critic_loss,
            'Actor Loss': actor_loss,
            'Reward': total_reward,
            'Success Rate': success_rate,
            'Crash Rate': crash_rate,
            'Timeout Rate': timeout_rate,
            'Travel Distance': travel_dist,
            'Goal Distance': goal_distance,
        }

        if USE_MULTI_CRITIC:
            for i, loss in enumerate(individual_losses):
                wandb_dict[f'Critic_{i}_Loss'] = loss

        wandb.log(wandb_dict, step=curr_episode)


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def main() -> None:
    # Experiment directory setup
    train_gifs_dir = os.path.join(TRAIN_PLOTS_DIR, 'gifs')
    for d in [CHECKPOINT_DIR, TENSORBOARD_DIR, TRAIN_PLOTS_DIR, train_gifs_dir]:
        os.makedirs(d, exist_ok=True)

    config_dict = get_config_dict()
    with open(os.path.join(EXPERIMENT_DIR, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

    # Wandb
    wandb_run = None
    if WANDB_ENABLED:
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=EXPERIMENT_NAME,
                config=config_dict,
                reinit=True,
            )
            print(f"wandb run: {wandb_run.url}")
        except Exception as e:
            print(f"wandb init failed ({e}), continuing without wandb")
            wandb_run = None

    # Ray / TensorBoard
    ray.init()
    print(f"Welcome to Quadcopter Point Navigation {EXPERIMENT_TYPE}!")
    writer = SummaryWriter(TENSORBOARD_DIR)

    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # ---- Network instantiation (algorithm-specific) ----
    actor_target = None
    log_alpha = None
    alpha_optimizer = None

    if EXPERIMENT_TYPE == 'TD3':
        actor = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        actor_target = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        CriticCls = MultiCritic if USE_MULTI_CRITIC else Critic
        critic = CriticCls(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        critic_target = CriticCls(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)

    elif EXPERIMENT_TYPE == 'DDPG':
        actor = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        actor_target = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        CriticCls = MultiDDPGCritic if USE_MULTI_CRITIC else DDPGCritic
        critic = CriticCls(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        critic_target = CriticCls(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)

    elif EXPERIMENT_TYPE == 'SAC':
        actor = SACActor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        CriticCls = MultiCritic if USE_MULTI_CRITIC else Critic
        critic = CriticCls(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        critic_target = CriticCls(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
        log_alpha = torch.tensor(
            [np.log(P.SAC_ALPHA_INIT)], dtype=torch.float32,
            requires_grad=True, device=device,
        )
        alpha_optimizer = optim.AdamW([log_alpha], lr=P.SAC_ALPHA_LR)

    else:
        raise ValueError(f"Unknown EXPERIMENT_TYPE: {EXPERIMENT_TYPE}")

    if actor_target is not None:
        actor_target.load_state_dict(actor.state_dict())
        actor_target.eval()
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    actor_optimizer = optim.AdamW(actor.parameters(), lr=LR)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss(reduction='none')

    _reward_dim = P.NUM_CRITICS if USE_MULTI_CRITIC else 1
    if P.USE_PER:
        replay_buffer = PrioritizedReplayBuffer(
            REPLAY_SIZE, P.PER_ALPHA, P.PER_BETA_START, P.PER_BETA_FRAMES, P.PER_EPSILON,
            reward_dim=_reward_dim,
        )
    else:
        replay_buffer = ReplayBuffer(REPLAY_SIZE, reward_dim=_reward_dim)

    curr_episode = 0
    total_steps = 0
    last_actor_loss = 0.0
    training_active = False
    policy_start_episode = 0
    individual_critic_losses = []  # Track individual critic losses for multi-critic logging

    # Resume from checkpoint
    if LOAD_MODEL:
        ckpt = os.path.join(CHECKPOINT_DIR, 'latest.pth')
        print(f'Loading model from {ckpt}...')
        checkpoint = torch.load(ckpt, map_location=device)
        if checkpoint.get('multi_critic', False) != USE_MULTI_CRITIC:
            raise ValueError(
                f"Checkpoint has multi_critic={checkpoint.get('multi_critic', False)} "
                f"but USE_MULTI_CRITIC={USE_MULTI_CRITIC}. Use a matching checkpoint."
            )
        actor.load_state_dict(checkpoint['actor_model'])
        critic.load_state_dict(checkpoint['critic_model'])
        if actor_target is not None and checkpoint.get('actor_target_model') is not None:
            actor_target.load_state_dict(checkpoint['actor_target_model'])
        critic_target.load_state_dict(checkpoint['critic_target_model'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if EXPERIMENT_TYPE == 'SAC' and 'log_alpha' in checkpoint:
            log_alpha.data.copy_(checkpoint['log_alpha'].to(device))
            alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        curr_episode = checkpoint['episode']
        total_steps = checkpoint.get('total_steps', 0)
        print(f"Resumed from episode {curr_episode}")

        # Load buffer if it exists
        buffer_path = os.path.join(CHECKPOINT_DIR, 'buffer_latest.pkl')
        if os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                replay_buffer = pickle.load(f)
            print(f"Loaded replay buffer ({replay_buffer.get_length()} transitions)")
        else:
            print(f"Buffer not found at {buffer_path}; starting with fresh buffer")

    # Launch Ray workers
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    weights_set = get_weights(actor, device, local_device)

    job_list = []
    for meta_agent in meta_agents:
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode, training_active, 0))

    metric_names = ['travel_dist', 'success_rate', 'total_reward', 'goal_distance', 'crash_rate', 'timeout_rate']
    perf_metrics = {n: [] for n in metric_names}
    training_data = []

    try:
        while True:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                job_results, metrics, info, viz_data = job
                for transition in job_results:
                    replay_buffer.add_sample(*transition)
                del job_results
                for n in metric_names:
                    if n in metrics:
                        perf_metrics[n].append(metrics[n])

                if viz_data is not None and 'goal_pos' in metrics and 'start_pos' in metrics:
                    save_navigation_episode_outputs(
                        viz_data, metrics, info['episode_number'],
                        TRAIN_PLOTS_DIR, train_gifs_dir,
                        experiment_name=EXPERIMENT_NAME,
                    )
                del viz_data

                # Dispatch replacement job for this worker
                curr_episode += 1
                use_policy = replay_buffer.get_length() >= MINIMUM_BUFFER_SIZE
                if use_policy and not training_active:
                    training_active = True
                    policy_start_episode = curr_episode
                policy_episode = max(0, curr_episode - policy_start_episode) if training_active else 0
                job_list.append(meta_agents[info['id']].job.remote(
                    weights_set, curr_episode, use_policy, policy_episode))
            del done_jobs

            # Training
            if replay_buffer.get_length() >= MINIMUM_BUFFER_SIZE:
                critic_loss_val = 0.0

                for _ in range(P.GRADIENT_STEPS_PER_EPISODE):
                    total_steps += 1

                    if P.USE_PER:
                        states, actions, rewards, next_states, dones, per_indices, is_weights = \
                            replay_buffer.sample(BATCH_SIZE)
                        w = torch.from_numpy(is_weights).unsqueeze(1).to(device)
                    else:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                        w = None

                    s = torch.from_numpy(states).to(device)
                    a = torch.from_numpy(actions).to(device)
                    r = torch.from_numpy(rewards).to(device)
                    ns = torch.from_numpy(next_states).to(device)
                    d = torch.from_numpy(dones).to(device)
                    del states, actions, rewards, next_states, dones

                    if EXPERIMENT_TYPE == 'TD3':
                        # Critic update with target policy smoothing
                        with torch.no_grad():
                            noise = (torch.randn_like(a) * P.POLICY_NOISE).clamp(
                                -P.POLICY_NOISE_CLIP, P.POLICY_NOISE_CLIP)
                            next_action = (actor_target(ns) + noise).clamp(-1.0, 1.0)
                            if USE_MULTI_CRITIC:
                                target_qs = critic_target(ns, next_action)
                                q_targets = [
                                    r[:, i:i+1] + (1.0 - d) * GAMMA * torch.min(q1_i, q2_i)
                                    for i, (q1_i, q2_i) in enumerate(target_qs)
                                ]
                            else:
                                q1_next, q2_next = critic_target(ns, next_action)
                                q_target = r + (1.0 - d) * GAMMA * torch.min(q1_next, q2_next)

                        if USE_MULTI_CRITIC:
                            critic_qs = critic(s, a)
                            all_td1, all_td2 = [], []
                            critic_loss = torch.tensor(0.0, device=device)
                            individual_losses_step = []  # Track individual critic losses
                            for i, (q1_i, q2_i) in enumerate(critic_qs):
                                td1_i = loss_fn(q1_i, q_targets[i])
                                td2_i = loss_fn(q2_i, q_targets[i])
                                all_td1.append(td1_i)
                                all_td2.append(td2_i)
                                if w is not None:
                                    critic_i_loss = (w * td1_i).mean() + (w * td2_i).mean()
                                else:
                                    critic_i_loss = td1_i.mean() + td2_i.mean()
                                critic_loss = critic_loss + critic_i_loss
                                individual_losses_step.append(critic_i_loss.item())
                        else:
                            q1, q2 = critic(s, a)
                            td1 = loss_fn(q1, q_target)
                            td2 = loss_fn(q2, q_target)
                            if w is not None:
                                critic_loss = (w * td1).mean() + (w * td2).mean()
                            else:
                                critic_loss = td1.mean() + td2.mean()

                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2.0)
                        critic_optimizer.step()
                        critic_loss_val = critic_loss.item()
                        if USE_MULTI_CRITIC:
                            individual_critic_losses.append(individual_losses_step)

                        if P.USE_PER:
                            if USE_MULTI_CRITIC:
                                per_td = torch.stack([torch.max(td1_i, td2_i) for td1_i, td2_i in zip(all_td1, all_td2)])
                                td_errors = per_td.max(dim=0).values.detach().cpu().numpy().flatten()
                            else:
                                td_errors = torch.max(td1, td2).detach().cpu().numpy().flatten()
                            replay_buffer.update_priorities(per_indices, td_errors)

                        # Delayed actor update — Q1_forward sums all critics in multi-critic mode
                        if total_steps % POLICY_UPDATE_FREQUENCY == 0:
                            actor_loss = -critic.Q1_forward(s, actor(s)).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2.0)
                            actor_optimizer.step()

                            last_actor_loss = actor_loss.item()
                            soft_update(actor_target, actor, TAU)
                            soft_update(critic_target, critic, TAU)

                    elif EXPERIMENT_TYPE == 'DDPG':
                        # Critic update (single Q, no target smoothing)
                        with torch.no_grad():
                            next_action = actor_target(ns)
                            if USE_MULTI_CRITIC:
                                q_nexts = critic_target(ns, next_action)
                                q_targets = [
                                    r[:, i:i+1] + (1.0 - d) * GAMMA * q_i
                                    for i, q_i in enumerate(q_nexts)
                                ]
                            else:
                                q_next = critic_target(ns, next_action)
                                q_target = r + (1.0 - d) * GAMMA * q_next

                        if USE_MULTI_CRITIC:
                            critic_qs = critic(s, a)
                            all_td = []
                            critic_loss = torch.tensor(0.0, device=device)
                            individual_losses_step = []  # Track individual critic losses
                            for i, q_i in enumerate(critic_qs):
                                td_i = loss_fn(q_i, q_targets[i])
                                all_td.append(td_i)
                                if w is not None:
                                    critic_i_loss = (w * td_i).mean()
                                else:
                                    critic_i_loss = td_i.mean()
                                critic_loss = critic_loss + critic_i_loss
                                individual_losses_step.append(critic_i_loss.item())
                        else:
                            q = critic(s, a)
                            td_err = loss_fn(q, q_target)
                            if w is not None:
                                critic_loss = (w * td_err).mean()
                            else:
                                critic_loss = td_err.mean()

                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2.0)
                        critic_optimizer.step()
                        critic_loss_val = critic_loss.item()
                        if USE_MULTI_CRITIC:
                            individual_critic_losses.append(individual_losses_step)

                        if P.USE_PER:
                            if USE_MULTI_CRITIC:
                                per_td = torch.stack(all_td)
                                td_errors = per_td.max(dim=0).values.detach().cpu().numpy().flatten()
                            else:
                                td_errors = td_err.detach().cpu().numpy().flatten()
                            replay_buffer.update_priorities(per_indices, td_errors)

                        # Actor update every step — Q1_forward sums all critics in multi-critic mode
                        actor_loss = -critic.Q1_forward(s, actor(s)).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2.0)
                        actor_optimizer.step()

                        last_actor_loss = actor_loss.item()
                        soft_update(actor_target, actor, TAU)
                        soft_update(critic_target, critic, TAU)

                    elif EXPERIMENT_TYPE == 'SAC':
                        alpha = log_alpha.exp().detach()

                        # Critic update (twin Q, entropy-regularised target)
                        with torch.no_grad():
                            next_action, next_log_prob = actor.sample(ns)
                            if USE_MULTI_CRITIC:
                                target_qs = critic_target(ns, next_action)
                                # Each critic gets its own per-critic soft Bellman target
                                q_targets = [
                                    r[:, i:i+1] + (1.0 - d) * GAMMA * (torch.min(q1_i, q2_i) - alpha * next_log_prob)
                                    for i, (q1_i, q2_i) in enumerate(target_qs)
                                ]
                            else:
                                q1_next, q2_next = critic_target(ns, next_action)
                                q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
                                q_target = r + (1.0 - d) * GAMMA * q_next

                        if USE_MULTI_CRITIC:
                            critic_qs = critic(s, a)
                            all_td1, all_td2 = [], []
                            critic_loss = torch.tensor(0.0, device=device)
                            individual_losses_step = []  # Track individual critic losses
                            for i, (q1_i, q2_i) in enumerate(critic_qs):
                                td1_i = loss_fn(q1_i, q_targets[i])
                                td2_i = loss_fn(q2_i, q_targets[i])
                                all_td1.append(td1_i)
                                all_td2.append(td2_i)
                                if w is not None:
                                    critic_i_loss = (w * td1_i).mean() + (w * td2_i).mean()
                                else:
                                    critic_i_loss = td1_i.mean() + td2_i.mean()
                                critic_loss = critic_loss + critic_i_loss
                                individual_losses_step.append(critic_i_loss.item())
                        else:
                            q1, q2 = critic(s, a)
                            td1 = loss_fn(q1, q_target)
                            td2 = loss_fn(q2, q_target)
                            if w is not None:
                                critic_loss = (w * td1).mean() + (w * td2).mean()
                            else:
                                critic_loss = td1.mean() + td2.mean()

                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2.0)
                        critic_optimizer.step()
                        critic_loss_val = critic_loss.item()
                        if USE_MULTI_CRITIC:
                            individual_critic_losses.append(individual_losses_step)

                        if P.USE_PER:
                            if USE_MULTI_CRITIC:
                                per_td = torch.stack([torch.max(td1_i, td2_i) for td1_i, td2_i in zip(all_td1, all_td2)])
                                td_errors = per_td.max(dim=0).values.detach().cpu().numpy().flatten()
                            else:
                                td_errors = torch.max(td1, td2).detach().cpu().numpy().flatten()
                            replay_buffer.update_priorities(per_indices, td_errors)

                        # Actor update — entropy applied once; Q1_forward sums all critics
                        new_action, new_log_prob = actor.sample(s)
                        q_new = critic.Q1_forward(s, new_action)
                        actor_loss = (alpha * new_log_prob - q_new).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2.0)
                        actor_optimizer.step()

                        last_actor_loss = actor_loss.item()

                        # Alpha (entropy coefficient) update
                        alpha_loss = -(log_alpha * (new_log_prob.detach() + P.SAC_TARGET_ENTROPY)).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()

                        # Soft update critic target (no actor target in SAC)
                        soft_update(critic_target, critic, TAU)

                perf_data = [np.nanmean(perf_metrics[n]) if perf_metrics[n] else 0.0 for n in metric_names]

                # Append training data with individual critic losses if multi-critic
                if USE_MULTI_CRITIC and individual_critic_losses:
                    avg_individual_losses = np.nanmean(individual_critic_losses, axis=0).tolist()
                    training_data.append([critic_loss_val, last_actor_loss, *avg_individual_losses, *perf_data])
                else:
                    training_data.append([critic_loss_val, last_actor_loss, *perf_data])

            # Logging
            if len(training_data) >= SUMMARY_WINDOW:
                log_metrics(training_data, curr_episode, total_steps, replay_buffer.get_length(),
                            writer, wandb_run)
                training_data = []
                perf_metrics = {n: [] for n in metric_names}
                individual_critic_losses = []  # Reset after logging

            weights_set = get_weights(actor, device, local_device)

            if curr_episode % CHECKPOINT_EVERY == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                if wandb_run:
                    wandb_run.log({}, commit=True)
                save_checkpoint(
                    actor, critic, actor_target, critic_target,
                    actor_optimizer, critic_optimizer,
                    curr_episode, total_steps,
                    log_alpha=log_alpha, alpha_optimizer=alpha_optimizer,
                )

            if curr_episode % SAVE_BUFFER_EVERY == 0:
                buffer_path = os.path.join(CHECKPOINT_DIR, 'buffer_latest.pkl')
                with open(buffer_path, 'wb') as f:
                    pickle.dump(replay_buffer, f)
                print(f'Saved replay buffer ({replay_buffer.get_length()} transitions) -> {buffer_path}')

    except KeyboardInterrupt:
        print("CTRL+C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
        save_checkpoint(
            actor, critic, actor_target, critic_target,
            actor_optimizer, critic_optimizer,
            curr_episode, total_steps,
            log_alpha=log_alpha, alpha_optimizer=alpha_optimizer,
        )
    finally:
        writer.close()
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
