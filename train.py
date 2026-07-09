import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from canvas import Canvas
from models import Critic, PainterPolicy


def image_to_tensor(image, device):
    image = image.convert("RGB")
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32, device=device)
    pixels = pixels.view(image.height, image.width, 3) / 255.0
    return pixels.permute(2, 0, 1).unsqueeze(0)


def canvas_to_tensor(canvas, device):
    return image_to_tensor(canvas.to_image(), device)


def load_target(path, image_size, device):
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    return image, image_to_tensor(image, device)


def tensor_action_to_list(action):
    return action.detach().cpu().squeeze(0).tolist()


def l1_reward(canvas_tensor, target_tensor):
    return -F.l1_loss(canvas_tensor, target_tensor).item()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def add_many(self, items):
        self.items.extend(items)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]

    def sample(self, batch_size):
        batch = random.sample(self.items, min(batch_size, len(self.items)))
        target, canvas, step, action, ret = zip(*batch)
        return (
            torch.cat(target, dim=0),
            torch.cat(canvas, dim=0),
            torch.cat(step, dim=0),
            torch.cat(action, dim=0),
            torch.cat(ret, dim=0),
        )

    def __len__(self):
        return len(self.items)


def rollout(
    policy,
    target_tensor,
    image_size,
    steps,
    gamma,
    device,
    min_radius,
    max_radius,
    max_length,
):
    canvas = Canvas(image_size, image_size)
    trajectory = []

    with torch.no_grad():
        for step in range(steps):
            canvas_tensor = canvas_to_tensor(canvas, device)
            step_percentage = torch.tensor([[step / steps]], dtype=torch.float32, device=device)
            action, _, _ = policy.sample_action(target_tensor, canvas_tensor, step_percentage)

            trajectory.append((
                target_tensor.detach().clone(),
                canvas_tensor.detach().clone(),
                step_percentage.detach().clone(),
                action.detach().clone(),
            ))

            canvas.draw_centered_normalized_stroke(
                tensor_action_to_list(action),
                min_radius=min_radius,
                max_radius=max_radius,
                max_length=max_length,
            )

    final_canvas_tensor = canvas_to_tensor(canvas, device)
    final_reward = l1_reward(final_canvas_tensor, target_tensor)

    transitions = []
    for index, (target, canvas_tensor, step_percentage, action) in enumerate(trajectory):
        discount = gamma ** (steps - 1 - index)
        ret = torch.tensor([[discount * final_reward]], dtype=torch.float32, device=device)
        transitions.append((target, canvas_tensor, step_percentage, action, ret))

    return canvas, final_reward, transitions


def train_critic(critic, optimizer, replay, batch_size, updates):
    if len(replay) == 0:
        return 0.0

    losses = []
    for _ in range(updates):
        target, canvas, step, action, ret = replay.sample(batch_size)
        predicted_return = critic(target, canvas, step, action)
        loss = F.mse_loss(predicted_return, ret)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses)


def train_policy(policy, critic, optimizer, replay, batch_size, updates):
    if len(replay) == 0:
        return 0.0

    for parameter in critic.parameters():
        parameter.requires_grad_(False)

    losses = []
    for _ in range(updates):
        target, canvas, step, _, _ = replay.sample(batch_size)
        mean, std = policy(target, canvas, step)
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.rsample().clamp(0.0, 1.0)

        predicted_return = critic(target, canvas, step, action)
        entropy_bonus = distribution.entropy().mean()
        loss = -predicted_return.mean() - 0.001 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    for parameter in critic.parameters():
        parameter.requires_grad_(True)

    return sum(losses) / len(losses)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement-learning AI painter.")
    parser.add_argument("target", help="Path to the target image.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--critic-updates", type=int, default=8)
    parser.add_argument("--policy-updates", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--min-radius", type=float, default=1.0)
    parser.add_argument("--max-radius", type=float, default=None)
    parser.add_argument("--max-length", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_image, target_tensor = load_target(args.target, args.image_size, device)
    target_image.save(output_dir / "target.png")

    policy = PainterPolicy(args.image_size).to(device)
    critic = Critic(args.image_size).to(device)
    replay = ReplayBuffer(args.buffer_size)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    for episode in range(1, args.episodes + 1):
        canvas, reward, transitions = rollout(
            policy=policy,
            target_tensor=target_tensor,
            image_size=args.image_size,
            steps=args.steps,
            gamma=args.gamma,
            device=device,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            max_length=args.max_length,
        )
        replay.add_many(transitions)

        critic_loss = train_critic(
            critic,
            critic_optimizer,
            replay,
            args.batch_size,
            args.critic_updates,
        )
        policy_loss = train_policy(
            policy,
            critic,
            policy_optimizer,
            replay,
            args.batch_size,
            args.policy_updates,
        )

        if episode == 1 or episode % 10 == 0 or episode == args.episodes:
            canvas.save(output_dir / f"episode_{episode:04d}.png")

        print(
            f"episode={episode:04d} "
            f"reward={reward:.4f} "
            f"critic_loss={critic_loss:.4f} "
            f"policy_loss={policy_loss:.4f} "
            f"buffer={len(replay)}"
        )

    torch.save(policy.state_dict(), output_dir / "policy.pt")
    torch.save(critic.state_dict(), output_dir / "critic.pt")


if __name__ == "__main__":
    main()
