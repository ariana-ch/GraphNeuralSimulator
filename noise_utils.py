import torch

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """
    Generate random-walk noise for velocity applied to positions.

    Args:
        position_sequence (torch.Tensor): Tensor of shape [num_particles, sequence_length, num_dimensions].
        noise_std_last_step (float): Standard deviation of noise for the last velocity step.

    Returns:
        torch.Tensor: Random-walk noise added to the position sequence.
    """
    sequence_length = position_sequence.shape[1] - 1  # exclude initial position

    # Compute per-step noise standard deviation
    noise_std_per_step = noise_std_last_step / (sequence_length ** 0.5)

    # Generate random velocity noise for each step
    velocity_noise = torch.randn_like(position_sequence[:, 1:]) * noise_std_per_step

    # Integrate velocity noise to get position noise
    position_noise = torch.zeros_like(position_sequence)
    position_noise[:, 1:] = torch.cumsum(velocity_noise, dim=1)

    return position_noise


if __name__ == '__main__':
    # Example usage
    num_particles = 100
    sequence_length = 6
    num_dimensions = 3
    noise_std_last_step = 0.01

    # Create dummy position sequence
    position_sequence = torch.randn((num_particles, sequence_length, num_dimensions))

    # Generate noise
    position_noise = get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step)

    print(f"Generated noise shape: {position_noise.shape}")
