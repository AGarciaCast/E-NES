import jax.numpy as jnp
import jax


def cosine_diminishing_schedule(
    init_value, min_value, total_steps, warmup_steps=0, freq=1.0
):
    """
    Creates a schedule with cosine annealing and linearly diminishing amplitude.
    The schedule starts at exactly init_value, ends at min_value, and has cosine oscillations
    with amplitude that decreases linearly over time.

    Args:
        init_value: Initial learning rate value after warmup
        min_value: Minimum learning rate value (floor)
        total_steps: Total number of steps for the schedule
        warmup_steps: Number of steps for linear warmup (default: 0)
        freq: Frequency of cosine cycles (default: 1.0)

    Returns:
        A schedule function mapping step counts to values
    """

    def warmup_value(step):
        # Linear warmup phase
        return init_value * (step / jnp.maximum(1.0, warmup_steps))

    def cosine_annealing_value(step):
        # Calculate progress after warmup
        progress = (step - warmup_steps) / jnp.maximum(1.0, total_steps - warmup_steps)

        # Linearly decreasing amplitude
        max_cosine_amplitude = 0.5 * (1.0 - progress)

        # Apply cosine schedule with linearly diminishing amplitude
        return min_value + (init_value - min_value) * max_cosine_amplitude * (
            1.0 + jnp.cos(freq * jnp.pi * progress)
        )

    def schedule_fn(step):
        # Use lax.cond instead of Python if-statement
        return jax.lax.cond(
            step < warmup_steps, warmup_value, cosine_annealing_value, step
        )

    return schedule_fn


def cosine_cycle_schedule(init_value, min_value, total_steps, warmup_steps=0, freq=1.0):
    def warmup_value(step):
        return init_value * (step / jnp.maximum(1.0, warmup_steps))

    def cosine_decay_value(step):
        progress = (step - warmup_steps) / jnp.maximum(1.0, total_steps - warmup_steps)
        return min_value + (init_value - min_value) * (
            0.5 * (1.0 + jnp.cos(freq * jnp.pi * progress))
        )

    def schedule_fn(step):
        return jax.lax.cond(step < warmup_steps, warmup_value, cosine_decay_value, step)

    return schedule_fn
