import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")


from experiments.fitting.ablation_auto_steps import run_ablation


if __name__ == "__main__":

    num_auto_steps = 500

    # Flat A equivariant plot
    df, name_ds = run_ablation(
        wandb_ref="9ybqwgwn",
        num_steps=[num_auto_steps],
        plot=num_auto_steps,
        save=False,
        warmup=False,
        equiv_plot=True,
        num_visualised=20,
        final_plot_equiv=True,
        final_plot_gt=False,
    )
