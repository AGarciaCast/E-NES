from equiv_eikonal.models.solvers import get_solver
from equiv_eikonal.steerable_attention.threeway_invariants import get_invariants
from equiv_eikonal.latents import get_latents


def get_models(cfg, vmin=1e-8, vmax=1.0, meta=False, functa=False):
    """Get autodecoders and snef based on the configuration."""

    # Determine whether we are doing meta-learning
    if not meta:
        # Init invariant
        invariant = get_invariants(cfg.geometry)

        # Init autodecoder
        train_autodecoder = get_latents(cfg.data.base_dataset.num_signals_train, cfg)
        val_autodecoder = get_latents(cfg.data.base_dataset.num_signals_val, cfg)
        test_autodecoder = get_latents(cfg.data.base_dataset.num_signals_test, cfg)

        # Init solver
        solver = get_solver(
            cfg=cfg,
            invariant=invariant,
            vmin=vmin,
            vmax=vmax,
        )

        return solver, train_autodecoder, val_autodecoder, test_autodecoder
    else:
        # Init invariant
        invariant = get_invariants(cfg.geometry)

        # Init autodecoder
        train_autodecoder = get_latents(cfg.data.train_batch_size, cfg, meta=True)
        # We have two different ones just in case in the future we want different batch size on val and test
        val_autodecoder = get_latents(cfg.data.test_batch_size, cfg, meta=True)
        test_autodecoder = get_latents(cfg.data.test_batch_size, cfg, meta=True)

        init_latents = get_latents(1, cfg, meta=True)

        # Init solver
        solver = get_solver(
            cfg=cfg,
            invariant=invariant,
            vmin=vmin,
            vmax=vmax,
        )

        return (
            solver,
            train_autodecoder,
            val_autodecoder,
            test_autodecoder,
            init_latents,
        )
