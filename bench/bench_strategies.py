"""Compare the fine-grained (master-worker) and coarse-grained (island-parallel)
strategies on a PMLB dataset, reusing the existing ``bench.eval`` driver.

This exercises the *expensive, load-imbalanced* fitness regime: the PMLB config uses
``use_constants: True``, so each individual with constants triggers a pygmo PSO run.

Run from the ``bench`` directory:

    python bench_strategies.py 192_vineyard
"""

import argparse
import time

import ray

from bench import eval as bench_eval


def make_step_timer(label):
    """Return a ``custom_logger`` that times the interval between calls.

    The regressor calls ``custom_logger(best_inds)`` once per generation
    (fine-grained) or once per migration block (coarse-grained). Printing the
    wall-clock delta between consecutive calls reveals whether per-step time stays
    flat (a perceived slowdown is just the bursty print cadence) or rises over the
    run (a genuine slowdown, typically GP bloat). ``best_size`` is the size of the
    current best individual, a quick proxy for tree growth.
    """
    state = {"last": time.perf_counter(), "step": 0, "cum": 0.0}

    def logger(best_inds):
        now = time.perf_counter()
        dt = now - state["last"]
        state["last"] = now
        state["step"] += 1
        state["cum"] += dt
        size = len(best_inds[0]) if best_inds else -1
        print(
            f"[{label} step {state['step']:>3}] +{dt:6.2f}s "
            f"(cum {state['cum']:7.1f}s)  best_size={size}",
            flush=True,
        )

    return logger


def warmup_workers():
    """Force every Ray worker to start and import the heavy dependencies.

    The first batch of fitness tasks otherwise pays a large one-time cost: ~10 fresh
    worker processes import pygmo/mygrad/flex and run their first pygmo solve. Whichever
    strategy is timed first absorbs this cold-start, which would inflate its time and
    the reported speed-up. Warming the workers up front makes both strategies timed in
    the same (warm) steady state.
    """
    n = max(int(ray.available_resources().get("CPU", 1)), 1)

    @ray.remote
    def _warm():
        import time as _t
        import bench  # noqa: F401  (imports pygmo, mygrad, flex, ...)
        import flex.gp.regressor  # noqa: F401  (registers creator classes)

        _t.sleep(0.3)  # hold the slot so the batch spreads across all workers
        return True

    t0 = time.perf_counter()
    ray.get([_warm.remote() for _ in range(n * 2)])
    print(f"[warmup] {n} workers warmed in {time.perf_counter() - t0:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", nargs="?", default="192_vineyard",
                        help="Name of a (cached) PMLB dataset.")
    parser.add_argument("--cfg", default="PMLB_quick.yaml",
                        help="Path of the YAML config file.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    print(f"\n### PROBLEM: {args.problem}  (cfg={args.cfg}, seed={args.seed})\n")

    # warm the workers so neither strategy is charged the one-time cold-start
    warmup_workers()

    print("=== FINE-GRAINED (original master-worker) ===")
    t0 = time.perf_counter()
    r2tr_f, r2te_f, model_f, fit_f = bench_eval(
        problem=args.problem, cfgfile=args.cfg, seed=args.seed,
        coarse_grained_islands=False, remove_init_duplicates=False,
        custom_logger=make_step_timer("fine  "),
    )
    wall_f = time.perf_counter() - t0

    print("\n=== COARSE-GRAINED (island-parallel) ===")
    t0 = time.perf_counter()
    r2tr_c, r2te_c, model_c, fit_c = bench_eval(
        problem=args.problem, cfgfile=args.cfg, seed=args.seed,
        coarse_grained_islands=True, remove_init_duplicates=False,
        custom_logger=make_step_timer("coarse"),
    )
    wall_c = time.perf_counter() - t0

    print("\n" + "=" * 74)
    print(f"{'Strategy':<18}{'fit time (s)':>14}{'wall (s)':>12}"
          f"{'R2 train':>12}{'R2 test':>12}")
    print("-" * 74)
    print(f"{'fine-grained':<18}{fit_f:>14.2f}{wall_f:>12.2f}"
          f"{r2tr_f:>12.3f}{r2te_f:>12.3f}")
    print(f"{'coarse-grained':<18}{fit_c:>14.2f}{wall_c:>12.2f}"
          f"{r2tr_c:>12.3f}{r2te_c:>12.3f}")
    print("-" * 74)
    speedup = fit_f / fit_c if fit_c > 0 else float("nan")
    print(f"Speed-up on fit time (fine / coarse): {speedup:.2f}x")
    print("=" * 74)
    print(f"best (fine)   = {model_f}")
    print(f"best (coarse) = {model_c}")

    ray.shutdown()


if __name__ == "__main__":
    main()
