"""
CLI for Digital Option Strike Shift Timing.

Usage:
    python main.py --spot 1.085 --strike 1.10 --vol 6 --expiry 30 --shift 5
    python main.py --interactive
"""

import argparse
import sys

from solver import ShiftTimingSolver, Decision


def print_header():
    """Print tool header."""
    print("=" * 60)
    print("  Digital Option Strike Shift Timing")
    print("  Optimal Timing via Certainty Equivalent")
    print("=" * 60)
    print()


def format_result(result, params: dict) -> str:
    """Format solver result for display."""
    lines = []

    # Decision
    if result.decision == Decision.SHIFT_NOW:
        lines.append(">>> RECOMMENDATION: SHIFT NOW <<<")
    else:
        lines.append(">>> RECOMMENDATION: WAIT <<<")

    lines.append("")

    # Cost comparison
    lines.append("Cost Analysis:")
    lines.append(f"  Cost if shift now:      {result.cost_now:.6f}")
    lines.append(f"  Expected cost (wait):   {result.expected_cost_wait:.6f}")
    lines.append(f"  Certainty equiv (wait): {result.ce_wait:.6f}")
    lines.append("")

    # Stats
    import numpy as np
    std_wait = np.sqrt(result.variance_cost_wait)
    lines.append("Statistics:")
    lines.append(f"  Current gamma:          {result.gamma_now:.6f}")
    lines.append(f"  Std dev (wait):         {std_wait:.6f}")
    lines.append(f"  Days to deadline:       {result.days_remaining}")
    lines.append(f"  Risk aversion (γ):      {result.risk_aversion}")
    lines.append("")

    # Interpretation
    lines.append("Interpretation:")
    if result.decision == Decision.SHIFT_NOW:
        lines.append(f"  Current cost ({result.cost_now:.4f}) < CE of waiting ({result.ce_wait:.4f})")
        lines.append("  Risk-adjusted analysis favors shifting now.")
    else:
        diff = result.cost_now - result.ce_wait
        lines.append(f"  Current cost ({result.cost_now:.4f}) > CE of waiting ({result.ce_wait:.4f})")
        lines.append(f"  Potential savings of {diff:.4f} by waiting.")
        lines.append("  Spot may move away from strike, reducing gamma.")

    return "\n".join(lines)


def analyze(
    spot: float,
    strike: float,
    vol: float,
    expiry_days: float,
    shift_bps: float,
    risk_aversion: float = 2.0,
    deadline_days: int = 14,
    n_paths: int = 10000,
    verbose: bool = True,
) -> dict:
    """Run analysis and return results."""
    if verbose:
        print_header()
        print("Parameters:")
        print(f"  Spot:           {spot:.4f}")
        print(f"  Strike:         {strike:.4f}")
        print(f"  Volatility:     {vol:.1%}")
        print(f"  Days to expiry: {expiry_days}")
        print(f"  Shift size:     {shift_bps} bps")
        print(f"  Risk aversion:  {risk_aversion}")
        print(f"  Deadline:       {deadline_days} days before expiry")
        print()

    solver = ShiftTimingSolver(
        spot=spot,
        strike=strike,
        vol=vol,
        expiry_days=expiry_days,
        shift_bps=shift_bps,
        risk_aversion=risk_aversion,
        deadline_days=deadline_days,
        n_paths=n_paths,
        seed=42,
    )

    if verbose:
        print(f"Running simulation ({n_paths:,} paths)...")
        print()

    result = solver.solve()

    if verbose:
        params = {'spot': spot, 'strike': strike, 'vol': vol}
        print(format_result(result, params))
        print()

    return {
        "decision": result.decision.value,
        "shift_now": result.decision == Decision.SHIFT_NOW,
        "cost_now": result.cost_now,
        "expected_cost_wait": result.expected_cost_wait,
        "ce_wait": result.ce_wait,
        "gamma_now": result.gamma_now,
        "days_remaining": result.days_remaining,
    }


def interactive_mode():
    """Run in interactive mode."""
    print_header()
    print("Interactive Mode")
    print("-" * 40)

    try:
        spot = float(input("Spot price: "))
        strike = float(input("Strike price: "))
        vol_pct = float(input("Volatility (%): "))
        vol = vol_pct / 100
        expiry_days = float(input("Days to expiry: "))
        shift_bps = float(input("Shift size (bps): "))

        risk_str = input("Risk aversion [2.0]: ").strip()
        risk_aversion = float(risk_str) if risk_str else 2.0

        deadline_str = input("Deadline days [14]: ").strip()
        deadline_days = int(deadline_str) if deadline_str else 14

        print()
        analyze(
            spot=spot,
            strike=strike,
            vol=vol,
            expiry_days=expiry_days,
            shift_bps=shift_bps,
            risk_aversion=risk_aversion,
            deadline_days=deadline_days,
        )

    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError: {e}")
        sys.exit(1)


def run_example():
    """Run example analysis."""
    print("Running example: EURUSD digital option")
    print()

    analyze(
        spot=1.0850,
        strike=1.1000,
        vol=0.06,
        expiry_days=30,
        shift_bps=5,
        risk_aversion=2.0,
        deadline_days=14,
        n_paths=10000,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Digital Option Strike Shift Timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --spot 1.085 --strike 1.10 --vol 6 --expiry 30 --shift 5
  python main.py --interactive
  python main.py --example
        """
    )

    parser.add_argument("--spot", type=float, help="Current spot price")
    parser.add_argument("--strike", type=float, help="Option strike")
    parser.add_argument("--vol", type=float, help="Volatility in %% (e.g., 6 for 6%%)")
    parser.add_argument("--expiry", type=float, help="Days to expiry")
    parser.add_argument("--shift", type=float, help="Shift size in bps")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="Risk aversion parameter (default: 1.0)")
    parser.add_argument("--deadline", type=int, default=14,
                       help="Days before expiry deadline (default: 14)")
    parser.add_argument("--paths", type=int, default=10000,
                       help="Number of MC paths (default: 10000)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--example", "-e", action="store_true",
                       help="Run example analysis")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.example:
        run_example()
    elif all([args.spot, args.strike, args.vol, args.expiry, args.shift]):
        analyze(
            spot=args.spot,
            strike=args.strike,
            vol=args.vol / 100,  # Convert from % to decimal
            expiry_days=args.expiry,
            shift_bps=args.shift,
            risk_aversion=args.gamma,
            deadline_days=args.deadline,
            n_paths=args.paths,
        )
    else:
        parser.print_help()
        print("\nError: Provide all required parameters or use --interactive/--example")
        sys.exit(1)


if __name__ == "__main__":
    main()
