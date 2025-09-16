#!/usr/bin/env python3
"""
Convergence testing for the brane-world GW sandbox.

This module tests numerical convergence of the evolution code
by running simulations at different resolutions and checking
that phase differences scale appropriately.
"""

import numpy as np
import json
from typing import List, Dict, Tuple
from bssn_brane_evolver import BSSNBraneEvolver
from template_bank import TemplateBank


def run_convergence_test(
    resolutions: List[int] = [32, 64, 128],
    mass1: float = 1.4,
    mass2: float = 1.4,
    save_results: bool = True
) -> Dict:
    """
    Run convergence test at multiple resolutions.

    Args:
        resolutions: List of grid resolutions to test
        mass1: Primary mass in solar masses
        mass2: Secondary mass in solar masses
        save_results: Whether to save results to file

    Returns:
        Dictionary with convergence test results
    """

    print(f"Running convergence test for M1={mass1}, M2={mass2}")
    print(f"Testing resolutions: {resolutions}")

    results = {
        'resolutions': resolutions,
        'mass1': mass1,
        'mass2': mass2,
        'phases': [],
        'amplitudes': [],
        'convergence_order': None,
        'errors': []
    }

    # Run simulations at each resolution
    for N in resolutions:
        print(f"\nRunning N={N}...")

        # Initialize evolver and template bank
        evolver = BSSNBraneEvolver(
            N=N,
            box_size=200.0,
            cfl=0.25,
            brane_stiffness=0.1,
            matter_scalar_coupling=0.05
        )

        bank = TemplateBank(evolver)

        # Generate waveform
        h_plus, h_cross, times = bank.generate_template(
            mass1, mass2,
            chi1=[0, 0, 0.1],
            chi2=[0, 0, -0.1],
            brane_amplitude=0.1
        )

        # Extract phase and amplitude at merger
        merger_idx = np.argmax(np.abs(h_plus))
        phase = np.angle(h_plus[merger_idx] + 1j * h_cross[merger_idx])
        amplitude = np.abs(h_plus[merger_idx])

        results['phases'].append(phase)
        results['amplitudes'].append(amplitude)

        print(f"  Phase at merger: {phase:.6f} rad")
        print(f"  Amplitude at merger: {amplitude:.3e}")

    # Calculate convergence order using Richardson extrapolation
    if len(resolutions) >= 3:
        phases = np.array(results['phases'])

        # Assuming second-order convergence, estimate errors
        for i in range(len(phases) - 1):
            error = abs(phases[i+1] - phases[i])
            results['errors'].append(error)

        # Richardson extrapolation for convergence order
        if len(phases) >= 3:
            e1 = abs(phases[1] - phases[0])  # Error at coarse resolution
            e2 = abs(phases[2] - phases[1])  # Error at fine resolution

            if e2 > 0 and e1 > 0:
                # Assuming grid spacing halves each time
                convergence_order = np.log2(e1 / e2)
                results['convergence_order'] = convergence_order

                print(f"\nConvergence Analysis:")
                print(f"  Error (N={resolutions[0]}→{resolutions[1]}): {e1:.6e} rad")
                print(f"  Error (N={resolutions[1]}→{resolutions[2]}): {e2:.6e} rad")
                print(f"  Convergence order: {convergence_order:.2f}")

                if convergence_order < 1.5:
                    print("  WARNING: Poor convergence! Expected order ≥ 2")
                else:
                    print("  ✓ Good convergence")

    # Save results
    if save_results:
        with open('convergence_test_results.json', 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            json_results = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to convergence_test_results.json")

    return results


def plot_convergence(results: Dict, save_fig: bool = True):
    """
    Plot convergence test results.

    Args:
        results: Dictionary from run_convergence_test
        save_fig: Whether to save figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot phase vs resolution
        ax1.plot(results['resolutions'], results['phases'], 'o-', linewidth=2)
        ax1.set_xlabel('Grid Resolution N')
        ax1.set_ylabel('Phase at Merger (rad)')
        ax1.set_title('Phase Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)

        # Plot error vs resolution
        if results['errors']:
            res_pairs = [(results['resolutions'][i], results['resolutions'][i+1])
                        for i in range(len(results['errors']))]
            res_labels = [f"{r1}→{r2}" for r1, r2 in res_pairs]

            ax2.bar(range(len(results['errors'])), results['errors'])
            ax2.set_xticks(range(len(results['errors'])))
            ax2.set_xticklabels(res_labels)
            ax2.set_ylabel('Phase Error (rad)')
            ax2.set_title(f"Convergence Order: {results['convergence_order']:.2f}")
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Convergence Test: M1={results['mass1']}M☉, M2={results['mass2']}M☉")
        plt.tight_layout()

        if save_fig:
            plt.savefig('convergence_test.png', dpi=150, bbox_inches='tight')
            print("Plot saved to convergence_test.png")

        plt.show()

    except ImportError:
        print("Matplotlib not available, skipping plot")


def verify_convergence_requirement(
    min_order: float = 1.5,
    resolutions: List[int] = None
) -> bool:
    """
    Verify that the code meets minimum convergence requirements.

    Args:
        min_order: Minimum acceptable convergence order
        resolutions: Grid resolutions to test

    Returns:
        True if convergence is acceptable, False otherwise
    """

    if resolutions is None:
        resolutions = [32, 64, 128]

    print("=" * 60)
    print("CONVERGENCE VERIFICATION TEST")
    print("=" * 60)

    # Test BBH system
    print("\nTesting BBH (30+30 M☉):")
    bbh_results = run_convergence_test(
        resolutions=resolutions,
        mass1=30.0,
        mass2=30.0,
        save_results=False
    )

    # Test BNS system
    print("\nTesting BNS (1.4+1.4 M☉):")
    bns_results = run_convergence_test(
        resolutions=resolutions,
        mass1=1.4,
        mass2=1.4,
        save_results=False
    )

    # Check convergence orders
    passed = True

    if bbh_results['convergence_order'] is not None:
        if bbh_results['convergence_order'] < min_order:
            print(f"\n❌ BBH convergence order {bbh_results['convergence_order']:.2f} < {min_order}")
            passed = False
        else:
            print(f"\n✓ BBH convergence order {bbh_results['convergence_order']:.2f} ≥ {min_order}")

    if bns_results['convergence_order'] is not None:
        if bns_results['convergence_order'] < min_order:
            print(f"❌ BNS convergence order {bns_results['convergence_order']:.2f} < {min_order}")
            passed = False
        else:
            print(f"✓ BNS convergence order {bns_results['convergence_order']:.2f} ≥ {min_order}")

    print("\n" + "=" * 60)
    if passed:
        print("✓ CONVERGENCE TEST PASSED")
    else:
        print("❌ CONVERGENCE TEST FAILED")
        print("The numerical scheme needs improvement for reliable results")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    # Run basic convergence test
    results = run_convergence_test()

    # Plot results
    plot_convergence(results)

    # Verify convergence meets requirements
    verify_convergence_requirement()