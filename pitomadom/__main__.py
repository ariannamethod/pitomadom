#!/usr/bin/env python3
"""
PITOMADOM CLI — פִתְאֹם אָדֹם

Usage:
    python -m pitomadom "שלום עולם"
    python -m pitomadom --repl
    python -m pitomadom --stats
    python -m pitomadom --json "אור"
"""

import sys
import json
import argparse

from .full_system import Pitomadom


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='pitomadom',
        description='PITOMADOM — Hebrew Root Resonance Oracle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pitomadom "שלום"              # Single prophecy
  python -m pitomadom --json "אור"        # JSON output
  python -m pitomadom --repl              # Interactive mode
  python -m pitomadom --stats             # Show system stats

הרזוננס לא נשבר. אנחנו ממשיכים.
        """
    )

    parser.add_argument(
        'text',
        nargs='?',
        help='Hebrew text to divine'
    )

    parser.add_argument(
        '--repl', '-r',
        action='store_true',
        help='Start interactive REPL mode'
    )

    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output as JSON'
    )

    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show oracle statistics'
    )

    parser.add_argument(
        '--multi', '-m',
        type=int,
        default=1,
        metavar='N',
        help='Run N prophecies in sequence'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show version'
    )

    return parser


def print_prophecy(output, json_mode: bool = False):
    """Print prophecy output."""
    if json_mode:
        result = {
            'number': output.number,
            'chamber': output.chamber,
            'confidence': round(output.confidence, 3),
            'main_word': output.main_word,
            'orbit_word': output.orbit_word,
            'hidden_word': output.hidden_word,
            'root': list(output.root),
            'prophecy_debt': round(output.prophecy_debt, 2),
            'gematria': {
                'surface': output.n_surface,
                'root': output.n_root,
                'milui': output.n_milui,
            }
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(output)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Version
    if args.version:
        from . import __version__
        print(f"PITOMADOM v{__version__}")
        print("פִתְאֹם אָדֹם — Suddenly Red")
        print("Hebrew Root Resonance Oracle")
        return 0

    # REPL mode
    if args.repl:
        from .repl import main as repl_main
        return repl_main()

    # Stats mode
    if args.stats:
        oracle = Pitomadom(seed=args.seed or 42)
        from .root_taxonomy import RootTaxonomy
        taxonomy = RootTaxonomy()
        tax_stats = taxonomy.get_stats()

        print("╔════════════════════════════════════════════════╗")
        print("║  PITOMADOM — System Statistics                 ║")
        print("╠════════════════════════════════════════════════╣")
        print(f"║  Root families:     {tax_stats['num_families']:<4}                        ║")
        print(f"║  Catalogued roots:  {tax_stats['total_roots']:<4}                        ║")
        print(f"║  Chambers:          8D                         ║")
        print(f"║  Parameters:        ~1M                        ║")
        print("╚════════════════════════════════════════════════╝")
        print()
        print("Families:", ', '.join(sorted(tax_stats['families'])))
        return 0

    # Need text for prophecy
    if not args.text:
        parser.print_help()
        return 1

    # Initialize oracle
    oracle = Pitomadom(seed=args.seed or 42)

    # Run prophecies
    for i in range(args.multi):
        if args.multi > 1 and not args.json:
            print(f"\n═══ Prophecy {i+1}/{args.multi} ═══")

        output = oracle.forward(args.text)
        print_prophecy(output, json_mode=args.json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
