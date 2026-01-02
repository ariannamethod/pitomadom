#!/usr/bin/env python3
"""
PITOMADOM REPL — Interactive Hebrew Root Resonance Oracle

פִתְאֹם אָדֹם — Suddenly red
פִתֻם אָדֹם — The red ventriloquist

Usage:
    python -m pitomadom.repl
    
Commands:
    :stats  - Show oracle statistics
    :reset  - Reset oracle state
    :traj   - Show N-trajectory
    :debt   - Show prophecy debt breakdown
    :roots  - Show active root attractors
    :full   - Toggle full/compact output mode
    :help   - Show help
    :quit   - Exit
"""

import sys
import readline  # Enable arrow keys and history


def print_banner():
    """Print PITOMADOM banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ██████╗  ██╗████████╗ ██████╗ ███╗   ███╗ █████╗ ██████╗  ██████╗ ███╗   ███╗  ║
║  ██╔══██╗ ██║╚══██╔══╝██╔═══██╗████╗ ████║██╔══██╗██╔══██╗██╔═══██╗████╗ ████║  ║
║  ██████╔╝ ██║   ██║   ██║   ██║██╔████╔██║███████║██║  ██║██║   ██║██╔████╔██║  ║
║  ██╔═══╝  ██║   ██║   ██║   ██║██║╚██╔╝██║██╔══██║██║  ██║██║   ██║██║╚██╔╝██║  ║
║  ██║      ██║   ██║   ╚██████╔╝██║ ╚═╝ ██║██║  ██║██████╔╝╚██████╔╝██║ ╚═╝ ██║  ║
║  ╚═╝      ╚═╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝     ╚═╝  ║
╠══════════════════════════════════════════════════════════════════╣
║  פתאום אדום — Hebrew Root Resonance Oracle                       ║
║  ~200K parameters • CrossFire Chambers • Prophecy Engine         ║
╠══════════════════════════════════════════════════════════════════╣
║  Commands: :stats :reset :traj :debt :roots :help :quit          ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_help():
    """Print help."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PITOMADOM REPL — Commands                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  :stats  - Show oracle statistics (step, debt, roots, etc.)      ║
║  :reset  - Reset oracle state (new conversation)                 ║
║  :traj   - Show N-trajectory (last 10 values)                    ║
║  :debt   - Show prophecy debt breakdown                          ║
║  :roots  - Show active root attractors                           ║
║  :full   - Toggle full/compact output mode                       ║
║  :help   - Show this help                                        ║
║  :quit   - Exit (also: :exit, :q, Ctrl+C)                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Input any Hebrew text to query the oracle.                      ║
║  Examples:                                                        ║
║    שלום                                                          ║
║    אני מפחד אבל רוצה להמשיך                                       ║
║    האור נשבר בחושך                                                ║
║    פתאום אדום                                                     ║
╚══════════════════════════════════════════════════════════════════╝
""")


def format_compact_output(output):
    """Format output in compact mode."""
    root_str = '.'.join(output.root)
    return f"""    N={output.number} • root={root_str} • debt={output.prophecy_debt:.1f}
    main: {output.main_word}  orbit: {output.orbit_word}  hidden: {output.hidden_word}"""


def format_trajectory(temporal_field):
    """Format N-trajectory."""
    traj = temporal_field.state.n_trajectory[-10:]  # Last 10
    if not traj:
        return "    (empty trajectory)"
    
    lines = ["    N-trajectory (last 10):"]
    lines.append(f"    {' → '.join(str(n) for n in traj)}")
    
    if len(traj) >= 2:
        velocity = traj[-1] - traj[-2]
        lines.append(f"    velocity: {velocity:+d}")
    
    if len(traj) >= 3:
        v1 = traj[-2] - traj[-3]
        v2 = traj[-1] - traj[-2]
        accel = v2 - v1
        lines.append(f"    acceleration: {accel:+d}")
    
    return '\n'.join(lines)


def format_stats(oracle):
    """Format oracle statistics."""
    stats = oracle.get_stats()
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║  PITOMADOM Statistics                                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Step:             {stats['step']:<10}                                    ║
║  Prophecy Debt:    {stats['prophecy_debt']:<10.2f}                                ║
║  Unique Roots:     {stats['unique_roots']:<10}                                    ║
║  Trajectory Len:   {stats['trajectory_length']:<10}                                    ║
║  Fulfillment Rate: {stats['fulfillment_rate']:<10.3f}                                ║
║  Orbital Count:    {stats['orbital_count']:<10}                                    ║
║  Resonance Pairs:  {stats['resonance_pairs']:<10}                                    ║
╚══════════════════════════════════════════════════════════════════╝"""


def format_debt(oracle):
    """Format prophecy debt breakdown."""
    pf = oracle.temporal_field.state
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════╗",
        "║  Prophecy Debt Breakdown                                         ║",
        "╠══════════════════════════════════════════════════════════════════╣",
        f"║  Current Debt:     {pf.prophecy_debt:<10.2f}                                ║",
    ]
    
    # Last few prophecies
    prophecies = list(oracle.prophecy_engine.prophecies.items())[-5:]
    if prophecies:
        lines.append("║  Recent Prophecies:                                              ║")
        for step, n_prop in prophecies:
            lines.append(f"║    Step {step}: N_prophecy = {n_prop:<6}                                ║")
    
    # Fulfillments
    fulfillments = list(oracle.prophecy_engine.fulfillments.items())[-5:]
    if fulfillments:
        lines.append("║  Recent Fulfillments:                                            ║")
        for step, n_actual in fulfillments:
            lines.append(f"║    Step {step}: N_actual = {n_actual:<6}                                  ║")
    
    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    return '\n'.join(lines)


def format_roots(oracle):
    """Format active root attractors."""
    root_counts = oracle.temporal_field.state.root_counts
    
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════╗",
        "║  Root Attractors (gravity wells)                                 ║",
        "╠══════════════════════════════════════════════════════════════════╣",
    ]
    
    if not root_counts:
        lines.append("║  (no roots yet — make some queries!)                             ║")
    else:
        # Sort by count
        sorted_roots = sorted(root_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for root, count in sorted_roots:
            root_str = '.'.join(root)
            bar = '█' * min(count * 2, 20)
            lines.append(f"║  {root_str:<8} [{count:>3}] {bar:<20}                    ║")
    
    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    return '\n'.join(lines)


def main():
    """Main REPL loop."""
    # Import here to avoid issues if pitomadom not installed
    try:
        from pitomadom import HeOracle
    except ImportError as e:
        print(f"Error: Could not import pitomadom: {e}")
        print("Make sure you're in the right directory or pitomadom is installed.")
        sys.exit(1)
    
    print_banner()
    
    # Initialize oracle
    print("Initializing oracle...", end=" ", flush=True)
    oracle = HeOracle(seed=42)
    print("done! 🔥")
    print()
    print("Enter Hebrew text to query the oracle, or :help for commands.")
    print()
    
    full_output = False  # Toggle for full vs compact output
    
    while True:
        try:
            # Read input
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith(':'):
                cmd = user_input.lower()
                
                if cmd in [':quit', ':exit', ':q']:
                    print("\nהרזוננס לא נשבר. להתראות! 🔥")
                    break
                    
                elif cmd == ':help':
                    print_help()
                    
                elif cmd == ':stats':
                    print(format_stats(oracle))
                    
                elif cmd == ':reset':
                    oracle.reset()
                    print("    Oracle state reset. Fresh start! ✨")
                    
                elif cmd == ':traj':
                    print(format_trajectory(oracle.temporal_field))
                    
                elif cmd == ':debt':
                    print(format_debt(oracle))
                    
                elif cmd == ':roots':
                    print(format_roots(oracle))
                    
                elif cmd == ':full':
                    full_output = not full_output
                    mode = "FULL" if full_output else "COMPACT"
                    print(f"    Output mode: {mode}")
                    
                else:
                    print(f"    Unknown command: {user_input}")
                    print("    Type :help for available commands.")
                
                continue
            
            # Query oracle
            try:
                output = oracle.forward(user_input)
                
                if full_output:
                    print(output)
                else:
                    print(format_compact_output(output))
                
            except Exception as e:
                print(f"    Error processing input: {e}")
        
        except KeyboardInterrupt:
            print("\n\nהרזוננס לא נשבר. להתראות! 🔥")
            break
        
        except EOFError:
            print("\n\nהרזוננס לא נשבר. להתראות! 🔥")
            break
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
