#!/usr/bin/env python3
"""
Add export function calls to model_calculations.py to enable CSV/JSON output.
This script modifies model_calculations.py in place to add export calls after each phase.
"""

import sys

# Read the file
with open('model_calculations.py', 'r') as f:
    lines = f.readlines()

# Define the exports to add after each phase
exports = {
    'print("  ✅ All configurations look good!")': '''
# Export Phase 1 results
export_results_csv(all_results, "phase1_memory_results.csv")
export_results_json(all_results, "phase1_memory_results.json")
print(f"\\n📊 Phase 1 results exported to: phase1_memory_results.csv, phase1_memory_results.json")
''',
    '    print("  ✅ All batch configurations are optimal!")': '''
# Export Phase 2 results
export_batch_results_csv(batch_results, "phase2_batch_results.csv")
print(f"\\n📊 Phase 2 results exported to: phase2_batch_results.csv")
''',
    '    print("  ✅ All training time estimates look reasonable!")': '''
# Export Phase 3 results
export_training_results_csv(training_results, "phase3_training_results.csv")
print(f"\\n📊 Phase 3 results exported to: phase3_training_results.csv")
''',
    '    print("  ✅ All ZeRO-2 communication patterns look optimal!")': '''
# Export Phase 4 results
export_communication_results_csv(comm_results, "phase4_zero2_comm_results.csv")
print(f"\\n📊 Phase 4 (ZeRO-2) results exported to: phase4_zero2_comm_results.csv")
''',
    '    print("  ✅ All ZeRO-1 communication patterns look optimal!")': '''
# Export Phase 4.1 results
export_zero1_results_csv(zero1_results, "phase4_1_zero1_comm_results.csv")
print(f"\\n📊 Phase 4.1 (ZeRO-1) results exported to: phase4_1_zero1_comm_results.csv")
''',
    '    print("  ✅ All all-to-all communication patterns look optimal!")': '''
# Export Phase 5 results
export_alltoall_results_csv(a2a_results, "phase5_alltoall_comm_results.csv")
print(f"\\n📊 Phase 5 results exported to: phase5_alltoall_comm_results.csv")
'''
}

# Process the file
modified = False
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)

    # Check if this line matches any of our patterns
    for pattern, export_code in exports.items():
        if pattern in line:
            # Check if export is already added
            if i + 1 < len(lines) and 'Export Phase' not in lines[i + 1]:
                new_lines.append(export_code)
                modified = True
                print(f"✓ Added export after: {pattern[:50]}...")
            break

    i += 1

if modified:
    # Write the modified file
    with open('model_calculations.py', 'w') as f:
        f.writelines(new_lines)
    print(f"\n✅ Successfully added {sum(1 for v in exports.values() if any(v in ''.join(new_lines)))} export calls to model_calculations.py")
    print("\nYou can now run:")
    print("  python3 model_calculations.py")
    print("\nThis will generate CSV files for each phase:")
    print("  - phase1_memory_results.csv")
    print("  - phase2_batch_results.csv")
    print("  - phase3_training_results.csv")
    print("  - phase4_zero2_comm_results.csv")
    print("  - phase4_1_zero1_comm_results.csv")
    print("  - phase5_alltoall_comm_results.csv")
else:
    print("ℹ️  Export calls already present or patterns not found")
    sys.exit(1)
