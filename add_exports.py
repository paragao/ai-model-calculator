#!/usr/bin/env python3
"""
Add export function calls to model_calculations.py to enable CSV/JSON output.
"""

# Read the original file
with open('model_calculations.py', 'r') as f:
    content = f.read()

# Define the insertions (search pattern -> insertion)
insertions = [
    # Phase 1
    ('else:\n    print("  ✅ All configurations look good!")\n\n# ============================================================================\n# PHASE 2:',
     'else:\n    print("  ✅ All configurations look good!")\n\n# Export Phase 1 results\nexport_results_csv(all_results, "phase1_memory_results.csv")\nexport_results_json(all_results, "phase1_memory_results.json")\nprint(f"\\n📊 Phase 1 results exported to: phase1_memory_results.csv, phase1_memory_results.json")\n\n# ============================================================================\n# PHASE 2:'),

    # Phase 2
    ('else:\n    print("  ✅ All batch configurations are optimal!")\n\n# ============================================================================\n# PHASE 3:',
     'else:\n    print("  ✅ All batch configurations are optimal!")\n\n# Export Phase 2 results\nexport_batch_results_csv(batch_results, "phase2_batch_results.csv")\nprint(f"\\n📊 Phase 2 results exported to: phase2_batch_results.csv")\n\n# ============================================================================\n# PHASE 3:'),

    # Phase 3
    ('else:\n    print("  ✅ All training time estimates look reasonable!")\n\n# ============================================================================\n# PHASE 4: ZeRO-2 Communication',
     'else:\n    print("  ✅ All training time estimates look reasonable!")\n\n# Export Phase 3 results\nexport_training_results_csv(training_results, "phase3_training_results.csv")\nprint(f"\\n📊 Phase 3 results exported to: phase3_training_results.csv")\n\n# ============================================================================\n# PHASE 4: ZeRO-2 Communication'),

    # Phase 4
    ('else:\n    print("  ✅ All ZeRO-2 communication patterns look optimal!")\n\n# ============================================================================\n# PHASE 4.1: ZeRO-1 Communication',
     'else:\n    print("  ✅ All ZeRO-2 communication patterns look optimal!")\n\n# Export Phase 4 results\nexport_communication_results_csv(comm_results, "phase4_zero2_comm_results.csv")\nprint(f"\\n📊 Phase 4 (ZeRO-2) results exported to: phase4_zero2_comm_results.csv")\n\n# ============================================================================\n# PHASE 4.1: ZeRO-1 Communication'),

    # Phase 4.1
    ('else:\n    print("  ✅ All ZeRO-1 communication patterns look optimal!")\n\n# ============================================================================\n# PHASE 5: All-to-all Communication',
     'else:\n    print("  ✅ All ZeRO-1 communication patterns look optimal!")\n\n# Export Phase 4.1 results\nexport_zero1_results_csv(zero1_results, "phase4_1_zero1_comm_results.csv")\nprint(f"\\n📊 Phase 4.1 (ZeRO-1) results exported to: phase4_1_zero1_comm_results.csv")\n\n# ============================================================================\n# PHASE 5: All-to-all Communication'),

    # Phase 5 - at end of file
    ('else:\n    print("  ✅ All all-to-all communication patterns look optimal!")',
     'else:\n    print("  ✅ All all-to-all communication patterns look optimal!")\n\n# Export Phase 5 results\nexport_alltoall_results_csv(a2a_results, "phase5_alltoall_comm_results.csv")\nprint(f"\\n📊 Phase 5 results exported to: phase5_alltoall_comm_results.csv")\nprint(f"\\n✅ All results exported successfully!")')
]

# Apply replacements
for old, new in insertions:
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Added export for: {old[:40]}...")
    else:
        print(f"⚠ Pattern not found: {old[:40]}...")

# Write the modified file
with open('model_calculations.py', 'w') as f:
    f.write(content)

print("\n✅ Export calls added successfully!")
print("\nGenerated files:")
print("  - phase1_memory_results.csv & .json")
print("  - phase2_batch_results.csv")
print("  - phase3_training_results.csv")
print("  - phase4_zero2_comm_results.csv")
print("  - phase4_1_zero1_comm_results.csv")
print("  - phase5_alltoall_comm_results.csv")
