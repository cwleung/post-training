#!/usr/bin/env python3
"""
Notebook Auditor Script.
Automated utility to inspect Jupyter notebooks for:
- Syntax & compilation integrity of all code cells
- Single-GPU pinning (CUDA_VISIBLE_DEVICES)
- Dynamic precision detection (is_bfloat16_supported)
- Reward function signature and extraction handlers
"""

import sys
import json
import os

def audit_notebook(notebook_path: str):
    if not os.path.exists(notebook_path):
        print(f"❌ Error: File '{notebook_path}' not found.")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"❌ Error parsing JSON in '{notebook_path}': {e}")
            return False

    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]

    print(f"\n{'='*70}")
    print(f"🔍 AUDITING NOTEBOOK: {notebook_path}")
    print(f"{'='*70}")
    print(f"Total Cells:    {len(cells)} ({len(code_cells)} code, {len(md_cells)} markdown)")

    # 1. Syntax & Compilation Check
    errors = []
    for i, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        # Filter out shell commands (e.g., !pip install)
        clean_lines = [line for line in src.split("\n") if not line.strip().startswith("!")]
        clean_code = "\n".join(clean_lines)
        try:
            compile(clean_code, f"code_cell_{i}", "exec")
        except SyntaxError as e:
            errors.append(f"Cell {i} SyntaxError: {e}")

    if errors:
        print(f"\n❌ Compilation Failures ({len(errors)}):")
        for err in errors:
            print(f"   • {err}")
    else:
        print("✅ Compilation: All code cells compiled without syntax errors.")

    # 2. Check for Essential Guardrails
    all_code = "\n".join("".join(c.get("source", [])) for c in code_cells)

    has_gpu_pin = "CUDA_VISIBLE_DEVICES" in all_code
    has_dynamic_prec = "is_bfloat16_supported" in all_code
    has_completion_extractor = "extract_completion_text" in all_code or "conversational" in all_code.lower()
    has_grpo = "GRPOTrainer" in all_code

    print("\n📋 Guardrail Checks:")
    print(f"   [{'✅' if has_gpu_pin else '⚠️'}] GPU Pinning (CUDA_VISIBLE_DEVICES)")
    print(f"   [{'✅' if has_dynamic_prec else '⚠️'}] Dynamic Precision (fp16/bf16 auto-detection)")
    print(f"   [{'✅' if has_completion_extractor else '⚠️'}] Completion Text Extraction Handler")
    print(f"   [{'✅' if has_grpo else '⚠️'}] GRPOTrainer Integration")

    all_passed = len(errors) == 0 and has_gpu_pin and has_dynamic_prec
    print(f"\nOverall Audit Status: {'PASSED ✅' if all_passed else 'NEEDS ATTENTION ⚠️'}")
    print(f"{'='*70}\n")
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python notebook_auditor.py <path_to_notebook.ipynb>")
        sys.exit(1)
    
    success = True
    for path in sys.argv[1:]:
        if not audit_notebook(path):
            success = False
    
    sys.exit(0 if success else 1)
