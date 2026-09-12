---
name: rubric-evaluation
description: >-
  Systematic multi-dimensional rubric evaluation, auditing, and self-improving workflow.
  Use when evaluating codebases, notebooks, machine learning pipelines, agent systems,
  architectures, or technical deliverables against strict quality criteria, scoring quality,
  identifying defects, and iteratively improving assets to production grade.
---

# 📋 Rubric-Driven Evaluation & Self-Correction Workflow

This skill provides a systematic, repeatable framework for **evaluating, scoring, auditing, and iteratively improving** complex technical deliverables (codebases, notebooks, machine learning systems, agent architectures, APIs, and documentation).

---

## 🔄 The 5-Phase Rubric Evaluation Lifecycle

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 1. Rubric       │ ──▶ │ 2. Audit &      │ ──▶ │ 3. Dynamic      │
│    Design       │     │    Static Check │     │    Verification │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              ▼
│ 5. Audit Report │ ◀── │ 4. Fix &        │ ◀────────────┘
│    & Packaging  │     │    Iterate      │
└─────────────────┘     └─────────────────┘
```

---

## 📐 Phase 1: Establish the Evaluation Rubric

Before evaluating any deliverable, define a domain-specific, weighted **100-Point Rubric** across 5–6 distinct dimensions:

### Standard Dimension Categories

| Dimension | Typical Weight | Focus Area |
| :--- | :---: | :--- |
| **1. Environment & Hardware Compatibility** | 10–15% | Runtime isolation, device pinning, memory bounds, dependency safety. |
| **2. Architecture & Design Integrity** | 15–20% | Modular structure, separation of concerns, schema adherence. |
| **3. Functional Correctness & Edge Cases** | 20–25% | Deterministic verifiability, error handling, boundary conditions. |
| **4. Performance, Optimization & Efficiency** | 15–20% | Resource budget, algorithmic complexity, latency, batching. |
| **5. Testing Rigor & Validation Harness** | 15–20% | Out-of-sample testing, pre/post baseline comparisons, reproducibility. |
| **6. Code Quality, Safety & Usability** | 10–15% | Documentation, sandbox isolation, security, interactive ergonomics. |

### Quality Thresholds
- **A+ (95–100 pts)**: Production-ready; robust against edge cases; zero syntax or runtime errors.
- **A (85–94 pts)**: High quality; minor optimizations or non-critical formatting improvements needed.
- **B (70–84 pts)**: Functional but fragile; missing validation splits, unhandled edge cases, or potential memory leaks.
- **F (< 70 pts)**: Critical failures; runtime exceptions, incompatible dependencies, or unverifiable outputs.

---

## 🔍 Phase 2: Static & Automated Inspection

Perform programmatic static analysis on all files:

1. **Compilation & Syntax Validation**:
   - Python files: `python -m py_compile src/*.py`
   - Jupyter Notebooks: Extract all code cells and compile with `compile(cell_source, 'cell', 'exec')`.
2. **Environment & Hardware Guardrails**:
   - Check device isolation (e.g., `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`).
   - Check dynamic precision auto-detection (`is_bfloat16_supported()`).
3. **Dependency & Import Audit**:
   - Verify all imported packages are explicitly declared in requirements or install commands.
   - Provide safe fallback imports where necessary (e.g., standard library fallbacks).

---

## ⚡ Phase 3: Dynamic Verification & Edge-Case Testing

Execute and stress-test the core logic against real and adversarial inputs:

1. **Happy Path Testing**: Ensure core functionality produces expected outputs on standard inputs.
2. **Edge-Case Matrix**:
   - **Type Variations**: Strings vs. lists vs. dictionaries vs. objects.
   - **Numerical Edge Cases**: Float precision (`abs(a - b) < tol`), thousands separators (`,`), currency symbols (`$`, `€`), null/zero values.
   - **Schema Violations**: Malformed JSON, missing required keys, invalid types, empty completions.
3. **Sandbox Security Check**:
   - Ensure external/untrusted code runs in isolated environments with safe builtins and restricted filesystem/network access.

---

## 🛠️ Phase 4: Self-Correction & Iterative Improvement

Whenever a gap, fragility, or defect is identified:

1. **Root-Cause Analysis**: Document exactly why the failure occurred.
2. **Apply Targeted Fixes**: Modify source files, notebook cells, or reward functions.
3. **Re-Run Dynamic Verification**: Execute automated checks to confirm the defect is 100% resolved without regression.
4. **Log the Resolution**: Record the before/after change for the final audit report.

---

## 📊 Phase 5: Generate the Rubric Audit Report

Document the complete evaluation as an artifact (`rubric_evaluation.md`) with:

1. **Executive Scorecard**: Score per dimension + Total Score (out of 100) + Final Grade.
2. **Dimension-by-Dimension Breakdown**: Strengths, evidence, and points awarded.
3. **Issues Identified & Upgrades Applied**: Concrete list of bugs discovered and the exact code fixes applied.
4. **Verification Proof**: Command outputs and assertion logs confirming 100% passing status.

---

## 💡 Best Practices

- **Never Evaluate Subjectively**: Ground every point deduction in concrete test results, missing safeguards, or edge-case failures.
- **Separate In-Sample from Out-of-Sample**: Always evaluate models or pipelines on held-out test splits to measure generalization.
- **Automate the Auditor**: Use Python scripts to parse notebook JSON, test functions, and measure execution metrics automatically.
