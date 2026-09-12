# 📋 [Asset Name] Evaluation Rubric

**Target Asset / Project**: [Insert Asset Name e.g., Agentic RLVR Pipeline / Web App / ML Model]  
**Target Platform / Environment**: [e.g., Kaggle T4 GPU / AWS Lambda / Local Docker]  
**Date**: [YYYY-MM-DD]  
**Auditor**: [Agent / Pair Programmer]

---

## 🏆 Evaluation Dimensions (100 Points Total)

| Dimension | Max Points | Evaluation Criteria | Pass/Fail Standard |
| :--- | :---: | :--- | :--- |
| **1. Environment & Hardware Compatibility** | **15** | Platform isolation, device pinning, memory bounds, dynamic precision. | Zero device mismatch errors; memory fits hardware budget. |
| **2. Architecture & Design Integrity** | **15** | Modularity, clean separation of concerns, structured schema enforcement. | Adheres to declared schemas and structured formats. |
| **3. Functional Correctness & Verifiers** | **25** | Deterministic verifiability, error handling, edge cases, input normalization. | Verifiers return accurate, cheat-proof scores on all test cases. |
| **4. Optimization & Training Dynamics** | **20** | Hyperparameter calibration, stability, loss/reward curves, convergence. | Loss decreases, reward increases, KL remains stable. |
| **5. Testing Rigor & Validation Harness** | **15** | Out-of-sample generalization, baseline comparison, quantitative metrics. | Objective metrics tracked on held-out dataset. |
| **6. Code Safety, Sandbox & Usability** | **10** | Security guards, safe execution sandbox, self-contained documentation. | No unhandled exceptions; sandbox prevents malicious execution. |

---

## 🎯 Scoring & Grading Scale

- **A+ (95–100 pts)**: Production Ready. Flawless execution, resilient to edge cases.
- **A  (85–94 pts)**: High Quality. Fully functional with minor cosmetic improvements.
- **B  (70–84 pts)**: Acceptable with Reservations. Fragile on some edge cases.
- **F  (< 70 pts)**: Critical Failure. Requires significant refactoring before deployment.
