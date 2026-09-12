# simulation (frontend/src/entities/simulation)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
24 interactive visual simulation labs modeling RL algorithms, agent harness dynamics, state-space control, and evaluation telemetry.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `CartPoleSimulator.tsx` | TSX Component | Real-time inverted pendulum physics simulator with state vector controls |
| `GrpoSimulator.tsx` | TSX Component | Group Relative Policy Optimization reward ranking and group advantage simulator |
| `PpoSimulator.tsx` | TSX Component | Proximal Policy Optimization clipped surrogate objective and ratio visualizer |
| `VramSimulator.tsx` | TSX Component | GPU memory allocation calculator (KV cache, optimizer states, activations) |
| `GenericInteractiveLab.tsx` | TSX Component | Fallback configurable interactive sandbox for algorithmic parameter tuning |
| `SimulationModal.tsx` | TSX Component | Modal wrapper launching full-screen simulation canvases |
| `labCatalog.ts` | TS Module | Registry mapping chapter IDs to interactive lab configurations |
| `index.ts` | TS Module | Simulation exports and factory helpers |

## Invariants & Rules
- All simulations must be client-side executable without requiring backend GPU resources.
