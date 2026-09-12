# privacy (frontend/src/widgets/privacy)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Discrete authorization dialogs for developer mode and advanced diagnostic settings.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `CareerUnlockModal.tsx` | TSX Component | Minimalist developer mode authentication dialog with SHA-256 verification and one-click locking |

## Invariants & Rules
- Plaintext passphrases must never be stored or evaluated in plaintext. Only one-way cryptographic SHA-256 digests are verified.
- State is synced with `usePrivacyStore`.
