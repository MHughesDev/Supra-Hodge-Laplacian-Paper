# Supra-Hodge Laplacians for Multi-View Reasoning

Reference code for the **Supra-Hodge Laplacian** on coupled simplicial complexes: a block operator that combines layerwise Hodge Laplacians with cross-view coupling, together with a small **end-to-end reasoning pipeline** that turns spectral diagnostics into optional **OpenAI** chat context.

## Contents

| File | Description |
|------|-------------|
| `supra_hodge_reasoning_pipeline.py` | Dense linear algebra (including a Jacobi symmetric eigensolver), boundary operators, Hodge and Supra-Hodge matrices, simplicial lifting, an eight-node mock planning scenario, and HTTP calls to the OpenAI Chat Completions API via the standard library only. |

The mathematical definitions follow the accompanying research write-up (graph and supra-Laplacians, Hodge operators \(H_p = B_p^\top B_p + B_{p+1} B_{p+1}^\top\), Supra-Hodge blocks, energy identity, and the simplicial / LLM pipeline described there).

## Requirements

- **Python 3.9+** (tested with recent 3.x)
- **No required third-party packages** for the core math and demo: only the Python standard library (`json`, `urllib`, etc.).

## Quick start

From the repository root:

```bash
python supra_hodge_reasoning_pipeline.py
```

This prints:

- A **two-layer** (semantic vs. evidential) comparison and diagnostics.
- A **four-layer** eight-node run.
- A numerical check that the **Supra-Hodge energy** matches the block expansion (Equation (9) in the paper).
- A **spectral augmentation** block and either a **mock** LLM reply or a **live** OpenAI response (see below).

## OpenAI API (optional)

**Recommended:** put your key in a `.env` file at the project root (same folder as `supra_hodge_reasoning_pipeline.py`). The script loads it automatically and does not override keys already set in the shell.

```
OPENAI_API_KEY=sk-...
```

The real `.env` file is listed in `.gitignore` so it is not committed. You can start from `.env.example` if you need a template.

You can still set the variable for one session instead:

**Windows (PowerShell)**

```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
python supra_hodge_reasoning_pipeline.py
```

**macOS / Linux**

```bash
export OPENAI_API_KEY="your-api-key-here"
python supra_hodge_reasoning_pipeline.py
```

If `OPENAI_API_KEY` is not set, the script skips the network request and prints a short placeholder assistant message instead.

Default model in code: `gpt-4o-mini` (change the `model` argument in `supra_hodge_pipeline` / `openai_chat_completion` if you prefer another chat model).

## Design notes

- **Eigenvalues and eigenvectors** are computed with a **hand-written Jacobi method** for real symmetric matrices (no NumPy or SciPy).
- **Mock data** (embeddings, edge signals) is **fabricated** for demonstration; it is not tied to a production retrieval or agent trace.
- Reported numerical examples in the paper (for example \(\mu_2\) before and after an evidential update) depend on exact lifts and couplings; the script explains when local updates have a small effect on global \(\lambda_2\) under the default synthetic parameters.

## License

Add a `LICENSE` file if you plan to distribute the project; none is bundled here by default.
