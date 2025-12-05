Short answer: your code captures **almost all of the process equations (1–8, 16–22, 23–29, 30)** but **drops / simplifies the “engineering wrapper” of the model** and some structural details, especially around the schmutzdecke, permeability and calibration. Here’s the breakdown.

---

### 1. Schmutzdecke structure & layering (Eqs. 9–15)

What the paper does

* Treats the schmutzdecke as **multiple layers** that appear after a lag phase and then grow sequentially with a defined “growing time” Tgrow.
* Uses Eqs. (9), (11), (13–15) to compute:

  * When each new layer starts,
  * Daily % growth ϕ based on total biomass over the whole run,
  * Evolving **porosity of each layer** ε₀s(i,j),
  * Number of layers NL for a given run. 

What your code does

* Uses a **single lumped schmutzdecke layer**:

  * `SB` is updated by `SB += Ain * dt_hours` (Eq. 9 in spirit, but no layer index j, no Tlag/Tgrow logic).
  * `eps_s` is updated with a ϕ derived from the incremental dSB only, not from the global relation of Eq. (14).
  * `dLs` is computed once from Eq. (10) using the *current* SB and a single ε₀s, but there is no stack of layers and no explicit lag / growing-time schedule.

So: **layering, explicit lag phase, and the NL logic (Eqs. 14–15)** are not represented, only a simplified “one effective layer” version.

---

### 2. Full headloss / permeability treatment (Eqs. 20–22 & 21)

What the paper does

1. **Breakdown of bulk specific deposit** (Eq. 20)

   * σ_tot(L,t) = σ_I + σ_Cp + σ_X + σ_A + σ_P
     (inert, POC, bacteria, algae, protozoa each contribute separately).

2. **Explicit permeability of schmutzdecke** via Happel’s fibrous-medium model (Eq. 21), then clean-bed headloss from Darcy’s law.

3. **Headloss model**

   * Uses Sembi & Ives (Eq. 22) with H(L,1)/L and H(L,0)/L explicitly, and applies different treatment for schmutzdecke vs sand because ε₀s and σ for the schmutzdecke are evolving.

What your code does

* Uses **one bulk σ** computed from σ_a via `sigma = b_bulk * sigma_a` and drops the decomposition (no σ_I, σ_Cp, σ_X, σ_A, σ_P individually).
* Does **not compute permeability k** from Eq. (21) nor initial clean-bed headloss H(L,0) from Darcy’s law; instead you work directly with a *relative* headloss ratio:

  ```python
  H_ratio = [1 + σ/(1−ε0)]^c1 * [ε0/(ε0−σ)]^c2
  ```

  i.e. the multiplicative factor from Eq. (22), but not the absolute H(L,t).
* Uses a **constant u = q** rather than u = q/ε where ε = ε0 − σ as in the paper.

So: the **permeability model, absolute headloss calculation and deposition-component breakdown** are not fully implemented; you only use a compacted version of Eq. (22) with a simplified σ and constant u.

---

### 3. Covered vs uncovered filter configurations

What the paper does

* Explicitly distinguishes **covered filters** (no supernatant algae growth, no schmutzdecke, only sand-bed processes) from **uncovered filters** (supernatant + schmutzdecke + sand bed). 

What your code does

* Hard-wires the **uncovered filter** case:

  * Always computes supernatant algae + nutrients (Eqs. 1–8),
  * Always runs schmutzdecke module,
  * No switch to “covered” mode.

So: the **covered-filter configuration path** discussed in the paper is not represented.

---

### 4. Multiple runs, cleaning and initial/boundary conditions

What the paper does

* Simulates **sequential filtration runs** (Runs 1–4, etc.):

  * After scraping/removal of top sand, the deeper σ and λ field from the previous run is used as the **initial condition** for the next run. 
* Includes explicit handling of **clean-bed initial conditions** vs “post-cleaning” conditions, and boundary conditions from measured influent time series.

What your code does

* Implements a **single continuous run** of length `cfg.n_steps` with:

  * Clean initial bed (`sigma_a = 0`, `beta = beta0_sand`),
  * Constant influent conditions (`infl` is time-invariant).
* No logic to:

  * Remove top X cm of sand and remap σ/β fields,
  * Carry σ/β over between runs,
  * Use different influent time series per run.

So: the **multi-run operation and cleaning/reset dynamics** from the paper are not included.

---

### 5. Calibration, sensitivity analysis, temperature-scaling of *parameters*

What the paper does

* Full **a priori sensitivity analysis** with sensitivity coefficients SC for each parameter.
* **Monte-Carlo calibration** (300 runs) using SSE (Eq. 35) on headloss data, selecting the 10 best parameter sets; discussion of parameter ranges (Tables 3–5). 
* **Verification and seasonal adjustment**:

  * Uses temperature to adjust *parameters* c₁, c₂, b and A_in between runs (Eq. 36).

What your code does

* Reads a **single fixed parameter set** from Excel:

  * No sensitivity analysis,
  * No SSE or automatic calibration loop,
  * No Monte-Carlo or multi-parameter search,
  * No Eq. (36)-style temperature scaling of c₁, c₂, b, Ain.
* `k_temp` is used **only for kinetic rates**, not for headloss parameters.

So: all of the **calibration / sensitivity / verification machinery** in the paper is deliberately omitted; you’ve implemented a “forward model”, not the full calibration framework.

---

### 6. Numerical-method constraints & stability criteria

What the paper does

* Uses forward Euler but explicitly states constraints like:

  * Δt cannot be so large that u Δt > ΔL,
  * Δt = 0.1 h and ΔL = 1 cm chosen as compromise between speed and accuracy. 

What your code does

* Uses dt from Excel (`cfg.dt`) with **no built-in CFL/stability check**:

  * No enforcement of u·dt ≤ ΔL,
  * No dynamic time-step adjustment.

So: the **numerical-stability criteria** described in the “Numerical Solution of Equations” section are not encoded; they are left to the user.

---

### 7. Spatial details: top 2 cm vs one cell, schmutzdecke biology

What the paper does

* Distinguishes clearly between:

  * **Top 2 cm** of sand (receiving schmutzdecke-derived carbon, with Eq. 25 & modified DOC uptake factor 2/41 + SBc term),
  * Deeper layers (Eq. 24 with 40/41 DOC, 1/41 P, no SBc). 
* Applies microbial ODEs **within both schmutzdecke and sand** layers.

What your code does

* Approximates the top 2 cm by **the first grid cell**:

  * `is_top_layer = (j == 0)` with factors 2/41 and 38/41, which is conceptually correct but depends on ΔL being ≈ 2 cm; it’s not explicitly tied to 2 cm.
* You **do not carry separate state variables for schmutzdecke biomass inside the pores** (algae/bacteria/protozoa as a separate layer); SBc is a lumped m⁻² quantity used only in the top sand layer.

So: the **exact 2 cm geometric definition and explicit schmutzdecke “internal” biology as separate grid cells** are simplified / collapsed into the first sand layer.

---

### 8. Data-handling / pilot-plant specifics

Not captured in your code:

* Pilot-plant configuration at Kempton Park (two uncovered + one covered SSF, upstream processes, real time-series influent).
* Run-specific depths, temperatures, cleanup policies, etc.
* Verification on SSF A using parameters from SSF B.

These are **experimental/operational details**, not core equations; your implementation is intentionally generic and uses a single Excel workbook instead.

---

### Bottom line

Your script is a **deterministic forward simulator** that:

* Implements the **core process ODE/PDE structure** (Eqs. 1–8, 16–19, 22–29, 30) and a simplified schmutzdecke thickness / porosity update.
* **Does not implement**:

  * Multilayer schmutzdecke growth with lag & growing time (Eqs. 9, 11, 13–15 in full),
  * Happel permeability / absolute headloss calculation & detailed σ-component breakdown (Eqs. 20–21),
  * Covered-filter configuration,
  * Multi-run cleaning & state carry-over,
  * Sensitivity analysis, SSE-based Monte-Carlo calibration, seasonal temperature-correction on c₁, c₂, b, A_in,
  * Numerical-stability safeguards and some of the geometric detail (exact 2 cm region, explicit schmutzdecke-bed microbial grid).

If you want, I can mark these as TODO sections directly in the code with references to the exact equations they correspond to, without touching the logic that already works.
