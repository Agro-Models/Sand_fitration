
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


# =============================================================================
# CAMPOS ET AL. (2006) SLOW SAND FILTER MODEL – IMPLEMENTATION OVERVIEW
# =============================================================================
# Paper: Campos, L. C., Smith, S. R., & Graham, N. J. D. (2006)
# "Deterministic-based model of slow sand filtration. I: Model development."
#
# -------------------------------------------------------------------------
# STATE VARIABLES
# -------------------------------------------------------------------------
# Supernatant (well-mixed water above sand bed):
#   a_sup   [mg Chla L^-1]  – chlorophyll-a (algae)
#   ps_sup  [mg P   L^-1]   – soluble phosphorus
#   na_sup  [mg N   L^-1]   – ammonium nitrogen (NH4-N)
#   ni_sup  [mg N   L^-1]   – nitrate  nitrogen (NO3-N)
#
# Depth-resolved profiles (for each bed layer j = 0..nz-1):
#   Cp[j]       [mg L^-1]   – suspended solids in pore water (C in paper)
#   sigma_a[j]  [vol vol^-1 or mg L^-1 eq.] – absolute specific deposit σ_a
#   sigma[j]    [vol vol^-1] – bulk specific deposit σ = b σ_a (Eq. 19)
#   beta[j]     [m^-1]      – filtration coefficient λ (called β here)
#   H_ratio[j]  [-]         – relative headloss H(L,t) / H(L,0) (Eq. 22)
#
# In-bed biology (per depth cell j):
#   a_bed[j] [mg Chla L^-1] – algae attached in bed
#   x_bed[j] [mg C    L^-1] – bacteria biomass
#   p_bed[j] [mg C    L^-1] – protozoa biomass
#   cp_bed[j][mg C    L^-1] – nonliving POC (particulate organic carbon)
#   cd_bed[j][mg C    L^-1] – DOC (dissolved organic carbon)
#   ps_bed[j][mg P    L^-1] – phosphorus in bed
#
# Schmutzdecke (surface layer treated as lumped + mapped to grid cells):
#   SB   [mg Chla m^-2] – schmutzdecke biomass as chlorophyll-a
#   SBc  [mg C    m^-2] – schmutzdecke biomass as carbon
#   dLs  [m]            – schmutzdecke thickness (Eq. 10)
#   eps_s[-]            – schmutzdecke porosity (Eqs. 11–13)
#
# -------------------------------------------------------------------------
# PARAMETERS – ALL READ FROM EXCEL (NO NUMERIC DEFAULTS IN CODE)
# -------------------------------------------------------------------------
# Sheet "SSFConstants" (Table 2: physical parameters, stoichiometry, θ):
#   g, nu, q, Af, L0, Da, df, Tgrow_h, Tlag_h,
#   Abw, Aca, Ana, Apc, Apa, Awc, Awa,
#   theta_kga, theta_kgb, theta_kgp,
#   theta_kra, theta_krb, theta_krp,
#   theta_Cgz, theta_kp, theta_kh,
#   theta_kn, theta_kdb, theta_kdp
#
# Sheet "SSFCalibrationParams" (Table 3: initial ranges / averages):
#   beta0_sand, beta0_schmutz, b_bulk,
#   a1, a2, c1, c2, Ain, eps0, eps0_s,
#   kdb20, kdp20, kp20, kh20, kn20,
#   kgmaxa20, kgmaxb20, kgmaxp20,
#   kra20, krb20, krp20,
#   kscd, ksa, ksb, ksp, ksn,
#   ksl, fd, Im, kew, kam, Eh,
#   Cgz20, Yb, Yp
#
# Sheet "SSFConfig" (numerical configuration):
#   nz, dt, n_steps, is_covered (0/1)
#
# Sheet "SSFInfluent" (influent water quality and operating conds.):
#   T, Nv, D, z, Cp_inlet_inert, n_tot (optional/legacy)
#
# -------------------------------------------------------------------------
# DISCRETIZATION SUMMARY
# -------------------------------------------------------------------------
# Space:
#   Filter depth interval [0, L0] is divided into nz uniform cells.
#   Let ΔL = L[1] − L[0]. All depth-dependent equations are discretized
#   with first-order upwind in L (for solids) or simple cell values.
#
# Time:
#   All ODEs are integrated using explicit forward Euler:
#       y^{n+1} = y^n + Δt * (dy/dt)(y^n)
#
# Equation-specific discrete forms are written in comments right where
# they are implemented (Eqs. 1–29, 3–5, 16–22, 30).
#
# TODO [Campos Sections 3.3–3.4]:
#   - Build a separate calibration script that:
#       * Samples parameter sets within Table 3 ranges.
#       * Runs this model and computes SSE against observed headloss/
#         turbidity (standard sum of squared errors).
#       * Optionally applies seasonal correction to c1, c2, b_bulk,
#         Ain as per the publication once Eq. (36) is explicitly coded.
# =============================================================================


# expose results to IDE / interactive workspace
df_results = None


# =============================================================================
# DATA STRUCTURES FOR PARAMETERS (ALL LOADED FROM EXCEL)
# =============================================================================

@dataclass
class SSFConstants:
    """Table 2 constants – loaded from Excel sheet 'SSFConstants'."""
    g: float
    nu: float
    q: float
    Af: float
    L0: float
    Da: float
    df: float
    Tgrow_h: float
    Tlag_h: float

    Abw: float
    Aca: float
    Ana: float
    Apc: float
    Apa: float
    Awc: float
    Awa: float

    theta_kga: float
    theta_kgb: float
    theta_kgp: float
    theta_kra: float
    theta_krb: float
    theta_krp: float
    theta_Cgz: float
    theta_kp: float
    theta_kh: float
    theta_kn: float
    theta_kdb: float
    theta_kdp: float

    @classmethod
    def from_excel(cls, path: str, sheet: str = "SSFConstants") -> "SSFConstants":
        df = pd.read_excel(path, sheet_name=sheet)
        values = dict(zip(df["name"], df["value"]))
        return cls(
            g=values["g"],
            nu=values["nu"],
            q=values["q"],
            Af=values["Af"],
            L0=values["L0"],
            Da=values["Da"],
            df=values["df"],
            Tgrow_h=values["Tgrow_h"],
            Tlag_h=values["Tlag_h"],
            Abw=values["Abw"],
            Aca=values["Aca"],
            Ana=values["Ana"],
            Apc=values["Apc"],
            Apa=values["Apa"],
            Awc=values["Awc"],
            Awa=values["Awa"],
            theta_kga=values["theta_kga"],
            theta_kgb=values["theta_kgb"],
            theta_kgp=values["theta_kgp"],
            theta_kra=values["theta_kra"],
            theta_krb=values["theta_krb"],
            theta_krp=values["theta_krp"],
            theta_Cgz=values["theta_Cgz"],
            theta_kp=values["theta_kp"],
            theta_kh=values["theta_kh"],
            theta_kn=values["theta_kn"],
            theta_kdb=values["theta_kdb"],
            theta_kdp=values["theta_kdp"],
        )


@dataclass
class SSFCalibrationParams:
    """Table 3 model parameters – loaded from 'SSFCalibrationParams'.
    NOTE (Seasonal Parameters):
        The following four parameters MUST be changed for wet vs dry runs:
            * b_bulk   – deposit bulk factor (Eq. 19)
            * Ain      – schmutzdecke growth rate (Eq. 9)
            * c1, c2   – headloss exponents (Eq. 22) - less likely applicable to Sri Lanka, but worth the check

        These are treated as constants WITHIN a filtration run,
        but different constant values must be provided for wet-season
        and dry-season simulations (Campos seasonal methodology).

    All other parameters in this class are season-independent.

    This dataclass only stores the parameter values. Calibration / Monte-Carlo
    wrappers should be implemented in a separate script using these fields.
    """
    # Filtration / headloss parameters
    beta0_sand: float
    beta0_schmutz: float
    b_bulk: float
    a1: float
    a2: float
    c1: float
    c2: float
    Ain: float
    eps0: float
    eps0_s: float
    # NEW: daily schmutzdecke porosity growth [%/day], ζ in Eqs. (11–13)
    phi_daily: float  # daily % increase in schmutzdecke porosity

    # Kinetic rates at 20 °C
    kdb20: float
    kdp20: float
    kp20: float
    kh20: float
    kn20: float
    kgmaxa20: float
    kgmaxb20: float
    kgmaxp20: float
    kra20: float
    krb20: float
    krp20: float

    # Saturation / half-saturation / light, grazing etc.
    kscd: float
    ksa: float
    ksb: float
    ksp: float
    ksn: float
    ksl: float
    fd: float
    Im: float
    kew: float
    kam: float
    Eh: float
    Cgz20: float
    Yb: float
    Yp: float

    @classmethod
    def from_excel(cls, path: str,
                   sheet: str = "SSFCalibrationParams") -> "SSFCalibrationParams":
        df = pd.read_excel(path, sheet_name=sheet)
        values = dict(zip(df["name"], df["value"]))
        return cls(
            beta0_sand=values["beta0_sand"],
            beta0_schmutz=values["beta0_schmutz"],
            b_bulk=values["b_bulk"],
            a1=values["a1"],
            a2=values["a2"],
            c1=values["c1"],
            c2=values["c2"],
            Ain=values["Ain"],
            eps0=values["eps0"],
            eps0_s=values["eps0_s"],
            phi_daily=values["phi_daily"],
            kdb20=values["kdb20"],
            kdp20=values["kdp20"],
            kp20=values["kp20"],
            kh20=values["kh20"],
            kn20=values["kn20"],
            kgmaxa20=values["kgmaxa20"],
            kgmaxb20=values["kgmaxb20"],
            kgmaxp20=values["kgmaxp20"],
            kra20=values["kra20"],
            krb20=values["krb20"],
            krp20=values["krp20"],
            kscd=values["kscd"],
            ksa=values["ksa"],
            ksb=values["ksb"],
            ksp=values["ksp"],
            ksn=values["ksn"],
            ksl=values["ksl"],
            fd=values["fd"],
            Im=values["Im"],
            kew=values["kew"],
            kam=values["kam"],
            Eh=values["Eh"],
            Cgz20=values["Cgz20"],
            Yb=values["Yb"],
            Yp=values["Yp"],
        )


@dataclass
class SSFConfig:
    """Numerical configuration – loaded from 'SSFConfig'.

    Fields:
        nz         – number of depth cells
        dt         – time step [h]
        n_steps    – number of steps per run
        is_covered – 1 for covered filter (no supernatant algae/schmutz),
                     0 for uncovered (full supernatant module).
    """
    nz: int
    dt: float
    n_steps: int
    is_covered: bool
    H_tot0: float
    H_stop_m: float  

    @classmethod
    def from_excel(cls, path: str, sheet: str = "SSFConfig") -> "SSFConfig":
        df = pd.read_excel(path, sheet_name=sheet)
        values = dict(zip(df["name"], df["value"]))
        return cls(
            nz=int(values["nz"]),
            dt=float(values["dt"]),
            n_steps=int(values["n_steps"]),
            is_covered=bool(int(values.get("is_covered", 0))),
            H_tot0=float(values["H_tot0"]),
            H_stop_m=float(values["H_stop_m"]),
        )


@dataclass
class SSFInfluent:
    """
    Influent water quality and operating conditions – sheet 'SSFInfluent'.

    For now this is a single set of constants over the whole run, but it is
    kept on a separate sheet so you can easily switch to a time-series later.

    Fields:
        T               – temperature [°C]
        Nv, D, z        – optical parameters for supernatant (Eqs. 3–5)
        Cp_inlet_inert  – inert suspended solids concentration [mg/L] that
                          always enter the bed (both covered/uncovered).
        n_tot           – optional legacy; supernatant module now uses
                          na_sup + ni_sup for nitrogen limitation.
    """
    T: float
    Nv: float
    D: float
    z: float
    Cp_inlet_inert: float
    n_tot: float

    @classmethod
    def from_excel(cls, path: str, sheet: str = "SSFInfluent") -> "SSFInfluent":
        df = pd.read_excel(path, sheet_name=sheet)
        values = dict(zip(df["name"], df["value"]))
        return cls(
            T=float(values["T"]),
            Nv=float(values["Nv"]),
            D=float(values["D"]),
            z=float(values["z"]),
            Cp_inlet_inert=float(values["Cp_inlet_inert"]),
            n_tot=float(values.get("n_tot", 0.0)),
        )


# =============================================================================
# Utility: temperature correction, Eq. (30)
#   k(T) = k20 * θ^(T − 20)
# =============================================================================

def k_temp(k20: float, theta: float, T: float) -> float:
    """
    Eq. (30):  k_T = k_20 * θ^(T − 20)

    Discrete: just evaluated at current T whenever a kinetic rate is needed.
    """
    #return k20 * (theta ** (T - 20.0))
    return k20


# =============================================================================
# Utility: total headloss (Campos-consistent, dimensional)
# =============================================================================


def compute_total_headloss_m(
    H_ratio_layer: np.ndarray,
    dL: float,
    L0: float,
    H_clean_m: float,
) -> tuple[float, float, float]:
    """
    Campos-consistent total headloss calculation (sand + schmutzdecke).

    Inputs:
      H_ratio_layer : array of F(L,t) factors (dimensionless) for each depth cell.
                      This is what your code calls H_ratio_layer / model.H_ratio.
      dL            : depth step [m]
      L0            : total bed depth [m]
      H_clean_m     : clean-bed total headloss H_clean = H_tot(0) [m]

    Returns:
      H_total_m     : absolute total headloss across bed [m]
      H_total_rel   : relative total headloss H_tot(t)/H_tot(0) [-]
      dH0_dL        : clean-bed hydraulic gradient (dH/dL)_0 [m/m]
    """
    H_ratio_layer = np.asarray(H_ratio_layer, dtype=float)

    # Clean-bed hydraulic gradient (dimensional anchor)
    dH0_dL = H_clean_m / (L0 + 1e-30)  # [m/m]

    # Integral of dimensionless factor over depth
    I = float(np.sum(H_ratio_layer) * dL)  # [m]

    # Absolute headloss (meters)
    H_total_m = dH0_dL * I  # [m]

    # Relative headloss (dimensionless)
    # since H_clean_m = dH0_dL * L0
    H_total_rel = H_total_m / (H_clean_m + 1e-30)

    return H_total_m, H_total_rel, float(dH0_dL)



# =============================================================================
# MAIN MODEL (Eqs. 1–29)
# =============================================================================

@dataclass
class CamposSSFModel:
    nz: int
    L0: float
    dt: float
    H_tot0: float
    const: SSFConstants = field(repr=False)
    par: SSFCalibrationParams = field(repr=False)
    is_covered: bool = False  # from SSFConfig

    def __post_init__(self):
        # Depth grid [0, L0] with nz cells (cell centers for simplicity)
        # self.L = np.linspace(0.0, self.L0, self.nz)
        # self.dL = self.L[1] - self.L[0]
        # nz CELLS spanning [0, L0]
        self.dL = self.L0 / float(self.nz)
        self.L = (np.arange(self.nz) + 0.5) * self.dL  # cell centers
        # --------------------------------------------------------------
        # CLEAN-BED HYDRAULIC INITIAL CONDITION (Campos Eq. 22 reference)
        # --------------------------------------------------------------
        self.H_tot0 = float(self.H_tot0)            # [m]
        self.K0 = self.H_tot0 / (self.L0 + 1e-30)   # (dH/dL)_0  [m/m]
        # --------------------------------------------------------------
        # HYDRAULICS STORAGE (per depth cell, per time step)
        # --------------------------------------------------------------
        self.dH_dL = np.zeros(self.nz)     # (dH/dL)_j^n   [m/m]
        self.dH_cell = np.zeros(self.nz)   # ΔH_j^n        [m]
        self.H_cum = np.zeros(self.nz)     # cumulative H(L_j,t) [m]




        # Numerical-stability (CFL-type) check for advection in Eq. (16–17)
        # Using pore-velocity approximation u ≈ q (constant porosity simplif.).
        u_est = self.const.q
        if u_est * self.dt > self.dL:
            print(
                "[WARN] q * dt > dL; consider reducing dt or increasing nz "
                "to maintain numerical stability for solids advection."
            )

        # Schmutzdecke geometry: number of cells representing top 2 cm
        # for biology (Eqs. 24–25) and separate hydraulics if desired.
        self.n_top_cells_2cm = max(1, int(0.02 / self.dL))  # 2 cm region

        # Number of grid cells currently representing schmutzdecke hydraulically.
        # Starts at 0 (no schmutzdecke); will be updated from dLs in
        # update_schmutzdecke_daily.
        self.n_schm_cells = 0


        # --------------------------------------------------------------
        # INITIAL STATE VALUES (these are state, not "parameters")
        # --------------------------------------------------------------

        # Supernatant – can be later moved to Excel or initialized from data.
        self.a_sup = 0.1    # [mg Chla/L]
        self.ps_sup = 0.02  # [mg P/L]
        self.na_sup = 0.2   # [mg N/L] NH4
        self.ni_sup = 0.1   # [mg N/L] NO3


        # Filtration and deposits
        self.Cp = np.zeros(self.nz)                 # C_p(L,t)
        self.sigma_a = np.zeros(self.nz)            # σ_a(L,t)
        self.sigma = np.zeros(self.nz)              # σ(L,t)
        self.beta = np.full(self.nz, self.par.beta0_sand)  # λ(L,t)
        self.H_ratio = np.ones(self.nz)             # H(L,t)/H(L,0)

        # In-bed biology
        self.a_bed = np.zeros(self.nz)
        self.x_bed = np.zeros(self.nz) + 0.05       # small bacteria
        self.p_bed = np.zeros(self.nz) + 0.01       # small protozoa
        self.cp_bed = np.zeros(self.nz)
        self.cd_bed = np.zeros(self.nz) + 0.1
        self.ps_bed = np.zeros(self.nz) + 0.02

        # NOTE: For pilot-plant reproduction / multi-run, a higher-level
        # driver should override these with:
        #   * clean-bed conditions in the very first run, and
        #   * carry-over σ, β, biomass, and reduced L0 after scraping
        #     for subsequent runs.

        # Schmutzdecke (lumped state, mapped to grid via n_schm_cells)
        self.eps_s = self.par.eps0_s                # ε0s
        self.SB = 0.0                               # [mg Chla m^-2]
        self.SBc = 0.0                              # [mg C m^-2]
        self.dLs = 0.0                              # [m]
         # Depth-integrated relative headloss H(t)/H(0) for whole bed
        self.H_total_ratio = 1.0
        self.H_total_m = 0.0

        # Store clean-bed headloss and clean-bed gradient as model properties
        # (so we don't depend on global 'cfg' inside methods)
        # self.H_tot0 = None   # will be assigned by driver after model is created
        # self.K0 = None       # (dH/dL)_0 = H_tot0/L0


        # self.t_hours = 0.0


    # ========================================================
    # Supernatant module (Eqs. 1–8)
    # ========================================================

    def supernatant_optics(self, a: float, Nv: float, D: float):
        """
        Light extinction and mean irradiance in supernatant.

        Eq. (4): k_c' = k_ew + 0.052 N_v + 0.174 D
        Eq. (3): k_e  = k_c' + 0.0088 a + 0.054 a^(2/3)
        Eq. (5): I_a  = I_m / 2   (average daylight intensity)

        Discretization: these are purely algebraic, evaluated directly.
        """
        ke_star = self.par.kew + 0.052 * Nv + 0.174 * D
        ke = ke_star + 0.0088 * a + 0.054 * (a ** (2.0 / 3.0))
        Ia = self.par.Im / 2.0
        return ke_star, ke, Ia

    def algal_growth_rate_sup(self, T: float,
                              Nv: float, D: float, z: float) -> float:
        """
        Eq. (2) – algal growth rate in supernatant:

        k_g = k_gmaxa(T) * min( n/(k_sn+n), ps/(k_sp+ps) ) * f_d
              * [ k_e z / ln( (k_sl + I_a)/(k_sl + I_a e^{-k_e z}) ) ]

        Here n = na_sup + ni_sup (total dissolved N available to algae),
        as implied by coupling Eq. (2) with Eqs. (7–8).

        Discrete implementation:
          - evaluate k_g at current state (explicit Euler)
          - use k_g in Eq. (1) and subsequent nutrient ODEs.
        """
        _, ke, Ia = self.supernatant_optics(self.a_sup, Nv, D)
        kgmaxa_T = k_temp(self.par.kgmaxa20, self.const.theta_kga, T)

        n_tot = self.na_sup + self.ni_sup
        limN = n_tot / (self.par.ksn + n_tot + 1e-12)
        limP = self.ps_sup / (self.par.ksp + self.ps_sup + 1e-12)
        limNP = min(limN, limP)

        num = ke * z
        den = np.log((self.par.ksl + Ia) /
                     (self.par.ksl + Ia * np.exp(-ke * z) + 1e-30))
        light_term = num / (den + 1e-30)

        kg = kgmaxa_T * limNP * self.par.fd * light_term
        return kg

    def update_supernatant(self, T: float, Nv: float, D: float,
                           z: float):
        """
        Forward Euler discretization of supernatant equations.

        Eq. (1)  da/dt  = k_g a - k_ra a
        Eq. (6)  dps/dt = A_pa (k_ra a - k_g a)
        Eq. (7)  dna/dt = A_na k_ra a - k_n n_a
                          - (n_a/(k_am+n_a)) A_na k_g a
        Eq. (8)  dni/dt = k_n n_a
                          - (1 - n_a/(k_am+n_a)) A_na k_g a

        Discrete form for each state y in {a, ps, na, ni}:
          y^{n+1} = y^n + Δt * RHS(y^n)
        """
        kg = self.algal_growth_rate_sup(T, Nv, D, z)
        kra = k_temp(self.par.kra20, self.const.theta_kra, T)
        kn = k_temp(self.par.kn20, self.const.theta_kn, T)

        # Eq. (1)
        da_dt = kg * self.a_sup - kra * self.a_sup
        self.a_sup += da_dt * self.dt

        # Eq. (6)
        dps_dt = self.const.Apa * (kra * self.a_sup - kg * self.a_sup)
        self.ps_sup += dps_dt * self.dt

        # Eq. (7)
        pref = self.na_sup / (self.par.kam + self.na_sup + 1e-12)
        dna_dt = (self.const.Ana * kra * self.a_sup
                  - kn * self.na_sup
                  - pref * self.const.Ana * kg * self.a_sup)
        self.na_sup += dna_dt * self.dt

        # Eq. (8)
        dni_dt = kn * self.na_sup - (1.0 - pref) * self.const.Ana * kg * self.a_sup
        self.ni_sup += dni_dt * self.dt

    # ========================================================
    # Schmutzdecke (Eqs. 9–15, lumped single layer)
    # ========================================================

    def update_schmutzdecke_daily(self, dt_hours: float):
        """
        Simplified schmutzdecke dynamics, consistent with Eqs. (9–13).

        Eq. (9): SB_{j}(t_i) = SB_{j}(t_{i-1}) + A_n t_i
              -> here, incremental:  SB ← SB + A_n Δt

        Eq. (10): dL_s,j = SB_0,j / (ρ_a (1 − ε_0s))

        Eqs. (11–13): describe thickness and porosity update as a % growth
                      proportional to daily biomass increase.

        Here we use a single effective schmutzdecke layer, and then map
        its thickness to a number of hydraulic cells (n_schm_cells) in
        the depth grid.
        """
        if dt_hours <= 0.0:
            return

        # Biomass increase
        dSB = self.par.Ain * dt_hours
        SB_prev = self.SB
        self.SB += dSB

        # Carbon-equivalent surface biomass
        self.SBc = self.SB * self.const.Aca  # SBc = A_ca * SB

        # Porosity update (qualitative representation of Eqs. 11–13)

        # -----------------------------
        # Correct porosity update (Campos Eqs. 11–13)
        # Update porosity only once per day
        # -----------------------------
        if not hasattr(self, "schm_accum_time"):
            self.schm_accum_time = 0.0

        self.schm_accum_time += dt_hours

        # Eq. 9: biomass accumulation is continuous
        # Already done above: self.SB += dSB

        # Update porosity every 24 hours (Campos Section 3.2)
        # When a full day has passed, apply the % change ζ = phi_daily
        while self.schm_accum_time >= 24.0:
            self.schm_accum_time -= 24.0

            phi = self.par.phi_daily  # [%/day], ζ in Campos
            # Eq. (13): ε_0s^{i,j} = ε_0s^{i-1,j} + (ζ/100)(ε_0s^{i-1,j} − 1)
            self.eps_s = self.eps_s + (phi / 100.0) * (self.eps_s - 1.0)

            # Enforce Eq. (12) constraints: ε_1 ≤ ε_0s and ε_0s ≤ 1
            #   - upper bound: cannot exceed initial eps0_s
            #   - lower bound: keep > 0 to avoid zero-porosity singularities
            self.eps_s = min(self.eps_s, self.par.eps0_s)
            self.eps_s = max(self.eps_s, 1e-6)
            
        
        

        # Eq. (10): effective thickness (using initial ε0s)
        if self.par.eps0_s < 1.0:
            self.dLs = self.SB / (self.const.Da * (1.0 - self.par.eps0_s) + 1e-15)
        else:
            self.dLs = 0.0

        # Map thickness to number of schmutzdecke cells for hydraulics
        if self.dLs > 0.0:
            self.n_schm_cells = max(1, min(self.nz,
                                           int(np.ceil(self.dLs / self.dL))))
            #self.n_schm_cells = int(np.ceil(self.dLs / self.dL))

        else:
            self.n_schm_cells = 0
        #print("Schmutzdecke cells:", self.n_schm_cells, "Thickness (m):", self.dLs, "Porosity:", self.eps_s)
    
    def _happel_permeability(self, eps: float) -> float:
        """
        Happel (1958) permeability model for packed fibers,
        used by Campos et al. for schmutzdecke hydraulic resistance.

        k = (d_f^2 / 32) * [ -ln(phi) - (1 - phi^2)/(1 + phi^2) ]
        where phi = 1 - eps   (solid fraction) 
        """
        # Clip eps so phi stays in (0,1)
        eps = np.clip(eps, 1e-6, 1.0 - 1e-6)
        phi = 1.0 - eps
        df = self.const.df  # mean filamentous algae diameter (Table 2)

        happel_term = -np.log(phi) - (1.0 - phi**2) / (1.0 + phi**2)
        k = (df**2 / 32.0) * happel_term
        # Avoid zero / negative permeability
        k = max(k, 1e-12)
        return k

    def compute_happel_schmutzdecke(self) -> float:
        """
        Return the RELATIVE headloss gradient (dH/dL)_s / (dH/dL)_s0
        for the schmutzdecke, using Happel permeability.

        We avoid an undefined global 'dH_dL_initial' by computing the
        ratio analytically:

            (dH/dL)_s  ∝  u / k(eps_s)
            (dH/dL)_s0 ∝  u0 / k(eps0_s)

        with u = q/eps. Thus:

            ratio = (u/k) / (u0/k0)
                  = (eps0_s/eps_s) * (k(eps0_s)/k(eps_s))
        """
        # current schmutzdecke porosity (eps_s) and initial (eps0_s)
        eps_s = np.clip(self.eps_s, 1e-6, 0.999999)
        eps0_s = np.clip(self.par.eps0_s, 1e-6, 0.999999)

        k_now = self._happel_permeability(eps_s)
        k0 = self._happel_permeability(eps0_s)

        # u = q/eps, u0 = q/eps0  -> ratio u/u0 = eps0/eps
        ratio = (eps0_s / eps_s) * (k0 / k_now)
        return float(ratio)

    


    # ========================================================
    # Filtration and headloss (Eqs. 16–22)
    # ========================================================

    # def update_filtration_and_headloss(self, Cp_inlet: float, u_dummy: float):
    #     """
    #     Filtration + deposition + headloss, following Campos Eqs. (16–22).

    #     - Eq. (16): ∂C_p/∂L = −β C_p  (solved with upwind in L)
    #     - Eq. (17): u ∂C_p/∂L + ∂α/∂t = 0   with u = q / ε
    #     - Eq. (19): θ = b α   (bulk specific deposit)
    #     - ε       : ε = ε0 − θ   (effective porosity)
    #     - Eq. (18): β = β0 + a1 θ − a2 θ^2 / (ε0 − θ)
    #     - Eq. (22): dH/dL = (dH/dL)_0 * f(θ, ε0); we store relative factor
    #                  and integrate over the full depth to get H_total_ratio.
    #     """

    #     nz = self.nz
    #     dL = self.dL

    #     # --- (a) Clean-bed porosity per cell: sand vs schmutzdecke (ε0) ---
    #     # For sand cells use eps0 from calibration params.
    #     eps0_vec = np.full(nz, self.par.eps0)

    #     # For schmutzdecke cells use current schmutzdecke porosity eps_s
    #     if self.n_schm_cells > 0:
    #         eps0_vec[: self.n_schm_cells] = self.eps_s

    #     # --- (b) Eq. (16): solids profile C_p(L) using current β ---
    #     self.Cp[0] = Cp_inlet
    #     for i in range(1, nz):
    #         # Exact solution of dCp/dL = -β Cp
    #         self.Cp[i] = self.Cp[i - 1] * np.exp(-self.beta[i - 1] * dL)

    #     # --- (c) Eq. (17): update absolute specific deposit α via continuity ---
    #     # Use u = q / ε with ε taken from previous θ (inert-only for now).
    #     # α is stored in self.sigma_a [mg/L]; θ = b_bulk * α [vol/vol].
    #     alpha_old = self.sigma_a.copy()
    #     theta_old = self.par.b_bulk * alpha_old  # θ (inert contribution only)

    #     # Physical bound: 0 ≤ θ ≤ 0.95 ε0 so θ never reaches ε0.
    #     #theta_old = np.clip(theta_old, 0.0, 0.95 * eps0_vec)
    #     eps_old = eps0_vec - theta_old               # ε = ε0 − θ
    #     # Avoid zero porosity
    #     #eps_old = np.maximum(eps_old, 1e-6)

    #     # Approach velocity per cell (u = q/ε)
    #     u_vec = self.const.q / eps_old

    #     # Apply Eq. (17) with backward-space and forward-time:
    #     #   dα/dt = -u * (C_p[i] - C_p[i-1]) / ΔL
    #     for i in range(1, nz):
    #         dCp_dL = (self.Cp[i] - self.Cp[i - 1]) / (dL + 1e-10)
    #         dalpha_dt = -u_vec[i] * dCp_dL
    #         self.sigma_a[i] += dalpha_dt * self.dt
    #         # No negative deposits
    #         if self.sigma_a[i] < 0.0:
    #             self.sigma_a[i] = 0.0

    #     # --- (d) Eq. (19): θ = b α ---
    #     theta = self.par.b_bulk * self.sigma_a

    #     # Physical bound again after update
    #     #theta = np.clip(theta, 0.0, 0.8 * eps0_vec)
    #     #theta = np.minimum(theta, 0.8 * eps0_vec)

    #     # Store bulk specific deposit as σ (θ in the paper)
    #     self.sigma = theta

    #     # New effective porosity ε = ε0 − θ
    #     eps_eff = eps0_vec - theta
    #     #eps_eff = np.maximum(eps_eff, 1e-6)

    #     # --- (e) Eq. (18): filtration coefficient β(θ) ---
    #     beta0_vec = np.full(nz, self.par.beta0_sand)
    #     if self.n_schm_cells > 0:
    #         beta0_vec[: self.n_schm_cells] = self.par.beta0_schmutz

    #     denom = eps0_vec - theta + 1e-10
    #     self.beta = (beta0_vec
    #                  + self.par.a1 * theta
    #                  - self.par.a2 * (theta ** 2) / denom)

    #     # --- (f) Eq. (22): relative headloss per layer and depth-integrated total ---
    #     # For sand:
    #     #   H/H0 = [1 + θ/(1−ε0)]^c1 * [ε0/(ε0−θ)]^c2
    #     # For schmutzdecke, Campos uses a separate hydraulic model (Happel),
    #     # so we will replace the Eq. (22) factor by a Happel-based relative factor.

    #     # Apply Eq. 22 ONLY to sand layers (below schmutzdecke)
    #     sand_start = self.n_schm_cells

    #     eps0 = self.par.eps0  # clean-bed sand porosity
    #     term1 = np.ones_like(theta)
    #     term2 = np.ones_like(theta)

    #     term1[sand_start:] = 1.0 + theta[sand_start:] / (1.0 - self.par.eps0 + 1e-10)
    #     #term2 = eps0_vec / (eps0_vec - theta + 1e-10)
    #     term2[sand_start:] = self.par.eps0 / (self.par.eps0 - theta[sand_start:] + 1e-10)

    #     H_ratio_layer = np.ones_like(theta)

        

        

    #     H_ratio_layer[sand_start:] = (term1[sand_start:] ** self.par.c1) * \
    #                          (term2[sand_start:] ** self.par.c2)

    #     # Now override schmutzdecke using Happel hydraulics
        


    #     # --- Override top n_schm_cells with Happel-based schmutzdecke factor ---
    #     if self.n_schm_cells > 0:
    #         schm_ratio = self.compute_happel_schmutzdecke()
    #         # Use a constant relative factor over the schmutzdecke thickness
    #         H_ratio_layer[: self.n_schm_cells] = schm_ratio

    #     # Depth-integrated headloss ratio H(t)/H(0) for the whole bed.
    #     # Take (dH/dL)_0 such that integral over depth for clean bed = 1,
    #     # so the integral of H_ratio_layer gives the global ratio.
    #     dH0_dL = 1.0 / self.L0
    #     dH_dL_rel = dH0_dL * H_ratio_layer
    #     H_total_ratio = np.sum(dH_dL_rel) * dL

    #     self.H_ratio = H_ratio_layer
    #     self.H_total_ratio = H_total_ratio
    def update_filtration_and_headloss(self, Cp_inlet: float, u_dummy: float):
        """
        Filtration + deposition + headloss, following Campos Eqs. (16–22).

        - Eq. (16): ∂C_p/∂L = −β C_p  (solved analytically cell-by-cell)
        - Eq. (17) + Eq. (16): ∂α/∂t = u β C_p with u = q / ε
        - Eq. (19): θ = b α   (bulk specific deposit)
        - ε       : ε = ε0 − θ   (effective porosity)
        - Eq. (18): β = β0 + a1 θ − a2 θ^2 / (ε0 − θ)
        - Eq. (22): dH/dL = (dH/dL)_0 * f(θ, ε0)
                       -> we store a relative factor and integrate over depth
                          to get H_total_ratio.

        This implementation removes the spurious ΔL dependence by using
        ∂α/∂t = u β C_p (Campos Eqs. 16–17) directly instead of a finite
        difference on ∂C_p/∂L.
        """

        nz = self.nz
        dL = self.dL

        # --- (a) Clean-bed porosity per cell: sand vs schmutzdecke (ε0) ---
        # Sand cells: ε0 from calibration parameters.
        eps0_vec = np.full(nz, self.par.eps0)

        #test
        #self.n_schm_cells = 0


        # Schmutzdecke cells: use current schmutzdecke porosity ε_s
        if self.n_schm_cells > 0:
            eps0_vec[: self.n_schm_cells] = self.eps_s

        # Store β from previous step for use in deposition (explicit in time)
        beta_old = self.beta.copy()

        # --- (b) Eq. (16): solids profile C_p(L) using current β ---
        # Analytical solution for each cell assuming β constant in the cell:
        #   C_p[i] = C_p[i-1] * exp(-β[i-1] * ΔL)
        self.Cp[0] = Cp_inlet
        beta_eff = float(np.mean(beta_old))
        for i in range(1, nz):
            #self.Cp[i] = self.Cp[i - 1] * np.exp(-beta_old[i - 1] * dL)
            self.Cp[i] = self.Cp[i - 1] * np.exp(-beta_eff * dL)


        # --- (c) Eq. (17) + Eq. (16): ∂α/∂t = u β C_p (local, no ΔL) ---
        # Start from previous deposits (α_old) and porosity (ε_old).
        alpha_old = self.sigma_a.copy()
        theta_old = self.par.b_bulk * alpha_old  # θ (inert contribution only)

        # Physical bound: 0 ≤ θ ≤ 0.95 ε0 so θ never reaches ε0.
        theta_old = np.clip(theta_old, 0.0, 0.95 * eps0_vec)
        eps_old = eps0_vec - theta_old  # ε = ε0 − θ
        eps_old = np.maximum(eps_old, 1e-6)  # avoid zero porosity

        # Interstitial velocity per cell: u = q / ε
        u_vec = self.const.q / eps_old
        # Use clean-bed porosity for u to avoid runaway feedback
        # (sand: eps0, schmutz: eps_s already in eps0_vec)
        #u_vec = self.const.q / np.maximum(eps0_vec, 1e-6)


        # Local deposition rate from Eqs. (16–17):
        #   ∂α/∂t = u β C_p
        # for i in range(nz):
        #     dalpha_dt = u_vec[i] * beta_old[i] * self.Cp[i]
        #     self.sigma_a[i] += dalpha_dt * self.dt
        #     # No negative deposits
        #     if self.sigma_a[i] < 0.0:
        #         self.sigma_a[i] = 0.0

        # Local deposition from Eq. (17) in conservative (finite-volume) form:
        # for i in range(nz):
        #     # Cp_in  = self.Cp[i-1] if i > 0 else Cp_inlet   # upstream value at cell entrance

        #     # Cp_out = self.Cp[i]

        #     # # cell-average Cp over ΔL (exact for exponential profile)
        #     # Cp_bar = (Cp_in - Cp_out) / (beta_old[i] * self.dL + 1e-30)

        #     # dalpha_dt = u_vec[i] * beta_old[i] * Cp_bar
        #     # self.sigma_a[i] += dalpha_dt * self.dt* self.dL

        #     # if self.sigma_a[i] < 0.0:
        #     #     self.sigma_a[i] = 0.0

            

        #     # Cp at cell center (already consistent with Eq. 16)
        #     Cp_local = self.Cp[i]

        #     # Campos Eq. (17): ∂α/∂t = u β Cp
        #     dalpha_dt = u_vec[i] * beta_old[i] * Cp_local

        #     self.sigma_a[i] += dalpha_dt * self.dt/self.nz

        #     if self.sigma_a[i] < 0.0:
        #         self.sigma_a[i] = 0.0

        # Campos Eq. (17): deposition applies to ONE representative bed volume

        # Depth-averaged Cp weighted by filtration (consistent with Eq. 16)
        #Cp_bar = float(np.mean(self.Cp))
        # Exact depth-average of Cp(L) for exponential profile
        Cp_bar = float(
            (self.Cp[0] - self.Cp[-1]) / (beta_eff * self.L0 + 1e-30)
        )


        # Depth-averaged beta
        beta_bar = float(np.mean(beta_old))

        # Depth-averaged pore velocity
        u_bar = float(self.const.q / self.par.eps0)

        # Single deposition rate for the bed
        dalpha_dt = u_bar * beta_bar * Cp_bar

        # Update ONE alpha, replicated across depth grid
        self.sigma_a += (dalpha_dt * self.dt)




        # --- (d) Eq. (19): θ = b α with physical bounds enforced ---
        theta = self.par.b_bulk * self.sigma_a
        #theta = np.clip(theta, 0.0, 0.3 * eps0_vec)

        # Store bulk specific deposit as σ (θ in the paper)
        self.sigma = theta

        # New effective porosity ε = ε0 − θ
        eps_eff = eps0_vec - theta
        eps_eff = np.maximum(eps_eff, 1e-6)

        # --- (e) Eq. (18): filtration coefficient β(θ) ---
        beta0_vec = np.full(nz, self.par.beta0_sand)
        if self.n_schm_cells > 0:
            beta0_vec[: self.n_schm_cells] = self.par.beta0_schmutz

        denom = eps0_vec - theta + 1e-10
        self.beta = (
            beta0_vec
            + self.par.a1 * theta
            - self.par.a2 * (theta ** 2) / denom
        )

        # --- (f) Eq. (22): relative headloss per layer and depth-integrated total ---

        # For sand:
        #   H/H0 = [1 + θ/(1−ε0)]^c1 * [ε0/(ε0−θ)]^c2
        # For schmutzdecke, we use a separate hydraulic model (Happel).

        # sand_start = self.n_schm_cells

        # term1 = np.ones_like(theta)
        # term2 = np.ones_like(theta)

        # # Sand layers only
        # term1[sand_start:] = 1.0 + theta[sand_start:] / (
        #     1.0 - self.par.eps0 + 1e-10
        # )
        # term2[sand_start:] = self.par.eps0 / (
        #     self.par.eps0 - theta[sand_start:] + 1e-10
        # )

        # sand_start = self.n_schm_cells

        # f = np.ones_like(theta)

        # # Campos Eq. (22) — sand only
        # f[sand_start:] = (
        #     (1.0 + theta[sand_start:] / (1.0 - self.par.eps0 + 1e-10)) ** self.par.c1
        #     * (self.par.eps0 / (self.par.eps0 - theta[sand_start:] + 1e-10)) ** self.par.c2
        # )

        # # (dH/dL)_j^n = (dH/dL)_0 * f_j
        # self.dH_dL[:] = self.K0 * f

        # # ΔH_j^n = (dH/dL)_j^n * ΔL
        # self.dH_cell[:] = self.dH_dL[:] * self.dL

        # # H(L_j, t^n) = Σ ΔH_k
        # self.H_cum[:] = np.cumsum(self.dH_cell)

        # self.H_total_m = float(np.sum(self.dH_cell))
        # self.H_total_ratio = float(self.H_total_m / (self.H_tot0 + 1e-30))

        # =========================================================
        # (f) Headloss from headloss gradient (Campos-style anchor)
        #     + nz-invariant schmutzdecke slab thickness
        # =========================================================

        sand_start = int(self.n_schm_cells)

        # --- 1) Compute f_sand from Campos Eq. (22) for sand cells only ---
        f_sand = np.ones_like(theta, dtype=float)

        if sand_start < nz:
            f_sand[sand_start:] = (
                (1.0 + theta[sand_start:] / (1.0 - self.par.eps0 + 1e-10)) ** self.par.c1
                * (self.par.eps0 / (self.par.eps0 - theta[sand_start:] + 1e-10)) ** self.par.c2
            )

        # --- 2) Clean-bed gradient anchor (dH/dL)_0 ---
        # These MUST already be set in __post_init__ from the passed H_tot0
        if (self.K0 is None) or (self.H_tot0 is None):
            raise ValueError("self.K0 or self.H_tot0 is None. Do NOT overwrite them to None in __post_init__.")

        K0 = float(self.K0)         # [m/m]
        H0 = float(self.H_tot0)     # [m]

        # --- 3) Sand: per-cell gradient and headloss ---
        # dH/dL in sand cells
        self.dH_dL[:] = 0.0
        self.dH_cell[:] = 0.0

        # if sand_start < nz:
        #     self.dH_dL[sand_start:] = K0 * f_sand[sand_start:]           # [m/m]
        #     self.dH_cell[sand_start:] = self.dH_dL[sand_start:] * self.dL # [m] using ΔL for sand


        # --- 3) Sand: integrate over FULL sand bed depth L0 (never shorten by n_schm_cells) ---
        # Campos Eq. (22) factor applied to all sand-depth cells (your grid always spans [0, L0])
        f_sand_full = (
            (1.0 + theta / (1.0 - self.par.eps0 + 1e-10)) ** self.par.c1
            * (self.par.eps0 / (self.par.eps0 - theta + 1e-10)) ** self.par.c2
        )

        self.dH_dL[:] = K0 * f_sand_full                # [m/m] across full sand depth
        self.dH_cell[:] = self.dH_dL[:] * self.dL       # [m] integrate with ΔL across [0, L0]

        # --- 4) Schmutzdecke: treat as slab thickness dLs (nz-invariant) ---
        # We still populate per-cell arrays for diagnostics, but we distribute
        # the slab headloss across the schmutz cells so ΣΔH_schm = (dH/dL)_schm * dLs.
        H_schm = 0.0
        if (not self.is_covered) and (sand_start > 0) and (self.dLs > 0.0):
            schm_ratio = self.compute_happel_schmutzdecke()   # dimensionless gradient ratio
            K_schm = K0 * float(schm_ratio)                   # [m/m]

            # distribute over schmutz cells so total equals K_schm * dLs
            dL_schm_cell = self.dLs / float(sand_start)       # [m] effective thickness per schmutz cell

            # self.dH_dL[:sand_start] = K_schm                  # [m/m] (diagnostic)
            # self.dH_cell[:sand_start] = K_schm * dL_schm_cell  # [m] nz-invariant total over schmutz

            # Schmutzdecke headloss as ADDED slab on top of the sand bed
            H_schm = float(K_schm * self.dLs)

        else:
            # no schmutzdecke hydraulics
            pass

        # --- 5) Cumulative headloss profile and totals ---
        # self.H_cum[:] = np.cumsum(self.dH_cell)               # [m] cumulative vs depth cell index
        # self.H_total_m = float(self.H_cum[-1])                # [m]
        self.H_cum[:] = np.cumsum(self.dH_cell)                 # sand-only cumulative profile
        H_sand = float(np.sum(self.dH_cell))                    # sand-only headloss [m]
        self.H_total_m = float(H_sand + H_schm)                 # add schmutz slab

        self.H_total_ratio = float(self.H_total_m / (H0 + 1e-30))

        # Optional: store Eq. 22 factor for plotting (sand only)
        self.H_ratio = f_sand


                # --- (f) Headloss from hydraulic gradient (Campos-style) ---
        # We compute total headloss as:
        #   H_tot = ∫_sand K0*f_sand dL  +  (K0*ratio_schm)*dLs
        # This prevents nz-dependence from mapping dLs -> n_schm_cells.

        # if (self.K0 is None) or (self.H_tot0 is None):
        #     raise ValueError("Model headloss anchor not set. Set model.H_tot0 and model.K0 in the driver.")

        # K0 = float(self.K0)      # clean-bed hydraulic gradient [m/m]
        # H_tot0 = float(self.H_tot0)

        # sand_start = self.n_schm_cells  # schmutz cells used for biology/mapping only

        # # Campos Eq. 22 factor for sand region only
        # f_sand = np.ones_like(theta)
        # f_sand[sand_start:] = (term1[sand_start:] ** self.par.c1) * (term2[sand_start:] ** self.par.c2)

        # # ---- Sand contribution: sum over sand cells ----
        # K_sand = K0 * f_sand[sand_start:]
        # H_sand = float(np.sum(K_sand) * self.dL)   # ∫ K dL over sand

        # # ---- Schmutzdecke contribution: treat as ONE slab of thickness dLs ----
        # H_schm = 0.0
        # if (not self.is_covered) and (self.dLs > 0.0):
        #     schm_ratio = self.compute_happel_schmutzdecke()  # ratio of gradients
        #     K_schm = K0 * schm_ratio
        #     H_schm = float(K_schm * self.dLs)                # (dH/dL)_schm * dLs

        # # ---- Total ----
        # self.H_total_m = float(H_sand + H_schm)
        # self.H_total_ratio = float(self.H_total_m / (H_tot0 + 1e-30))

        # # Optional diagnostics
        # # Keep f-like array to inspect sand headloss growth; do NOT use it for total integration.
        # self.H_ratio = f_sand







        # H_ratio_layer = np.ones_like(theta)

        # # Eq. (22) for sand region
        # H_ratio_layer[sand_start:] = (
        #     term1[sand_start:] ** self.par.c1
        #     * term2[sand_start:] ** self.par.c2
        # )

        # # Override top n_schm_cells using Happel-based schmutzdecke factor
        # # if self.n_schm_cells > 0:
        # #     schm_ratio = self.compute_happel_schmutzdecke()
        # #     H_ratio_layer[: self.n_schm_cells] = schm_ratio

        # # Depth-integrated headloss ratio H(t)/H(0) for the whole bed.
        # # Set (dH/dL)_0 so that clean-bed integral over depth = 1,
        # # so ∫ H_ratio_layer dL yields the global ratio.
        # # dH0_dL = 1.0 / self.L0
        # # dH_dL_rel = dH0_dL * H_ratio_layer
        # # H_total_ratio = np.sum(dH_dL_rel) * dL

        # H_total_m, H_total_ratio, _ = compute_total_headloss_m(
        # H_ratio_layer=H_ratio_layer,
        # dL=self.dL,
        # L0=self.L0,
        # H_clean_m=cfg.H_tot0
        # )   


        

        # self.H_ratio = H_ratio_layer
        # self.H_total_ratio = float(H_total_ratio)
        # self.H_total_m = float(H_total_m)

        # --- (f) ABSOLUTE hydraulic gradient integration (no ratio discretization) ---

        # H_clean_m = cfg.H_tot0
        # K0 = H_clean_m / (self.L0 + 1e-30)   # clean-bed hydraulic gradient [m/m]

        # # f(theta) for sand cells (Campos Eq. 22 factor)
        # f = np.ones_like(theta)
        # f[sand_start:] = (term1[sand_start:] ** self.par.c1) * (term2[sand_start:] ** self.par.c2)

        # # Dimensional local gradient [m/m]
        # K_layer = K0 * f

        # # Total headloss [m] = ∫ K dL
        # H_total_m = float(np.sum(K_layer) * self.dL)

        # # Relative headloss only for reporting (NOT used for discretization)
        # H_total_ratio = float(H_total_m / (H_clean_m + 1e-30))

        # # Store results
        # self.H_total_m = H_total_m
        # self.H_total_ratio = H_total_ratio

        # # Optional diagnostic only (can keep or remove)
        # self.H_ratio = f


    




    #Total headloss calculation
    
    # def total_headloss_ratio(self) -> float:
    #     """
    #     Compute relative total headloss H_tot(t)/H_tot(0).

    #     H_tot_rel = (1/L0) * ∫ H_ratio(L,t) dL
    #               ≈ (1/L0) * Σ (H_ratio[j] * ΔL)
    #     """
    #     integral_Hratio = np.trapz(self.H_ratio, self.L)
    #     H_tot_rel = integral_Hratio / self.L0
    #     return float(H_tot_rel)

    def total_headloss_ratio(self):
        return float(self.H_total_ratio)








    # ========================================================
    # Microbial dynamics in bed (Eqs. 23–29)
    # ========================================================

    def update_biology(self, T: float):
        """
        Depth-wise biomass and carbon dynamics with forward Euler.

        Eq. (23) – algae:
            da/dt = −k_ra a − (a/(k_sa + a)) C_gz p

        Eq. (24, 25) – bacteria:
            dx/dt = (DOC uptake + SB uptake + P uptake)
                    − (k_db + k_rb) x − grazing_on_bacteria

        Top ~2 cm: split carbon inputs 2/41 from DOC, 38/41 from SB;
        deeper layers: 40/41 from DOC, none from SB.

        Eq. (26) – protozoa:
            dp/dt = C to protozoa by grazing on algae and bacteria
                    − (k_dp + k_rp) p

        Eq. (27) – nonliving POC cp:
            dcp/dt = sources from grazing + bacterial/protozoan death
                     − k_p cp

        Eq. (28) – DOC cd:
            dcd/dt = k_p cp − k_h cd
                     − (1/Y_b)*f_b*k_gmaxb * cd/(k_scd + cd) * x

        Eq. (29) – phosphorus in bed ps:
            dps/dt = A_pc k_h cd + A_pa k_ra a
                     − (1/41)*(1/Y_b)*k_gmaxb * ps/(k_sp + ps) * x
        """
        kdb = k_temp(self.par.kdb20, self.const.theta_kdb, T)
        kdp = k_temp(self.par.kdp20, self.const.theta_kdp, T)
        kp = k_temp(self.par.kp20, self.const.theta_kp, T)
        kh = k_temp(self.par.kh20, self.const.theta_kh, T)
        kgmaxb = k_temp(self.par.kgmaxb20, self.const.theta_kgb, T)
        kgmaxp = k_temp(self.par.kgmaxp20, self.const.theta_kgp, T)
        kra = k_temp(self.par.kra20, self.const.theta_kra, T)
        krb = k_temp(self.par.krb20, self.const.theta_krb, T)
        krp = k_temp(self.par.krp20, self.const.theta_krp, T)
        Cgz = k_temp(self.par.Cgz20, self.const.theta_Cgz, T)

        for j in range(self.nz):
            a = self.a_bed[j]
            x = self.x_bed[j]
            p = self.p_bed[j]
            cp = self.cp_bed[j]
            cd = self.cd_bed[j]
            ps = self.ps_bed[j]

            # ---- Eq. (23): algae in bed ----
            graz_term = (a / (self.par.ksa + a + 1e-12)) * Cgz * p
            da_dt = -kra * a - graz_term
            a_new = a + da_dt * self.dt

            # ---- Eq. (24–25): bacteria ----
            is_top_region = (j < self.n_top_cells_2cm)
            if is_top_region:
                f_cd = 2.0 / 41.0
                f_SB = 38.0 / 41.0
            else:
                f_cd = 40.0 / 41.0
                f_SB = 0.0

            uptake_cd = f_cd * kgmaxb * (cd / (self.par.kscd + cd + 1e-12)) * x
            # For SB uptake, use SBc as a lumped surface carbon pool.
            uptake_SB = f_SB * kgmaxb * (self.SBc / (self.par.kscd + self.SBc + 1e-12)) * x
            uptake_P = (1.0 / 41.0) * kgmaxb * (ps / (self.par.ksp + ps + 1e-12)) * x

            grazing_on_bact = (1.0 / self.par.Yp) * kgmaxp * (x / (self.par.ksb + x + 1e-12)) * p

            dx_dt = uptake_cd + uptake_SB + uptake_P - kdb * x - krb * x - grazing_on_bact
            x_new = x + dx_dt * self.dt

            # ---- Eq. (26): protozoa ----
            graz_algae = (self.const.Aca * self.par.Eh
                          * (a / (self.par.ksa + a + 1e-12)) * Cgz * p)
            graz_bact = kgmaxp * (x / (self.par.ksb + x + 1e-12)) * p
            dp_dt = graz_algae + graz_bact - kdp * p - krp * p
            p_new = p + dp_dt * self.dt

            # ---- Eq. (27): POC ----
            cp_from_graz_algae = (self.const.Aca * (1.0 - self.par.Eh)
                                  * (a / (self.par.ksa + a + 1e-12)) * Cgz * p)
            cp_from_graz_bact = ((1.0 / self.par.Yp - 1.0)
                                 * kgmaxp * (x / (self.par.ksb + x + 1e-12)) * p)
            dcp_dt = cp_from_graz_algae + cp_from_graz_bact + kdb * x + kdp * p - kp * cp
            cp_new = cp + dcp_dt * self.dt

            # ---- Eq. (28): DOC ----
            factor_b = 2.0 / 41.0 if is_top_region else 40.0 / 41.0
            dcd_dt = (kp * cp
                      - kh * cd
                      - (1.0 / self.par.Yb) * factor_b * kgmaxb
                      * (cd / (self.par.kscd + cd + 1e-12)) * x)
            cd_new = cd + dcd_dt * self.dt

            # ---- Eq. (29): P in bed ----
            dps_dt = (self.const.Apc * kh * cd
                      + self.const.Apa * kra * a
                      - (1.0 / 41.0) * (1.0 / self.par.Yb) * kgmaxb
                      * (ps / (self.par.ksp + ps + 1e-12)) * x)
            ps_new = ps + dps_dt * self.dt

            self.a_bed[j] = a_new
            self.x_bed[j] = x_new
            self.p_bed[j] = p_new
            self.cp_bed[j] = cp_new
            self.cd_bed[j] = cd_new
            self.ps_bed[j] = ps_new

    # ========================================================
    # Cp_inlet coupling helper (covered vs uncovered)
    # ========================================================

    def compute_Cp_inlet(self, infl: SSFInfluent) -> float:
        """
        Compute Cp at the inlet to the bed.

        Covered filters:
            Cp_inlet = inert solids from influent only.

        Uncovered filters:
            Cp_inlet = inert solids + algal particulate from supernatant.

        The algal contribution is converted from chlorophyll-a using the
        stoichiometric factor Abw (algal dry weight per unit chlorophyll).
        """
        Cp_inert = infl.Cp_inlet_inert

        if self.is_covered:
            return Cp_inert

        # Uncovered: add supernatant algae contribution.
        # Convert a_sup [mg Chla/L] -> [mg solids/L] with Abw.
        Cp_algae = self.const.Abw * self.a_sup
        Cp_total = Cp_inert + Cp_algae

        # Ensure non-negative
        if Cp_total < 0.0:
            Cp_total = 0.0
        return Cp_total

    # ========================================================
    # One full time step: Eqs. 1–29 in sequence
    # ========================================================

    def step(self, infl: SSFInfluent):
        """
        Apply one global Δt step to all coupled components:

        1) Supernatant (Eqs. 1–8) – only if filter is uncovered.
        2) Schmutzdecke (Eqs. 9–15) – only if filter is uncovered.
        3) Filtration & headloss (Eqs. 16–22).
        4) In-bed biology (Eqs. 23–29).
        """
        # Ensure simulation clock exists
        if not hasattr(self, "t_hours"):
            self.t_hours = 0.0

        # 1) Supernatant module (Eqs. 1–8)
        if not self.is_covered:
            self.update_supernatant(
                T=infl.T,
                Nv=infl.Nv,
                D=infl.D,
                z=infl.z,
            )

        # 2) Schmutzdecke growth (Eqs. 9–15) – treat dt as hours
        # if not self.is_covered:
        #     self.update_schmutzdecke_daily(dt_hours=self.dt)
        if (not self.is_covered) and (self.t_hours >= self.const.Tlag_h):
            self.update_schmutzdecke_daily(dt_hours=self.dt)


        # 3) Filtration & headloss (Eqs. 16–22)
        Cp_inlet = self.compute_Cp_inlet(infl)
        # u is now computed internally as q/ε per layer (Eq. 17), so the
        # second argument is a dummy kept only for interface compatibility.
        self.update_filtration_and_headloss(Cp_inlet=Cp_inlet, u_dummy=0.0)

        # 4) In-bed biology (Eqs. 23–29)
        self.update_biology(T=infl.T)
        self.t_hours += self.dt



# =============================================================================
# Simple multi-run skeleton and calibration helper
# =============================================================================

# def run_single_run(model: CamposSSFModel,
#                    infl: SSFInfluent,
#                    n_steps: int) -> pd.DataFrame:
#     """
#     Run a single filtration run for n_steps and return a history DataFrame.

#     This does not apply cleaning; for multi-run pilot-plant reproduction,
#     call this in a loop and implement scraping/carry-over externally.
#     """
#     history = []
#     for istep in range(n_steps):
#         t = istep * model.dt
#         model.step(infl)
#         if istep == 0:
#             print(
#                 "Cp_bottom/Cp_top at t=0:",
#                 model.Cp[-1] / (model.Cp[0] + 1e-30)
#             )
#         history.append({
#             "time_h": t,
#             "a_sup": model.a_sup,
#             "ps_sup": model.ps_sup,
#             "na_sup": model.na_sup,
#             "ni_sup": model.ni_sup,
#             "Cp_top": model.Cp[0],
#             "Cp_bottom": model.Cp[-1],
#             "sigma_top": model.sigma[0],
#             "sigma_bottom": model.sigma[-1],
#             "H_ratio_top": model.H_ratio[0],
#             "H_ratio_bottom": model.H_total_ratio,
#             "H_ratio_bottom_local": model.H_ratio[-1],
#             "x_bed0": model.x_bed[0],
#             "p_bed0": model.p_bed[0],
#             "cd_bed0": model.cd_bed[0],
#             "cp_bed0": model.cp_bed[0],
#             "ps_bed0": model.ps_bed[0],
#         })
#     return pd.DataFrame(history)


def sse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Simple sum of squared errors between observed and simulated arrays.

    Intended as a generic stand-in for Eq. (35)-style objectives; exact
    formulation for the Campos paper can be enforced in a separate script.
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)
    return float(np.sum((simulated - observed) ** 2))

def write_results_definitions_csv(df: pd.DataFrame,
                                  path: str = "Campos_SSF_Results_DEFINITIONS.csv") -> None:
    """
    Write a separate CSV containing human-readable definitions of each
    column in the simulation results DataFrame.

    This keeps Campos_SSF_Results.csv purely numeric (for plotting and
    analysis), while Campos_SSF_Results_DEFINITIONS.csv documents what
    each variable means.

    The mapping below is based on the current history fields written in
    the main loop:
        time_h, a_sup, ps_sup, na_sup, ni_sup,
        Cp_top, Cp_bottom,
        sigma_top, sigma_bottom,
        H_ratio_top, H_ratio_bottom,
        x_bed0, p_bed0, cd_bed0, cp_bed0, ps_bed0
    """
    # Definitions for known columns. Any future column not listed here
    # will still be included with a placeholder definition.
    definitions = {
        "time_h": (
            "Time since start of filtration run [hours]."
        ),
        "a_sup": (
            "Supernatant algae concentration as chlorophyll-a [mg Chla/L] "
            "(Eqs. 1–2)."
        ),
        "ps_sup": (
            "Supernatant soluble phosphorus concentration [mg P/L] "
            "(Eq. 6)."
        ),
        "na_sup": (
            "Supernatant ammonium nitrogen (NH4-N) concentration [mg N/L] "
            "(Eq. 7)."
        ),
        "ni_sup": (
            "Supernatant nitrate nitrogen (NO3-N) concentration [mg N/L] "
            "(Eq. 8)."
        ),
        "Cp_top": (
            "Suspended solids concentration in pore water at top of sand bed "
            "[mg/L] (after filtration along depth, Eq. 16)."
        ),
        "Cp_bottom": (
            "Suspended solids concentration in pore water at bottom of sand "
            "bed [mg/L] (Eq. 16)."
        ),
        "sigma_top": (
            "Bulk specific deposit σ at top layer [dimensionless vol/vol], "
            "σ = b·σ_a (Eq. 19)."
        ),
        "sigma_bottom": (
            "Bulk specific deposit σ at bottom layer [dimensionless vol/vol], "
            "σ = b·σ_a (Eq. 19)."
        ),
        "H_ratio_top": (
            "Relative headloss H(L,t)/H(L,0) at top layer [-], computed from "
            "deposit via Eq. 22."
        ),
        "H_ratio_bottom": (
            "Relative headloss H(L,t)/H(L,0) at bottom layer [-], computed "
            "from deposit via Eq. 22."
        ),
        "x_bed0": (
            "Bacterial biomass concentration in first bed layer (near surface) "
            "[mg C/L] (Eqs. 24–25)."
        ),
        "p_bed0": (
            "Protozoa biomass concentration in first bed layer [mg C/L] "
            "(Eq. 26)."
        ),
        "cd_bed0": (
            "Dissolved organic carbon (DOC) concentration in first bed layer "
            "[mg C/L] (Eq. 28)."
        ),
        "cp_bed0": (
            "Nonliving particulate organic carbon (POC) in first bed layer "
            "[mg C/L] (Eq. 27)."
        ),
        "ps_bed0": (
            "Phosphorus concentration in first bed layer [mg P/L] (Eq. 29)."
        ),
                "H_total_rel": (
            "Relative total headloss across the full filter bed, "
            "H_tot(t)/H_tot(0), obtained by depth-integration of Eq. 22."
        ),
        "H_total_m": (
            "Absolute total headloss across the full filter bed [m of water], "
            "computed as H_tot0 * H_total_rel where H_tot0 is the clean-bed "
            "headloss supplied by the user."
        ),

    }

    rows = []
    for col in df.columns:
        desc = definitions.get(
            col,
            "NO DEFINITION AVAILABLE – please document this variable."
        )
        rows.append({"variable_name": col, "definition": desc})

    df_def = pd.DataFrame(rows, columns=["variable_name", "definition"])
    df_def.to_csv(path, index=False)

def run_single_run(model: CamposSSFModel,
                   infl: SSFInfluent,
                   n_steps: int) -> pd.DataFrame:
    """
    Run a single filtration run with a fixed influent and collect:

      1) Time-series summary outputs (returned as df_results)
      2) Full depth–time profiles (written to Campos_SSF_Profiles.csv)

    The main script is responsible only for writing df_results to
    Campos_SSF_Results.csv and the definitions CSV.
    """

    # -------------------------------
    # Containers for results
    # -------------------------------
    history = []          # time-series summary (one row per time step)
    profile_history = []  # depth profiles (one row per depth cell per time step)
    
    H_tot0 = cfg.H_tot0  # e.g. 0.20 m clean-bed headloss – value read from excel

    # -------------------------------
    # Time loop
    # -------------------------------
    for istep in range(n_steps):
        t = istep * model.dt  # [h] time since start of run

        # Advance the full Campos model by one time step
        model.step(infl)
        if istep == 0:
             print(
                 "Cp_bottom/Cp_top at t=0:",
                 model.Cp[-1] / (model.Cp[0] + 1e-30)
             )

        # --- total headloss (relative and absolute) ---
        H_total_rel = model.H_total_ratio      # H_tot(t)/H_tot(0)
        H_total_abs = model.H_total_m              # [m], if you want absolute

        # ---- STOP CRITERION (Campos operational termination) ----
        # if H_total_abs >= cfg.H_stop_m:
        #     print(
        #         f"Stopping run at t={t:.2f} h because headloss reached "
        #         f"H_stop_m={cfg.H_stop_m:.3f} m (H_total_m={H_total_abs:.3f} m)."
        #     )
        #     # record this final state row already added below, or break now:
        #     # If you want the stopping row included, place this block AFTER history.append(...)
        #     break


        # ---- 1) Summary outputs for this time step (top/bottom, supernatant, biology) ----
        history.append({
            "time_h": t,
            "a_sup": model.a_sup,
            "ps_sup": model.ps_sup,
            "na_sup": model.na_sup,
            "ni_sup": model.ni_sup,
            "Cp_top": model.Cp[0],
            "Cp_bottom": model.Cp[-1],
            "sigma_top": model.sigma[0],
            "sigma_bottom": model.sigma[-1],
            "H_ratio_top": model.H_ratio[0],
            "H_ratio_bottom": model.H_ratio[-1],
            "H_total_rel": H_total_rel,    
            "H_total_m": H_total_abs, 
            "x_bed0": model.x_bed[0],
            "p_bed0": model.p_bed[0],
            "cd_bed0": model.cd_bed[0],
            "cp_bed0": model.cp_bed[0],
            "ps_bed0": model.ps_bed[0],
            "theta_max": float(np.max(model.sigma)),
            "eps_eff_min_sand": float(model.par.eps0 - np.max(model.sigma[model.n_schm_cells:]) if model.n_schm_cells < model.nz else model.par.eps0),

        })

        # ---- 2) Full depth profiles for this time step ----
        for j in range(model.nz):
            z = model.L[j]          # depth [m]
            Cp_j = model.Cp[j]      # suspended solids [mg/L]
            sigma_j = model.sigma[j]  # bulk specific deposit [vol/vol]

            # Constant superficial velocity as used in the current hydraulics
            u_const = model.const.q

            # Effective pore velocity if porosity loss were enforced (diagnostic only)
            eps0 = model.par.eps0
            u_eff = u_const / (eps0 - sigma_j + 1e-30)

            profile_history.append({
                "time_h": t,
                "z_m": z,
                "Cp": Cp_j,
                "sigma": sigma_j,
                "u_const": u_const,
                "u_eff": u_eff,
                "dH_dL": model.dH_dL[j],     # gradient
                "dH_cell": model.dH_cell[j], # local headloss
                "H_cum": model.H_cum[j],     # cumulative headloss
            })

    # -------------------------------
    # Build DataFrames
    # -------------------------------
    df_results = pd.DataFrame(history)

    df_profiles = pd.DataFrame(profile_history)
    df_profiles.to_csv("Campos_SSF_Profiles.csv", index=False)
    print("Depth–time profiles written to Campos_SSF_Profiles.csv")
    print("df_profiles shape:", df_profiles.shape)

    return df_results




# =============================================================================
# DEMO DRIVER: RUN MODEL, COLLECT HISTORY, EXPORT CSV, EXPOSE df_results
# =============================================================================

if __name__ == "__main__":
    excel_path = "Campos_SSF_Parameters.xlsx"  # same folder as this script

    # --- Load all inputs from Excel ---
    const = SSFConstants.from_excel(excel_path, sheet="SSFConstants")
    par = SSFCalibrationParams.from_excel(excel_path, sheet="SSFCalibrationParams")
    cfg = SSFConfig.from_excel(excel_path, sheet="SSFConfig")
    infl = SSFInfluent.from_excel(excel_path, sheet="SSFInfluent")

    # --- Create model instance using config + constants ---
    model = CamposSSFModel(
        nz=cfg.nz,
        L0=const.L0,
        dt=cfg.dt,
        H_tot0=cfg.H_tot0,
        const=const,
        par=par,
        is_covered=cfg.is_covered,
    )

    # Define initial condition via headloss gradient (Campos-style anchor)
    model.H_tot0 = float(cfg.H_tot0)
    model.K0 = model.H_tot0 / (model.L0 + 1e-30)  # (dH/dL)_0 [m/m]


    # --- Run a single filtration run and collect history ---
    df_results = run_single_run(model, infl, cfg.n_steps)

    # --- Export CSV ---
    # --- Export CSV ---
    df_results.to_csv("Campos_SSF_Results.csv", index=False)
    print("\nTime series written to Campos_SSF_Results.csv")
    print("df_results shape:", df_results.shape)

    # --- Export definitions file ---
    write_results_definitions_csv(
        df_results,
        path="Campos_SSF_Results_DEFINITIONS.csv"
    )
    print("Variable definitions written to Campos_SSF_Results_DEFINITIONS.csv")
    

        # --- Plot results vs time ---

    # 1) Total headloss (m)
    plt.figure()
    plt.plot(df_results["time_h"], df_results["H_total_m"], label="Total headloss H_total_m")
    plt.xlabel("Time [h]")
    plt.ylabel("Headloss [m]")
    plt.title("Total headloss vs time")
    plt.legend()
    plt.grid(True)

    # 2) Suspended solids at top and bottom of bed
    plt.figure()
    plt.plot(df_results["time_h"], df_results["Cp_top"], label="Cp_top")
    plt.plot(df_results["time_h"], df_results["Cp_bottom"], label="Cp_bottom")
    plt.xlabel("Time [h]")
    plt.ylabel("Suspended solids Cp [mg/L]")
    plt.title("Suspended solids at top and bottom vs time")
    plt.legend()
    plt.grid(True)

    # 3) Supernatant algae and nutrients
    plt.figure()
    plt.plot(df_results["time_h"], df_results["a_sup"], label="a_sup (algae)")
    plt.plot(df_results["time_h"], df_results["ps_sup"], label="ps_sup (P)")
    plt.plot(df_results["time_h"], df_results["na_sup"], label="na_sup (NH4-N)")
    plt.plot(df_results["time_h"], df_results["ni_sup"], label="ni_sup (NO3-N)")
    plt.xlabel("Time [h]")
    plt.ylabel("Concentration [mg/L]")
    plt.title("Supernatant concentrations vs time")
    plt.legend()
    plt.grid(True)

    

    plt.figure()
    plt.plot(df_results["time_h"], df_results["theta_max"])
    plt.xlabel("Time [h]"); plt.ylabel("theta_max [-]"); plt.grid(True)
    

    plt.figure()
    plt.plot(df_results["time_h"], df_results["eps_eff_min_sand"])
    plt.xlabel("Time [h]"); plt.ylabel("eps_eff_min_sand [-]"); plt.grid(True)
    plt.show()


   
