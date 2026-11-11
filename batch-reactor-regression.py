# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regression: a reversible reaction in a batch reactor
#
# **CHME 5137, homework after Lectures 12 and 13 (October 20 and 23, 2026).** Section 5 of
# the Lecture 13 notebook started this. Here it is in full.
#
# *Written for 2026 by Claude Code (Claude Opus 5.5) from Prof. West's A ⇌ B fitting
# exercise of earlier years, and checked by Prof. West.*
#
# ---
#
# You run a reversible, first-order reaction in a batch reactor:
# $$\mathrm{A} \rightleftharpoons \mathrm{B}, \qquad
# \frac{dC_A}{dt} = k_r C_B - k_f C_A, \qquad C_B = C_{A,0} - C_A$$
# Every run starts with $C_{A,0} = 10$ mol/L and no B. You measured $C_A$ every 10 s for
# 100 s, at three temperatures.
#
# **Kinetics.** The forward rate constant follows the Arrhenius expression,
# $k_f = A \exp(-E_a/RT)$.
#
# **Thermodynamics.** At equilibrium the forward and reverse rates are equal, so the
# equilibrium constant is $K = k_f/k_r$. (Assume an ideal solution, so this
# concentration-based $K$ is the thermodynamic one.) From thermodynamics,
# $$\Delta G^\circ = -RT\ln K \qquad\text{and}\qquad \Delta G^\circ = \Delta H - T\,\Delta S$$
# A and B have very similar heat capacities, so take $\Delta H$ and $\Delta S$ to be
# independent of temperature. Then $k_r = k_f/K$, and the whole model has **four
# parameters**: $\log_{10} A$, $E_a$, $\Delta H$ and $\Delta S$.
#
# Before the experiments, from the literature and a quantum chemistry calculation, your
# estimates were $\log_{10}A \approx 6$ (A in s⁻¹), $E_a \approx 45$ kJ/mol,
# $\Delta H \approx -10$ kJ/mol and $\Delta S \approx -50$ J/mol/K. Let's see what the data say.
#
# ### How to work
#
# - **Commit as you go.** After each numbered task works, Restart & Run All, then commit
#   with a message that says what you did, and push. At least one commit per task.
#   **Commit the `.ipynb` and its paired `.py` together.**
# - **Answer the questions in words**, in the markdown cells marked *Answer*.
# - **Say what your ± means.** One standard error, or a 95% interval? Either is fine. Say which (Lecture 13).
# - **Let bugs shout.** If `solve_ivp` fails, `raise` an error with its `message`.
#   Don't let a half-finished solution reach `curve_fit` without anyone noticing.

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize
import scipy.stats

plt.rcParams['figure.figsize'] = [5, 3]
R = 8.314  # J/mol/K

# %% [markdown]
# The data, the same as in class:

# %%
cA0 = 10.0   # mol/L
times = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100.])   # s
cA_data = {   # mol/L, keyed by temperature in K
    298.15: np.array([8.649, 7.441, 7.141, 6.366, 6.215, 5.990, 5.852, 5.615, 5.481, 5.644]),
    308.15: np.array([7.230, 6.073, 5.452, 5.317, 5.121, 4.998, 4.951, 4.978, 5.015, 5.036]),
    323.15: np.array([5.137, 4.568, 4.548, 4.461, 4.382, 4.525, 4.483, 4.565, 4.459, 4.635]),
}
temperatures = np.array(list(cA_data))

for T, cA in cA_data.items():
    plt.plot(times, cA, 'o:', label=f'{T:.0f} K')
plt.plot(0, cA0, 'ko')
plt.xlabel('time (s)')
plt.ylabel('$C_A$ (mol/L)')
plt.ylim(0, None)
plt.legend()
plt.show()


# %% [markdown]
# ## 1. One temperature
#
# **(a) The model.** Write `cA_model(times, kf, kr)`, which solves the ODE with
# `scipy.integrate.solve_ivp` and returns $C_A$ at `times` (use `t_eval`). In Lecture 13 we fitted
# functions that were a formula. `curve_fit` doesn't care: any function that takes the
# independent variable first and then the parameters, and returns predictions, will do. Even
# one that runs a whole simulation.
#
# Then **test it** (Lecture 11). This ODE has an exact solution,
# $$C_A(t) = C_{A,\mathrm{eq}} + (C_{A,0} - C_{A,\mathrm{eq}})\,e^{-(k_f+k_r)t},
# \qquad C_{A,\mathrm{eq}} = C_{A,0}\frac{k_r}{k_f+k_r}$$
# Check that your function agrees with it, for some reasonable $k_f$ and $k_r$, with an
# `assert`. (We use `solve_ivp` anyway because your project's model won't have an exact
# solution.)

# %%

def cA_model(times, kf, kr):
    """Calculate C_A using an ODE solver."""
    times = np.asarray(times, dtype=float)

    if np.any(times < 0):
        raise ValueError("Time cannot be negative")
    if kf < 0 or kr < 0:
        raise ValueError("Rate constants must be nonnegative")
    if times.size == 0:
        return np.array([])
    if np.max(times) == 0:
        return np.full(times.shape, cA0)

    unique_times, order = np.unique(
        times, return_inverse=True
    )

    def rhs(t, y):
        CA = y[0]
        CB = cA0 - CA
        return [kr * CB - kf * CA]

    sol = scipy.integrate.solve_ivp(
        rhs,
        (0, float(unique_times[-1])),
        [cA0],
        t_eval=unique_times,
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y[0][order]


# Test against analytical solution
t_test = np.array([0, 5, 10, 20, 50, 100.])
kf_test = 0.03
kr_test = 0.02

Ceq = cA0 * kr_test / (kf_test + kr_test)

CA_exact = Ceq + (cA0 - Ceq) * np.exp(
    -(kf_test + kr_test) * t_test
)

CA_numerical = cA_model(t_test, kf_test, kr_test)

assert np.allclose(
    CA_numerical, CA_exact,
    rtol=1e-6, atol=1e-7
)

print("Task 1(a): ODE test passed")

# your test here


# %% [markdown]
# **(b) Fit it at 298.15 K**, with `scipy.optimize.curve_fit`. **Try it without `p0` first.**
# It may fail with an error from deep inside SciPy, or it may appear to succeed. Either way,
# find out what happened: print and plot what came back, or what `cA_model` returns
# at `curve_fit`'s default starting point. Then give it a starting guess you can justify
# from the data. Roughly how long does $C_A$ take to get halfway to where it ends up?

# %%

import warnings

# Try fitting without an initial guess
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default_fit, default_cov = scipy.optimize.curve_fit(
            cA_model,
            times,
            cA_data[298.15],
            maxfev=3000
        )

    print("Default fit:", default_fit)

    for warning in caught:
        print("Warning:", warning.message)

except (ValueError, RuntimeError, TypeError) as error:
    print("Fit without p0 failed:", error)

# Examine default starting values
default_prediction = cA_model(times, 1.0, 1.0)

plt.plot(times, cA_data[298.15], "ko",
         label="Experimental data")
plt.plot(times, default_prediction, "r--",
         label="Default starting guess")
plt.xlabel("Time (s)")
plt.ylabel("C_A (mol/L)")
plt.legend()
plt.show()


# %%

# Estimate initial parameters from the data
Ceq_guess = 5.5
half_time_guess = 18.0

rate_sum_guess = np.log(2) / half_time_guess

kr_guess = rate_sum_guess * Ceq_guess / cA0
kf_guess = rate_sum_guess - kr_guess

print("Initial guesses:")
print("kf =", kf_guess)
print("kr =", kr_guess)

# Fit the model
popt_298, pcov_298 = scipy.optimize.curve_fit(
    cA_model,
    times,
    cA_data[298.15],
    p0=[kf_guess, kr_guess],
    bounds=(0, np.inf),
    maxfev=20000
)

# Calculate parameter uncertainties
perr_298 = np.sqrt(np.diag(pcov_298))

print("Fitted parameters at 298.15 K:")
print(f"kf = {popt_298[0]:.6f} +/- {perr_298[0]:.6f} 1/s")
print(f"kr = {popt_298[1]:.6f} +/- {perr_298[1]:.6f} 1/s")


# %% [markdown]
# **(c) Look at it.** Plot the data and the fitted curve, and below it the residuals
# (data minus model). Report $k_f$ and $k_r$ with their uncertainties.
#
# What happened without `p0`, and why? What starting guess did you use, and how did you
# choose it?

# %%

# Generate fitted concentrations
t_dense = np.linspace(0, 100, 301)

model_298 = cA_model(times, *popt_298)
residual_298 = cA_data[298.15] - model_298

# Create two plots
fig, axes = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(6, 5),
    gridspec_kw={"height_ratios": [2, 1]}
)

# Experimental data and fitted curve
axes[0].plot(times, cA_data[298.15], "ko",
             label="Experimental data")

axes[0].plot(
    t_dense,
    cA_model(t_dense, *popt_298),
    "b-",
    label="Fitted model"
)

axes[0].set_ylabel("C_A (mol/L)")
axes[0].legend()

# Residual plot
axes[1].plot(times, residual_298, "bo")
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Data - Model (mol/L)")

plt.tight_layout()
plt.show()

# %% [markdown]
# *Answer:*
# The numerical ODE solution agrees with the analytical
# solution for the reversible first-order reaction.
# The assertion test confirms the accuracy of solve_ivp.
#
# Without specifying p0, curve_fit starts with
# kf = kr = 1 s^-1. These values predict that the
# concentration reaches approximately 5 mol/L very
# quickly, which differs from the experimental data.
# In this calculation, the fit without p0 failed
# because the optimizer attempted a negative rate
# constant, which was rejected by the model.
#
# I estimated the equilibrium concentration as
# approximately 5.5 mol/L and the half-time as 18 s.
# From these estimates, the initial guesses were
# kf = 0.0173 s^-1 and kr = 0.0212 s^-1.
#
# The fitted rate constants at 298.15 K are:
# kf = 0.01680 +/- 0.00079 s^-1
# kr = 0.02006 +/- 0.00154 s^-1
#
# The uncertainties represent one standard error
# from the covariance matrix, not 95% intervals.
#
# The fitted concentration curve follows the measured
# data closely. The residuals fluctuate around zero,
# indicating a reasonable fit without a strong
# systematic pattern.


# %% [markdown]
# ## 2. Three temperatures, in two steps
#
# **(a)** Fit $k_f$ and $k_r$ separately at each temperature, and make a small table of
# $k_f$, $k_r$ (each with its ±) and $K = k_f/k_r$.
#
# **Before fitting anything else:** from your $K$ values alone, is the reaction endothermic
# or exothermic? How can you tell?

# %%

# %% [markdown]
# **(b) Arrhenius.** As in Lecture 13: `scipy.stats.linregress` on $\ln k_f$ against
# $1/T$ gives $E_a$ from the slope and $\ln A$ from the intercept. Convert $\ln A$ to
# $\log_{10} A$.
#
# **(c) van 't Hoff.** Put the two expressions for $\Delta G^\circ$ together and divide by $-RT$:
# $$\ln K = -\frac{\Delta H}{R}\,\frac{1}{T} + \frac{\Delta S}{R}$$
# So $\ln K$ against $1/T$ is a straight line too, with slope $-\Delta H/R$ and intercept
# $\Delta S/R$. Fit it, and get $\Delta H$ and $\Delta S$, with their ±. Does the sign of
# $\Delta H$ agree with what you said in (a)?

# %%

# %% [markdown]
# **(d)** `linregress` gave you a ± on each of these. What does that ± know about,
# and what has it left out? Think about how many points each straight line had, and about
# the ± you got in (a).

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## 3. All the data at once
#
# Step 2 fitted 3 × 2 rate constants, then fitted lines through three points each. Better:
# fit the four parameters you actually want **directly to all 30 measurements**, so that
# every measurement counts and the uncertainties carry all the way through.
#
# `curve_fit` wants one `xdata` and one `ydata`. With several experiments, stack them:
# `ydata` is all 30 concentrations in one array, and `xdata` carries **both** the
# temperature and the time of each one. `curve_fit` never looks inside `xdata`; it just passes it
# to your function. So it can be a tuple of two arrays:

# %%
T_all = np.repeat(temperatures, len(times))     # 298.15 ten times, then 308.15 ten times, ...
t_all = np.tile(times, len(temperatures))        # 10, 20, ..., 100, three times over
cA_all = np.concatenate([cA_data[T] for T in temperatures])
x_all = (T_all, t_all)
print(T_all[:12], t_all[:12], cA_all[:12], sep='\n')


# %% [markdown]
# **(a)** Write `cA_model_all(x, log10A, Ea, dH, dS)`. Unpack `x` into temperatures and
# times. For each temperature, work out $k_f$, $K$ and $k_r$, then call your `cA_model` for
# that temperature's times, and put the results in the right places of the output array.
# Solve one ODE per temperature, not one per data point. A boolean mask such as
# `T_col == T` picks out the rows.
#
# Notice the **units**: $E_a$ and $\Delta H$ in kJ/mol, $\Delta S$ in J/mol/K, and $A$ as
# $\log_{10}A$. That keeps all four parameters between about 1 and 100. An optimizer
# has an easier time when its parameters are all a similar size than when one is
# $10^{9}$ and another is $10^{-2}$. So convert carefully inside the function.
#
# Then fit it, **starting from your answers to task 2**, and report the four parameters with
# their ±.

# %%
def cA_model_all(x, log10A, Ea, dH, dS):
    """C_A (mol/L) for every (T, t) pair in x. Ea and dH in kJ/mol, dS in J/mol/K."""
    T_col, t_col = x
    


# %%

# %% [markdown]
# **(b)** Now start from your estimates from before the experiments instead (6, 45, −10, −50).
# Does it find the same answer? How far did the data move you from what you expected?

# %%

# %% [markdown]
# **(c)** Compare with task 2. Are the centres the same? Are the error bars? Which set of
# error bars would you believe, and why?
#
# **(d) The correlation matrix.** The off-diagonal elements of `pcov` say how the errors in
# the parameters go together (Lecture 13). Turn the covariance into correlations:
# `pcov / np.outer(perr, perr)`. Which pairs are strongly correlated? Why those pairs?
# (Think about Lecture 13's ridge, and about the range of temperatures you measured over.)
# If someone quotes only your four ± values, what are they missing?

# %%

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## 4. Look at the residuals
#
# **Always look at the residuals** (Lecture 13). Plot the residuals of the task 3 fit against
# time, one colour per temperature, with a line at zero.
#
# - Do you see a pattern, or does it look like noise?
# - Is the scatter about the same width at all three temperatures? Least squares assumed it is.
# - Nobody told you the measurement uncertainty, so `curve_fit` (with
#   `absolute_sigma=False`, the default) estimated it from the scatter of the residuals.
#   Work out that estimate yourself:
#   $s = \sqrt{\sum_i r_i^2 / (N - p)}$, with $N$ data points and $p$ parameters. What does
#   it say about how precisely $C_A$ was measured?
# - Suppose the experimentalist now tells you their instrument is good to ±0.01 mol/L.
#   What would that tell you about your model, or about the experiment?

# %%

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## 5. Your project
#
# One paragraph. Which parameters in your project's model will come from fitting data, and
# where will the data come from: your own simulations, the literature, a research group? What
# can you say about the errors in those data? Are they the same absolute size, the same
# relative size, or known point by point? And are any of your parameters likely to be
# correlated, the way $E_a$ and $A$ are here?

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## Stretch problems (optional)
#
# **(a) Untangle the ridge.** Write $k_f$ in terms of its value at a reference temperature in
# the middle of your data, say $T_\mathrm{ref} = 310$ K:
# $$k_f(T) = k_\mathrm{ref}\exp\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_\mathrm{ref}}\right)\right]$$
# It's the same model, written differently. Fit $\log_{10}k_\mathrm{ref}$, $E_a$, $\Delta H$
# and $\Delta S$, and look at the correlation matrix again. What changed, and why? Can you do
# the same for $\Delta H$ and $\Delta S$?
#
# **(b) Draw the posterior.** At 298.15 K only, put $k_f$ and $k_r$ on a 2-D grid and draw
# their posterior, as in Lecture 13, part 5, with a flat prior and Gaussian errors of width
# $s$ from task 4. Do its 95% credible intervals (summed over the other parameter) agree with
# `curve_fit`'s 95% confidence intervals? Why should they, or why not?

# %%

# %%

# %% [markdown]
# ---
#
# ### When you're done
#
# Restart & Run All. Check that every cell ran, in order, with no errors, and that the plots
# are there. Commit and push. Then check on GitHub that your fork shows your latest commit.
