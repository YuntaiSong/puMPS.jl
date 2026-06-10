# Technical Report for GPT-5.5 Pro: puMPS Ising Primary-State Entanglement Computations

## 1. Purpose of the Project

This repository is being used to compute low-energy states of one-dimensional critical spin chains using periodic uniform matrix product states (puMPS), then to evaluate entanglement quantities for selected conformal primary states.

The concrete system currently studied is the critical transverse-field Ising chain with periodic boundary conditions,

```text
H = - sum_j X_j X_{j+1} - h_z sum_j Z_j - h_x sum_j X_j,
```

with parameters

```julia
hz = 1.0
hx = 0.0
```

as set in `examples/ising_example.jl` by `ISING_MODEL = (hz = 1.0, hx = 0.0)`.

The current analysis focuses only on the periodic-sector states identified as:

- `I`: vacuum / ground state.
- `sigma`: the first nontrivial primary in the PBC sector, odd under spin-flip parity.
- `epsilon`: the second primary in the PBC sector, even under spin-flip parity.

The current mutual-information and relative-entropy scans do not use the APBC `psi` / `psibar` fermion primaries.

## 2. Main Driver File and Execution Flow

The main user-facing file is:

```text
examples/ising_example.jl
```

It begins by including the package source:

```julia
include(joinpath(@__DIR__, "..", "src", "puMPS.jl"))
import .puMPS
const PM = puMPS
```

The implemented computation proceeds in this logical order:

1. Build the Ising Hamiltonian MPOs.
2. Initialize a random puMPS ground-state ansatz.
3. Optimize the puMPS ground state with VUMPS followed by local energy minimization.
4. Build the tangent-space effective Hamiltonian.
5. Solve for low-energy tangent-space excitations in the `k=0` sector.
6. Identify the first three PBC-sector states as `I`, `sigma`, and `epsilon` using energy ordering and parity.
7. Convert these compressed states to dense vectors for the finite system sizes currently used.
8. Build pairwise maximally entangled reference states for the code subspaces.
9. Compute mutual information `I(Q_1:R)` for subsystem sizes `ell = 2,3,4,5,6`.
10. Compute relative entropies between subsystem reduced density matrices.
11. Save results to CSV.
12. Plot the results in a Python notebook.

The main scan entry points are:

```julia
run_entanglement_scan(...)
run_entanglement_scan_variableD(...)
```

The variable-D scan currently uses:

```julia
const CONSERVATIVE_ISING_D_SCHEDULE = Dict(
    12 => 12,
    14 => 14,
    16 => 14,
    18 => 16,
    20 => 16,
    22 => 18,
    24 => 18,
)
```

In practice, the current dense-vector implementation successfully completed `L = 12,14,16,18,20`. It stalled at `L=22,D=18` during conversion of an excited tangent state to a full dense vector.

## 3. Package Source Organization

The top-level package file is:

```text
src/puMPS.jl
```

It defines module `puMPS`, imports dependencies, includes source files, and exports the public API.

Relevant included source files:

```text
src/MPS.jl
src/states.jl
src/z2symmetry.jl
src/tangentspace.jl
src/models.jl
```

The files used by the current Ising computation are:

- `src/MPS.jl`: base MPS/MPO tensor types, transfer matrices, dense/sparse transfer-matrix eigensolvers, MPO transfer application routines.
- `src/states.jl`: puMPS state type, dense-vector expansion, expectation values, canonicalization, VUMPS, local gradient optimization.
- `src/tangentspace.jl`: tangent-vector type, tangent-state dense-vector expansion, tangent-space metric and Hamiltonian construction, excitation eigensolver.
- `src/models.jl`: Ising MPO constructors.
- `src/z2symmetry.jl`: Z2 block structure and parity-restricted tangent-space helper code. This is present but not part of the current main `I/sigma/epsilon` scan.

## 4. Ising Hamiltonian Construction

The Ising Hamiltonian is constructed in `src/models.jl`.

Functions actually used in the PBC scan:

```julia
ising_local_MPO(::Type{T}; hz, hx)
ising_PBC_MPO_split(::Type{T}; hz, hx)
```

In `examples/ising_example.jl`, these are wrapped as:

```julia
ising_local_MPO(::Type{T}) where {T} =
    PM.ising_local_MPO(T; hz=ISING_MODEL.hz, hx=ISING_MODEL.hx)

ising_split(::Type{T}) where {T} =
    PM.ising_PBC_MPO_split(T; hz=ISING_MODEL.hz, hx=ISING_MODEL.hx)
```

`ising_local_MPO` supplies the local nearest-neighbor Hamiltonian for ground-state optimization. `ising_PBC_MPO_split` supplies the periodic Hamiltonian split into an OBC bulk part and a boundary term for tangent-space excitation calculations.

The APBC constructors also exist:

```julia
ising_APBC_MPO(...)
ising_APBC_MPO_split(...)
```

They are not used in the current `I/sigma/epsilon` mutual-information and relative-entropy scan.

## 5. puMPS Ansatz and Dense Vector Expansion

The finite periodic uniform MPS state is represented in `src/states.jl` by:

```julia
mutable struct puMPState{T}
    A::MPSTensor{T}
    N::Int
end
```

The ansatz corresponds to

```text
|psi(A)> = sum_{s_1,...,s_N} Tr[A^{s_1} A^{s_2} ... A^{s_N}]
           |s_1 s_2 ... s_N>.
```

The dense-vector expansion is implemented by:

```julia
Base.Vector(M::puMPState{T}) where T
```

This contracts the repeated tensor `A` around the periodic trace and returns a vector of length `d^N`. The code explicitly warns that this should only be used for small systems.

This dense-vector conversion is currently used downstream for all entanglement calculations. It is the main reason the current scan cannot efficiently reach `L=22` and `L=24`.

## 6. Ground-State Optimization

The main ground-state pipeline is in `examples/ising_example.jl`:

```julia
function optimize_state!(M::PM.puMPState, H::PM.MPO_open{T};
                         tol=1e-6, max_vumps=6, max_local=350, step=0.06) where {T}
    suppress_output(() -> PM.vumps_opt!(M, H, tol, maxitr=max_vumps))
    suppress_output(() -> PM.minimize_energy_local!(M, H, max_local, step=step))
    M
end
```

This calls two package-level optimizers:

```julia
vumps_opt!(M, hMPO, tol; maxitr, ncv)
minimize_energy_local!(M, hMPO, maxitr; ...)
```

Both are implemented in `src/states.jl`.

`vumps_opt!`:

- Normalizes the current state.
- Canonicalizes it with `canonicalize_left!`.
- Builds effective center-tensor and bond-matrix Hamiltonians.
- Solves local eigenproblems with KrylovKit.
- Updates the MPS tensor.

`minimize_energy_local!`:

- Computes energy derivatives.
- Converts derivatives to a physical gradient in center gauge by `gradient_central`.
- Uses a line search `line_search_energy`.
- Updates the tensor and normalizes repeatedly.

The current driver constructs the initial state by:

```julia
M = PM.rand_puMPState(ComplexF64, 2, D, N)
```

from `src/states.jl`, which uses a random unitary-like MPS tensor from `src/MPS.jl`.

## 7. Tangent-Space Excitations

The low-energy states above the optimized puMPS ground state are represented as puMPS tangent vectors.

In `src/tangentspace.jl`:

```julia
mutable struct puMPSTvec{T}
    state::puMPState{T}
    B::MPSTensor{T}
    k::Float64
end
```

The associated finite-size tangent excitation is

```text
|Phi_k(B)> = sum_{n=1}^N exp(i p n)
             sum_{s_1,...,s_N}
             Tr[A^{s_1} ... B^{s_n} ... A^{s_N}]
             |s_1 ... s_N>,
```

where

```text
p = 2 pi k / N.
```

The dense-vector expansion is implemented by:

```julia
Base.Vector(Tvec::puMPSTvec{T}) where T
```

This routine sums over all possible positions of the tangent tensor `B`. It creates large intermediate objects and is the observed bottleneck at `L=22,D=18`.

The excitation solver is:

```julia
excitations!(M, H, ks, num_states; pinv_tol=1e-10)
```

from `src/tangentspace.jl`.

It does:

1. Canonicalize the ground-state puMPS.
2. Build the tangent-space metric `G` and effective Hamiltonian `H_eff`:

   ```julia
   tangent_space_metric_and_MPO(M, H, ks, lambda_i)
   ```

3. Transform these operators to center gauge:

   ```julia
   tspace_ops_to_center_gauge!(...)
   ```

4. Solve the generalized eigenvalue problem by forming:

   ```text
   pinv(G) * H_eff
   ```

   and applying KrylovKit `eigsolve`.

5. Convert eigenvectors back into tangent tensors `B`.

The current Ising calculation requests only momentum sector:

```julia
ks = [0]
num_states = [3]
```

## 8. Computing the Three PBC States

The main state-construction function is:

```julia
compute_lowest_states(N::Int; D::Int=ISING_D, verbose::Bool=false)
```

in `examples/ising_example.jl`.

It performs:

1. Build local and periodic split Hamiltonians:

   ```julia
   Hloc = ising_local_MPO(ComplexF64)
   Hsplit = ising_split(ComplexF64)
   ```

2. Initialize and optimize ground state:

   ```julia
   M = PM.rand_puMPState(ComplexF64, 2, D, N)
   optimize_state!(M, Hloc)
   ```

3. Solve tangent-space excitations:

   ```julia
   ens, ks_full, exs = PM.excitations!(M, Hsplit, ks, num_states)
   ```

4. Sort the resulting energies and keep the three lowest objects:

   ```julia
   lowest = perm[1:3]
   ```

5. Compute Z2 parity for the ground and two excited states using:

   ```julia
   parity_MPO(ComplexF64, N)
   PM.expect(M, parity_op)
   PM.expect(exs[idx], parity_op)
   ```

The three states used downstream are:

```julia
psi0 = Vector(rel_data.state)          # I / vacuum
psi1 = Vector(rel_data.exs[idxs[2]])   # sigma
psi2 = Vector(rel_data.exs[idxs[3]])   # epsilon
```

The CSV stores:

```text
E0,E1,E2,gap_ratio,P0,P1,P2
```

where

```text
gap_ratio = (E2 - E0) / (E1 - E0).
```

For an ideal Ising identification, this ratio should approach:

```text
Delta_epsilon / Delta_sigma = 1 / (1/8) = 8,
```

and the parity pattern should be:

```text
P0 = +1, P1 = -1, P2 = +1.
```

The completed variable-D scan gives gap ratios approaching 8 and the expected parity signs.

## 9. Independent Exact-Diagonalization Validation

Small-system exact-diagonalization helpers are included in `examples/ising_example.jl`:

```julia
dense_ising_hamiltonian(N; apbc=false)
dense_translation(N; twisted=false)
joint_ising_spectrum(N; apbc=false)
identify_ising_primaries_ed(N)
run_primary_identification(...)
```

These are intended as independent checks of primary-state labels. They diagonalize the dense Ising Hamiltonian, simultaneously diagonalize translation inside degenerate energy windows, compute parity, and identify:

- `I`: PBC, parity `+1`, momentum `k=0`, scaling dimension `0`.
- `sigma`: PBC, parity `-1`, momentum `k=0`, scaling dimension `1/8`.
- `epsilon`: PBC, parity `+1`, momentum `k=0`, scaling dimension `1`.
- `psi`, `psibar`: APBC, parity `-1`, momentum `±1/2`, scaling dimension `1/2`.

The current entanglement scan uses only `I`, `sigma`, and `epsilon`.

## 10. Reduced Density Matrices

The current entanglement calculation converts states to dense vectors first.

For contiguous first-block subsystems `Q_1 = {1,...,ell}`, the reduced density matrix is computed by:

```julia
reduced_density_matrix_from_vector(psi, ell, N; d=2)
```

This reshapes:

```text
psi -> matrix of shape (2^ell, 2^(N-ell))
```

and computes:

```text
rho_Q1 = psi_matrix * psi_matrix^\dagger,
rho_Q1 = rho_Q1 / Tr(rho_Q1).
```

For arbitrary subsystems, used in mutual information with a reference qubit, the function is:

```julia
reduced_density_matrix_subsystem(psi, subset, T; d=2)
```

It:

1. Reshapes a length-`d^T` vector into a rank-`T` tensor.
2. Permutes axes so the requested subsystem is first.
3. Reshapes to `(d^|subset|, d^(T-|subset|))`.
4. Forms `rho = phi_mat * phi_mat'`.

The entropy function is:

```julia
entropy_from_rho(rho; base=2)
```

It diagonalizes `Hermitian(rho)`, clips small negative eigenvalues to zero, renormalizes, and computes:

```text
S(rho) = - sum_i lambda_i log(lambda_i)
```

with logarithm base `base`.

## 11. Maximally Entangled Reference State

The pairwise code subspace is built from two finite-system states.

For a pair `(psi_a, psi_b)`, the code constructs an orthonormal pair:

```text
phi_b = psi_b - <psi_a | psi_b> psi_a,
phi_b = phi_b / ||phi_b||.
```

Then it builds the maximally entangled physical-reference state:

```text
|Psi_RQ> = ( |0_R> tensor |psi_a>_Q
           + |1_R> tensor |phi_b>_Q ) / sqrt(2).
```

This is implemented by:

```julia
max_purification(psi_a, psi_b)
```

in `examples/ising_example.jl`.

Important convention: because Julia is column-major and the subsystem routines use `reshape`, the reference qubit is placed as site `N+1` by:

```julia
psi_a0 = kron(ket0, psi_a)
phi_b1 = kron(ket1, phi_b)
```

not by `kron(psi_a, ket0)`.

The total number of sites for the purified state is:

```text
T = N + 1,
R = {N+1}.
```

## 12. Mutual Information

The mutual information function is:

```julia
mutual_information(psi, A, C, T; d=2, base=2)
```

It computes:

```text
I(A:C) = S(A) + S(C) - S(A union C).
```

The current scan computes:

```text
I(Q_1 : R)
```

where:

```text
Q_1 = {1,...,ell}
R = {N+1}
ell = 2,3,4,5,6.
```

The helper used in the scan is:

```julia
pair_mutual_information_values(psi_a, psi_b, N, ell_values; base=2)
```

The three pairwise code subspaces are:

```text
I01 = I(Q_1:R) for span{I, sigma}
I02 = I(Q_1:R) for span{I, epsilon}
I12 = I(Q_1:R) for span{sigma, epsilon}
```

The current implementation uses base-2 entropy, so mutual information is in bits.

## 13. Relative Entropy

The relative entropy implementation is:

```julia
relative_entropy(rho, sigma)
```

It computes:

```text
S(rho || sigma) = Tr[ rho (log rho - log sigma) ].
```

The matrix logarithm helper is:

```julia
matrix_log_hermitian(rho; eps=1e-12)
```

It diagonalizes the Hermitian matrix and replaces eigenvalues smaller than `eps` by `eps` inside the logarithm. This is the current singularity-handling convention.

The current relative entropy uses the natural logarithm, not base-2 logarithms.

The scan helper is:

```julia
pair_relative_entropy_values(psi_a, psi_b, N, ell_values)
```

The three stored relative entropies are:

```text
S10 = S(rho_sigma || rho_I)
S20 = S(rho_epsilon || rho_I)
S12 = S(rho_sigma || rho_epsilon)
```

where each `rho` is the reduced density matrix on `Q_1 = {1,...,ell}`.

## 14. CSV Generation

The scan assembly function is:

```julia
entanglement_rows_for_states(psi0, psi1, psi2, N; ell_values=2:6, base=2)
```

It computes all mutual-information and relative-entropy observables for one system size.

The CSV writer is:

```julia
save_entanglement_scan_csv(path, rows)
```

The current CSV header is:

```text
N,D,ell,x,E0,E1,E2,gap_ratio,P0,P1,P2,I01,I02,I12,S10,S20,S12
```

where:

```text
x = ell / N.
```

The main produced variable-D CSV is:

```text
examples/ising_entanglement_scan_variableD.csv
```

It currently contains complete data for `N = 12,14,16,18,20`.

## 15. Plotting Notebook

The variable-D plotting notebook is:

```text
examples/Ising_entanglement_scan_variableD.ipynb
```

It:

1. Reads `examples/ising_entanglement_scan_variableD.csv`.
2. Prints a quality summary for each `L`:

   ```text
   L, D, gap_ratio, parity
   ```

3. Plots log-log mutual information data and power-law fits.
4. Plots log-log relative entropy data and power-law fits.
5. Saves:

   ```text
   examples/ising_mutual_information_variableD.pdf
   examples/ising_mutual_information_variableD.png
   examples/ising_relative_entropy_variableD.pdf
   examples/ising_relative_entropy_variableD.png
   ```

The current variable-D fit results are:

```text
Mutual information:
I-sigma:        gamma = 0.3130, A = 1.2274
I-epsilon:      gamma = 1.9268, A = 4.0985
sigma-epsilon:  gamma = 1.1324, A = 1.9402

Relative entropy:
S(sigma || I):        gamma = 2.0691, A = 0.9644
S(epsilon || I):      gamma = 3.6712, A = 20.7078
S(sigma || epsilon):  gamma = 2.7893, A = 5.1844
```

The notebook excludes the exact half-chain point from the fit by fitting only `x < 0.5`.

## 16. Current Numerical Results and Limitations

The successful variable-D systems are:

```text
L=12, D=12, gap_ratio=7.965794, parity=(+1,-1,+1)
L=14, D=14, gap_ratio=7.974941, parity=(+1,-1,+1)
L=16, D=14, gap_ratio=7.980738, parity=(+1,-1,+1)
L=18, D=16, gap_ratio=7.984794, parity=(+1,-1,+1)
L=20, D=16, gap_ratio=7.987662, parity=(+1,-1,+1)
```

This strongly supports the state assignment:

```text
psi0 = I
psi1 = sigma
psi2 = epsilon
```

The attempted `L=22,D=18` run reached the point where the three low-energy states were computed, but stalled during dense-vector conversion of an excited tangent state:

```julia
Vector(rel_data.exs[idxs[2]])
```

The relevant bottleneck is:

```julia
Base.Vector(Tvec::puMPSTvec)
```

from `src/tangentspace.jl`.

Therefore, to push beyond `L=20`, the entanglement calculation should be rewritten to avoid full dense-vector expansion. The needed next algorithmic step is to compute reduced density matrices directly from the puMPS ground tensor `A` and tangent tensor `B`, using transfer-matrix contractions.

## 17. Z2 Projection Status

The repository contains a Z2-symmetry helper file:

```text
src/z2symmetry.jl
```

and tangent-space projected eigensolver functions:

```julia
excitations_parity!(...)
solve_projected_eigs(...)
z2_allowed_indices(...)
```

However, the current main scan does not use these functions. The reason is that the current random dense `puMPState` and subsequent dense gauge transformations do not guarantee that the tensor is in the explicit Z2 block form required by the projection helper. The current `I/sigma/epsilon` calculation instead relies on ordinary dense tangent-space excitations and checks the resulting state parities afterwards.

This does not currently block the PBC `I/sigma/epsilon` calculation, but it matters if one wants symmetry-resolved excitation spaces or a reliable APBC/twist construction.

## 18. Tests

The project test entry point is:

```text
test/runtests.jl
```

It imports:

```julia
using puMPS
using puMPS.MPS
include("mps.jl")
```

The latest run after the current changes passed:

```text
MPS Tensors and Transfer Matrices:       74 / 74
MPS TM eigensolvers:                    216 / 216
MPS/MPO Tensors and Transfer Matrices:   24 / 24
```

## 19. Dependency Environment

The Julia project files are:

```text
Project.toml
Manifest.toml
```

Core dependencies used by the package:

```text
LinearAlgebra
TensorOperations
LinearMaps
KrylovKit
Optim
Random
Statistics
Printf
PyPlot
Test
```

The plotting notebook uses Python packages:

```text
csv
math
numpy
matplotlib
pathlib
```

## 20. Files GPT-5.5 Pro Should Read to Extract Explicit Formulas and Algorithms

For extracting formulas and algorithm constructions from the program, upload these files first:

```text
TECHNICAL_REPORT_FOR_GPT55_PRO.md
examples/ising_example.jl
src/puMPS.jl
src/MPS.jl
src/states.jl
src/tangentspace.jl
src/models.jl
src/z2symmetry.jl
Project.toml
```

If GPT Pro should also inspect exact generated numerical data and plotting conventions, upload:

```text
examples/ising_entanglement_scan_variableD.csv
examples/Ising_entanglement_scan_variableD.ipynb
examples/ising_primary_identification.txt
```

If GPT Pro should compare with the earlier D=8 baseline, upload:

```text
examples/ising_entanglement_scan_D8.csv
examples/Ising_entanglement_scan_D8.ipynb
```

Do not upload generated PNG/PDF figures unless the goal is visual inspection. They are derived from the CSV and notebooks.

Do not upload `examples/Ising_mutual_info.ipynb` unless specifically needed, because it contains older dirty calculation output rather than the current clean variable-D workflow.

`Manifest.toml` is optional. Upload it only if GPT Pro needs exact package versions or reproducibility details. For formula extraction, `Project.toml` is usually enough.

If GPT Pro is also expected to compare against the puMPS literature, upload the arXiv paper and source archive for:

```text
arXiv:1710.05397
arXiv:1907.10704
```

These are literature references, not program files.
