# Load the puMPS code from the src/ folder (one directory up from examples/)
include(joinpath(@__DIR__, "..", "src", "puMPS.jl"))
import .puMPS

using Printf
using LinearAlgebra
using PyPlot
using Statistics
using Random

const PM = puMPS

# Basic simulation parameters
const ISING_D =8
const ISING_MODEL = (hz = 1.0, hx = 0.0)
const CONSERVATIVE_ISING_D_SCHEDULE = Dict(
    12 => 12,
    14 => 14,
    16 => 14,
    18 => 16,
    20 => 16,
    22 => 18,
    24 => 18,
)

ising_local_MPO(::Type{T}) where {T} = PM.ising_local_MPO(T; hz=ISING_MODEL.hz, hx=ISING_MODEL.hx)
ising_split(::Type{T}) where {T} = PM.ising_PBC_MPO_split(T; hz=ISING_MODEL.hz, hx=ISING_MODEL.hx)

const DEVNULL = devnull

function suppress_output(f::Function)
    redirect_stdout(DEVNULL) do
        redirect_stderr(DEVNULL) do
            return f()
        end
    end
end

function parity_MPO(::Type{T}, N::Int) where {T}
    # On-site parity operator P = ∏ σᵢᶻ.
    Z = Matrix{T}(I, 2, 2)
    Z[2, 2] *= -one(T)

    tens = zeros(T, 1, 2, 1, 2)  # MPOTensor indices: (m1, ket, m2, bra)
    tens[1, :, 1, :] .= Z

    site = convert(PM.MPOTensor{T}, tens)
    PM.MPOTensor{T}[site for _ in 1:N]
end

function optimize_state!(M::PM.puMPState, H::PM.MPO_open{T};
                         tol=1e-6, max_vumps=6, max_local=350, step=0.06) where {T}
    suppress_output(() -> PM.vumps_opt!(M, H, tol, maxitr=max_vumps))
    suppress_output(() -> PM.minimize_energy_local!(M, H, max_local, step=step))
    M
end

function compute_lowest_states(N::Int; D::Int=ISING_D, verbose::Bool=false)
    verbose && println("=== Ising chain with N = $N, D = $D ===")
    Hloc = ising_local_MPO(ComplexF64)
    Hsplit = ising_split(ComplexF64)

    M = PM.rand_puMPState(ComplexF64, 2, D, N)
    optimize_state!(M, Hloc)

    ks = [0]
    num_states = [3]
    ens, ks_full, exs = suppress_output(() -> PM.excitations!(M, Hsplit, ks, num_states))

    energies = real.(ens)
    perm = sortperm(energies)
    lowest = perm[1:3]

    if verbose
        println("Three lowest-energy eigenstates (NS sector):")
        for idx in lowest
            @printf("  k = %3d  E = %.12f\n", ks_full[idx], energies[idx])
        end
    end

    parity_op = parity_MPO(ComplexF64, N)
    parities = Float64[]
    P0 = PM.expect(M, parity_op)
    push!(parities, real(P0))
    for idx in lowest[2:3]
        Pn = PM.expect(exs[idx], parity_op)
        push!(parities, real(Pn))
    end

    (state=M, energies=energies, ks=ks_full, exs=exs,
     indices=lowest, parities=parities)
end

function ratio_scaling_analysis(Ns::Vector{Int}; D::Int=ISING_D)
    data = NamedTuple[]
    for N in Ns
        res = compute_lowest_states(N; D=D, verbose=false)
        idxs = res.indices
        E0, E1, E2 = res.energies[idxs[1]], res.energies[idxs[2]], res.energies[idxs[3]]
        gap = E1 - E0
        ratio = (E2 - E0) / gap
        push!(data, (N=N, E0=E0, E1=E1, E2=E2, ratio=ratio,
                     parities=res.parities, energies=res.energies,
                     ks=res.ks, ground_idx=idxs[1]))
    end

    invN2 = [1.0 / (d.N^2) for d in data]
    ratios = [d.ratio for d in data]
    A = hcat(ones(length(invN2)), invN2)
    coeffs = A \ ratios
    ratio_inf = coeffs[1]
    (fit=ratio_inf, coeffs=coeffs, raw=data)
end

function save_ratio_results(path::AbstractString, ratio_info)
    open(path, "w") do io
        println(io, "Ising ratio analysis (ε gap units)")
        println(io, @sprintf("%6s %18s %18s %18s %10s", "N", "E0", "E1", "E2", "ratio"))
        println(io, "-"^70)
        for d in ratio_info.raw
            println(io, @sprintf("%6d %18.12f %18.12f %18.12f %10.6f",
                                 d.N, d.E0, d.E1, d.E2, d.ratio))
            println(io, @sprintf("       Parity: P0=%.6f  P1=%.6f  P2=%.6f",
                                 d.parities[1], d.parities[2], d.parities[3]))
        end
        println(io, "\nLinear fit ratio ≈ r∞ + a / N²:")
        println(io, @sprintf("  r∞ ≈ %.6f  (a = %.6f)", ratio_info.coeffs[1], ratio_info.coeffs[2]))
    end
end

function save_conformal_tower_plot(data, dir::AbstractString)
    for d in data
        fig = figure()
        energies = real.(d.energies)
        rel = energies .- d.E0
        scatter(d.ks, rel, c="C0")
        xlabel("Momentum sector k")
        ylabel("E - E₀")
        title("Ising conformal tower (N=$(d.N))")
        tight_layout()
        outfile = joinpath(dir, "ising_conformal_tower_N$(d.N).png")
        savefig(outfile)
        close(fig)
    end
end


function plot_conformal_tower(periodic, v_est; outfile=joinpath(@__DIR__, "ising_tower.png"))
    energies = periodic.energies
    ks = periodic.ks
    vacuum_idx = periodic.vacuum_idx

    E0 = energies[vacuum_idx]
    prefactor = ISING_N / (2π * v_est)
    deltas = prefactor .* (energies .- E0)

    fig = figure()
    scatter(ks, deltas, c="C0")
    xlabel("momentum sector k")
    ylabel("Δ_eff")
    title("Conformal tower (Ising, periodic sector)")
    grid(true, linestyle="--", linewidth=0.4, alpha=0.5)
    savefig(outfile)
    close(fig)
    deltas
end

function grow_block_forward(block::AbstractMatrix{T}, A::Array{T,3}) where {T}
    m_prev, D_left = size(block)
    D_left == size(A, 1) || error("Left dimension mismatch while growing block.")
    d = size(A, 2)
    D_right = size(A, 3)
    new_block = zeros(T, m_prev * d, D_right)
    for s in 1:d
        rows = (s - 1) * m_prev + 1 : s * m_prev
        @views new_block[rows, :] .= block * view(A, :, s, :)
    end
    new_block
end

function reduced_density_matrix_from_vector(ψ::AbstractVector{T}, ℓ::Int, N::Int; d::Int=2) where {T}
    ℓ >= 1 || error("Block size ℓ must be ≥ 1.")
    ℓ <= N || error("Block size ℓ cannot exceed system size.")
    left_dim = d^ℓ
    right_dim = d^(N - ℓ)
    length(ψ) == left_dim * right_dim || error("State vector length incompatible with block size.")
    ψ_tensor = reshape(ψ, (left_dim, right_dim))
    ρ = ψ_tensor * ψ_tensor'
    ρ ./= real(tr(ρ))
end

function matpow(mat::AbstractMatrix{T}, p::Int) where {T}
    n = p
    res = Matrix{T}(I, size(mat, 1), size(mat, 2))
    base = copy(mat)
    while n > 0
        if (n & 1) == 1
            res = res * base
        end
        base = base * base
        n >>= 1
    end
    return res
end

# 把 [1, left_dim] 的索引映射到长度 ℓ 的自旋配置 (每个在 1..d)
function index_to_digits(idx::Int, ℓ::Int, base::Int)
    digits = Vector{Int}(undef, ℓ)
    x = idx - 1  # 0-based
    for j in 1:ℓ
        digits[j] = (x % base) + 1
        x ÷= base
    end
    return digits
end

function reduced_density_matrix_first_block(M::PM.puMPState{T}, ℓ::Int, N::Int;
                                            d::Int = 2) where {T<:Complex}
    ℓ >= 1 || error("Block size ℓ must be ≥ 1.")
    ℓ <= N || error("Block size ℓ cannot exceed system size.")

    # 用 puMPS 自己的 num_sites 再确认一下 N
    N_M = PM.num_sites(M)
    if N_M != N
        @warn "Provided N=$N does not match num_sites(M)=$N_M, using num_sites(M)."
        N = N_M
    end

    A = PM.mps_tensor(M)  # 预期维度 (D, d, D)
    D = size(A, 1)
    dA = size(A, 2)
    dA == d || error("Physical dimension mismatch: got dA=$dA, expected d=$d.")

    D2 = D * D
    # A_s: D×D 矩阵列表
    As = [ @view(A[:, s, :]) for s in 1:d ]

    # double-layer 转移矩阵 E_env = sum_s A^s ⊗ conj(A^s)
    T2 = promote_type(T, ComplexF64)
    E_env = zeros(T2, D2, D2)
    for s in 1:d
        E_env .+= kron(Matrix{T2}(As[s]), conj(Matrix{T2}(As[s])))
    end

    # 预先算好 E_env^N 和 E_env^(N-ℓ)
    E_env_N = matpow(E_env, N)
    Z = real(tr(E_env_N))  # 归一化
    E_env_tail = matpow(E_env, N - ℓ)

    # E_st[s,t] = A^s ⊗ (A^t)*
    E_st = [ [zeros(T2, D2, D2) for t in 1:d] for s in 1:d ]
    for s in 1:d, t in 1:d
        E_st[s][t] .= kron(Matrix{T2}(As[s]), conj(Matrix{T2}(As[t])))
    end

    left_dim = d^ℓ
    ρ = zeros(T2, left_dim, left_dim)

    for i in 1:left_dim
        sconf = index_to_digits(i, ℓ, d)  # 长度 ℓ，每个在 1..d
        for j in 1:left_dim
            tconf = index_to_digits(j, ℓ, d)

            Mmat = Matrix{T2}(I, D2, D2)
            @inbounds for k in 1:ℓ
                Mmat = Mmat * E_st[sconf[k]][tconf[k]]
            end
            Mmat = Mmat * E_env_tail
            ρ[i, j] = tr(Mmat)
        end
    end

    ρ ./= Z
end

function reduced_density_matrix_first_block(Tvec::PM.puMPSTvec{T}, ℓ::Int, N::Int) where {T}
    ψ = Vector(Tvec)
    normψ = norm(ψ)
    normψ ≈ 0 && error("Excited state vector has zero norm.")
    ψ ./= normψ
    reduced_density_matrix_from_vector(ψ, ℓ, N)
end

"""
    reduced_density_matrix_subsystem(ψ, subset, T; d=2)

Return the reduced density matrix ρ_subset for a pure state vector `ψ`
on `T` sites (local dimension `d`), tracing out all sites not in `subset`.

- `ψ` is a complex vector of length d^T
- `subset` is a vector of site indices (1-based, between 1 and T)
"""
function reduced_density_matrix_subsystem(ψ::AbstractVector{S},
                                          subset::Vector{Int},
                                          T::Int; d::Int=2) where {S<:Complex}
    # Basic checks
    length(ψ) == d^T ||
        error("State vector length $(length(ψ)) incompatible with T=$T and d=$d.")
    all(1 .<= subset .<= T) ||
        error("Subset indices must be between 1 and T.")
    length(unique(subset)) == length(subset) ||
        error("Subset indices must not contain duplicates.")

    m = length(subset)
    dim_A = d^m
    dim_tot = d^T
    dim_B = div(dim_tot, dim_A)

    # Reshape ψ into rank-T tensor
    shape = ntuple(_ -> d, T)  # (d,d,...,d) length T
    φ = reshape(ψ, shape)

    # Build permutation: [subset..., complement...]
    all_sites = collect(1:T)
    complement = setdiff(all_sites, subset)
    perm = vcat(subset, complement)

    # Permute so that subset sits in the first m indices
    φ_perm = permutedims(φ, perm)

    # Reshape to (dim_A, dim_B)
    φ_mat = reshape(φ_perm, dim_A, dim_B)

    # Partial trace over complement: ρ_A = φ φ†
    ρ = φ_mat * φ_mat'
    ρ ./= real(tr(ρ))   # normalize numerically

    return ρ
end

"""
    entropy_from_rho(ρ; base=2)

Von Neumann entropy S(ρ) = -Tr(ρ log ρ) in the given logarithm base (default base 2).
Assumes ρ is positive semidefinite; small negatives from numerics are clipped to zero.
"""
function entropy_from_rho(ρ::AbstractMatrix{T}; base::Real=2) where {T<:Complex}
    vals = eigvals(Hermitian(ρ))
    λ = real.(vals)
    λ .= max.(λ, 0.0)              # clip small negative eigenvalues
    s = sum(λ)
    s ≈ 0 && return 0.0
    λ ./= s                         # renormalize

    mask = λ .> 0
    λnz = λ[mask]
    return -sum(λnz .* (log.(λnz) ./ log(base)))
end

"""
    entropy_subsystem(ψ, subset, T; d=2, base=2)

Von Neumann entropy of the reduced density matrix on `subset`
for a pure state vector `ψ` on `T` sites.
"""
function entropy_subsystem(ψ::AbstractVector{S},
                           subset::Vector{Int},
                           T::Int; d::Int=2, base::Real=2) where {S<:Complex}
    ρA = reduced_density_matrix_subsystem(ψ, subset, T; d=d)
    return entropy_from_rho(ρA; base=base)
end


"""
    mutual_information(ψ, A, C, T; d=2, base=2)

Mutual information I(A:C) = S(A) + S(C) - S(A ∪ C)
for a pure state vector ψ on T sites.

- `A`, `C` are vectors of site indices (1-based).
"""
function mutual_information(ψ::AbstractVector{S},
                            A::Vector{Int},
                            C::Vector{Int},
                            T::Int; d::Int=2, base::Real=2) where {S<:Complex}
    SA = entropy_subsystem(ψ, A, T; d=d, base=base)
    SC = entropy_subsystem(ψ, C, T; d=d, base=base)
    AC = sort(union(A, C))
    SAC = entropy_subsystem(ψ, AC, T; d=d, base=base)
    return SA + SC - SAC
end

function matrix_log_hermitian(ρ::AbstractMatrix{T}; eps::Float64=1e-12) where {T<:Complex}
    vals, vecs = eigen(Hermitian(ρ))
    log_vals = similar(vals)
    @inbounds for i in eachindex(vals)
        λ = real(vals[i])
        log_vals[i] = λ <= eps ? log(eps) : log(λ)
    end
    vecs * Diagonal(log_vals) * vecs'
end

function relative_entropy(ρ::AbstractMatrix{T}, σ::AbstractMatrix{T}) where {T<:Complex}
    size(ρ) == size(σ) || error("Density matrices must have the same size.")
    logρ = matrix_log_hermitian(ρ)
    logσ = matrix_log_hermitian(σ)
    real(tr(ρ * (logρ - logσ)))
end

function analyze_relative_entropy_Q1(M0::PM.puMPState{ComplexF64},
                                     M1::Union{PM.puMPState{ComplexF64}, PM.puMPSTvec{ComplexF64}},
                                     M2::Union{PM.puMPState{ComplexF64}, PM.puMPSTvec{ComplexF64}};
                                     ℓ_max::Int=6,
                                     N::Int=ISING_N,
                                     outfile::String=joinpath(@__DIR__, "Srel_Q1_vs_x.png"))
    ℓs = collect(1:ℓ_max)
    xs = [ℓ / N for ℓ in ℓs]
    Srel1 = zeros(Float64, ℓ_max)
    Srel2 = zeros(Float64, ℓ_max)
    Srel12 = zeros(Float64, ℓ_max)
    for (idx, ℓ) in enumerate(ℓs)
        ρ0 = reduced_density_matrix_first_block(M0, ℓ, N)
        ρ1 = reduced_density_matrix_first_block(M1, ℓ, N)
        ρ2 = reduced_density_matrix_first_block(M2, ℓ, N)
        Srel1[idx] = relative_entropy(ρ1, ρ0)
        Srel2[idx] = relative_entropy(ρ2, ρ0)
        Srel12[idx] = relative_entropy(ρ1, ρ2)
    end
    function fit_power_law(xs::Vector{Float64}, ys::Vector{Float64})
        mask = ys .> 0
        xs_pos = xs[mask]
        ys_pos = ys[mask]
        logx = log.(xs_pos)
        logy = log.(ys_pos)
        X = hcat(ones(length(logx)), logx)
        β = X \ logy
        A = exp(β[1])
        α = β[2]
        return A, α
    end
    A1, α1 = fit_power_law(xs, Srel1)
    A2, α2 = fit_power_law(xs, Srel2)
    A12, α12 = fit_power_law(xs, Srel12)
    println("Fit for first excited state: S₁(x) ≈ $(A1) * x^$(α1)")
    println("Fit for second excited state: S₂(x) ≈ $(A2) * x^$(α2)")
    println("Fit between excited states: S₁₂(x) ≈ $(A12) * x^$(α12)")
    figure()
    loglog(xs, Srel1, "o-", label="S(ρ₁ || ρ₀)")
    loglog(xs, Srel2, "s-", label="S(ρ₂ || ρ₀)")
    mask12 = Srel12 .> 0
    loglog(xs[mask12], Srel12[mask12], "d-", label="S(ρ₁ || ρ₂)")
    xs_fit = range(minimum(xs), maximum(xs), length=200)
    ys1_fit = A1 .* (xs_fit .^ α1)
    ys2_fit = A2 .* (xs_fit .^ α2)
    ys12_fit = A12 .* (xs_fit .^ α12)
    lbl1 = @sprintf("fit M1: A₁=%.3e α₁=%.3f", A1, α1)
    lbl2 = @sprintf("fit M2: A₂=%.3e α₂=%.3f", A2, α2)
    lbl12 = @sprintf("fit 1↔2: A₁₂=%.3e α₁₂=%.3f", A12, α12)
    loglog(xs_fit, ys1_fit, "-", label=lbl1)
    loglog(xs_fit, ys2_fit, "-", label=lbl2)
    loglog(xs_fit, ys12_fit, "-", label=lbl12)
    xlabel("x = |Q₁| / N")
    ylabel("S(ρ_i || ρ₀)")
    title("Relative entropy on Q₁")
    legend()
    grid(true, which="both", linestyle="--", linewidth=0.5)
    tight_layout()
    savefig(outfile)
    close()
    (ℓs=ℓs, x=xs, Srel1=Srel1, Srel2=Srel2, Srel12=Srel12,
     A1=A1, α1=α1, A2=A2, α2=α2, A12=A12, α12=α12)
end

function save_relative_entropy_results(path::AbstractString, rel_infos::Vector)
    open(path, "w") do io
        println(io, "Relative entropies on Q₁ for multiple system sizes")
        println(io, @sprintf("%6s %6s %12s %18s %18s %18s",
                             "N", "|Q₁|", "x = ℓ/N", "S(ρ₁||ρ₀)", "S(ρ₂||ρ₀)", "S(ρ₁||ρ₂)"))
        println(io, "-"^100)
        for entry in rel_infos
            N = entry.N
            info = entry.info
            for (idx, x) in enumerate(info.x)
                ℓ = info.ℓs[idx]
                s1 = info.Srel1[idx]
                s2 = info.Srel2[idx]
                s12 = info.Srel12[idx]
                println(io, @sprintf("%6d %6d %12.6f %18.10e %18.10e %18.10e",
                                     N, ℓ, x, s1, s2, s12))
            end
        end
        println(io, "\nPower-law fits per system size:")
        for entry in rel_infos
            N = entry.N
            info = entry.info
            println(io, "N = $N:")
            println(io, @sprintf("  S₁(x) ≈ %.6e * x^{%.6f}", info.A1, info.α1))
            println(io, @sprintf("  S₂(x) ≈ %.6e * x^{%.6f}", info.A2, info.α2))
            println(io, @sprintf("  S₁₂(x) ≈ %.6e * x^{%.6f}", info.A12, info.α12))
        end
    end
end



function check_rdm_consistency(N::Int; ℓ_max::Int=2, D::Int=ISING_D)
    Hloc = ising_local_MPO(ComplexF64)
    M = PM.rand_puMPState(ComplexF64, 2, D, N)
    optimize_state!(M, Hloc)

    # brute-force vector
    ψ = Vector(M)  

    for ℓ in 1:ℓ_max
        ρ_vec = reduced_density_matrix_from_vector(ψ, ℓ, N)
        ρ_mps = reduced_density_matrix_first_block(M, ℓ, N)
        diff = maximum(abs.(ρ_vec .- ρ_mps))
        ev_diff = maximum(abs.(eigvals(ρ_vec) .- eigvals(ρ_mps)))
        @printf("  ℓ = %d: ‖ρ_MPS - ρ_vec‖_∞ = %.3e, eigenvalue diff = %.3e\n",
                ℓ, diff, ev_diff)
    end
end


# check_reduced_dm_consistency(8; D=ISING_D, ℓs=[1,2])


############################################################
# Helper: maximally entangled purification for a pair of states
############################################################

"""
    max_purification(ψa, ψb)

Given two normalized states ψa, ψb (length 2^N), build a maximally
entangled purification on QR:

    |ψ_max⟩ = ( |0⟩_R⊗|ψ_a⟩_Q + |1⟩_R⊗|φ_b⟩_Q ) / √2

where |φ_b⟩ is the component of |ψ_b⟩ orthogonal to |ψ_a⟩.
With the column-major site convention used by `reshape`, this puts the
reference qubit on site N+1 and keeps the physical chain on sites 1,...,N.
The output lives on N+1 sites (dimension 2^(N+1)).
"""
function max_purification(ψa::Vector{ComplexF64},
                          ψb::Vector{ComplexF64})
    # Ensure both are normalized
    ψa_norm = norm(ψa)
    ψb_norm = norm(ψb)
    ψa_norm ≈ 0 && error("ψa has zero norm.")
    ψb_norm ≈ 0 && error("ψb has zero norm.")
    ψa = ψa ./ ψa_norm
    ψb = ψb ./ ψb_norm

    # Gram-Schmidt: make φ_b orthogonal to ψ_a
    proj = dot(ψa, ψb)
    φb = ψb .- proj .* ψa
    nb = norm(φb)
    nb < 1e-12 && error("ψa and ψb are (almost) linearly dependent.")
    φb ./= nb

    ket0 = ComplexF64[1.0, 0.0]
    ket1 = ComplexF64[0.0, 1.0]

    ψa0 = kron(ket0, ψa)
    φb1 = kron(ket1, φb)
    ψmax = (ψa0 .+ φb1) ./ sqrt(2.0)

    return ψmax
end

############################################################
# Helper: compute I(Q₁ : R) vs ℓ for a single pair of primaries
############################################################

"""
    analyze_I_Q1R_pair(ψa, ψb, N; ℓ_max, base=2)

Given two states ψa, ψb on N spins (dimension 2^N),
build the maximally entangled purification ψ_max on N+1 sites and
compute the mutual information

    I(Q₁ : R)

for Q₁ = {1,…,ℓ} and R = {N+1}, for ℓ = 1,…,ℓ_max.

Returns:
    (ℓs = 1:ℓ_max, xs = ℓs ./ N, Ivals = I(Q₁:R) for each ℓ)
"""
function analyze_I_Q1R_pair(ψa::Vector{ComplexF64},
                            ψb::Vector{ComplexF64},
                            N::Int; ℓ_max::Int,
                            base::Real = 2)
    # Build purification on N+1 sites
    ψmax = max_purification(ψa, ψb)
    T = N + 1               # total sites (physical + reference)
    ref_site = T            # reference is the last site

    ℓs = collect(1:ℓ_max)
    xs = [ℓ / N for ℓ in ℓs]
    Ivals = zeros(Float64, ℓ_max)

    for (i, ℓ) in enumerate(ℓs)
        Q1 = collect(1:ℓ)   # Q₁ = first ℓ sites
        R  = [ref_site]     # reference site
        Ivals[i] = mutual_information(ψmax, Q1, R, T; base=base)
    end

    return (ℓs = ℓs, xs = xs, Ivals = Ivals)
end

function save_I_Q1R_results(path::AbstractString, entries)
    open(path, "w") do io
        println(io, "Mutual information I(Q₁ : R) results")
        println(io, @sprintf("%6s %8s %6s %12s %18s", "N", "pair", "|Q₁|", "x = ℓ/N", "I(Q₁:R)"))
        println(io, "-"^70)
        for entry in entries
            N = entry.N
            pairs = [("0-1", entry.pair01), ("0-2", entry.pair02), ("1-2", entry.pair12)]
            for (label, info) in pairs
                for idx in eachindex(info.ℓs)
                    ℓ = info.ℓs[idx]
                    x = info.xs[idx]
                    Ival = info.Ivals[idx]
                    println(io, @sprintf("%6d %8s %6d %12.6f %18.10e", N, label, ℓ, x, Ival))
                end
            end
        end
    end
end

function normalize_state_vector(ψ::AbstractVector{<:Complex})
    ψn = Vector{ComplexF64}(ψ)
    nrm = norm(ψn)
    nrm ≈ 0 && error("State vector has zero norm.")
    ψn ./= nrm
end

function pair_mutual_information_values(ψa::Vector{ComplexF64},
                                        ψb::Vector{ComplexF64},
                                        N::Int,
                                        ell_values::AbstractVector{<:Integer};
                                        base::Real=2)
    ψmax = max_purification(ψa, ψb)
    T = N + 1
    R = [T]
    vals = Float64[]
    for ell in ell_values
        Q1 = collect(1:ell)
        push!(vals, mutual_information(ψmax, Q1, R, T; base=base))
    end
    vals
end

function pair_relative_entropy_values(ψa::Vector{ComplexF64},
                                      ψb::Vector{ComplexF64},
                                      N::Int,
                                      ell_values::AbstractVector{<:Integer})
    vals = Float64[]
    for ell in ell_values
        ρa = reduced_density_matrix_from_vector(ψa, ell, N)
        ρb = reduced_density_matrix_from_vector(ψb, ell, N)
        push!(vals, relative_entropy(ρa, ρb))
    end
    vals
end

function entanglement_rows_for_states(ψ0::Vector{ComplexF64},
                                      ψ1::Vector{ComplexF64},
                                      ψ2::Vector{ComplexF64},
                                      N::Int;
                                      ell_values::AbstractVector{<:Integer}=2:6,
                                      base::Real=2)
    ell_values = collect(Int, ell_values)
    all(1 .<= ell_values .<= N) || error("All subsystem sizes must be between 1 and N.")

    I01 = pair_mutual_information_values(ψ0, ψ1, N, ell_values; base=base)
    I02 = pair_mutual_information_values(ψ0, ψ2, N, ell_values; base=base)
    I12 = pair_mutual_information_values(ψ1, ψ2, N, ell_values; base=base)

    S10 = pair_relative_entropy_values(ψ1, ψ0, N, ell_values)
    S20 = pair_relative_entropy_values(ψ2, ψ0, N, ell_values)
    S12 = pair_relative_entropy_values(ψ1, ψ2, N, ell_values)

    rows = NamedTuple[]
    for (idx, ell) in enumerate(ell_values)
        push!(rows, (
            N = N,
            ell = ell,
            x = ell / N,
            I01 = I01[idx],
            I02 = I02[idx],
            I12 = I12[idx],
            S10 = S10[idx],
            S20 = S20[idx],
            S12 = S12[idx],
        ))
    end
    rows
end

const PuMPSCodeState = Union{PM.puMPState, PM.puMPSTvec}

function _positive_state_norm(state::PuMPSCodeState)
    nrm = ComplexF64(norm(state))
    tol = 1e-10 * max(1.0, abs(real(nrm)))
    abs(imag(nrm)) <= tol ||
        error("State norm has a significant imaginary part: $nrm")
    real_nrm = real(nrm)
    if real_nrm < 0 && abs(real_nrm) <= tol
        real_nrm = 0.0
    end
    real_nrm > 0 || error("State has zero norm.")
    real_nrm
end

function _state_tensors_for_direct_rdm(state::PM.puMPState)
    ComplexF64.(PM.mps_tensor(state)), nothing
end

function _state_tensors_for_direct_rdm(state::PM.puMPSTvec)
    A, B = PM.mps_tensors(state)
    ComplexF64.(A), ComplexF64.(B)
end

function _direct_rdm_state_data(state::PuMPSCodeState, ell::Int, d::Int)
    N = PM.num_sites(state)
    A, B = _state_tensors_for_direct_rdm(state)
    D = size(A, 1)
    size(A, 3) == D || error("MPS tensor must have equal virtual dimensions.")
    size(A, 2) == d || error("Physical dimension mismatch: got $(size(A, 2)), expected $d.")
    if B !== nothing
        size(B) == size(A) || error("Tangent tensor dimensions do not match the MPS tensor.")
    end

    terms = NamedTuple{(:coef, :block_pos, :env_pos),
                       Tuple{ComplexF64, Int, Int}}[]
    if B === nothing
        push!(terms, (coef = 1.0 + 0.0im, block_pos = 0, env_pos = 0))
    else
        p = PM.momentum(state)
        for n in 1:N
            coef = ComplexF64(cis(p * n))
            if n <= ell
                push!(terms, (coef = coef, block_pos = n, env_pos = 0))
            else
                push!(terms, (coef = coef, block_pos = 0, env_pos = n - ell))
            end
        end
    end

    (N = N, D = D, d = d, A = A, B = B, terms = terms)
end

function site_transfer(X::Array{ComplexF64,3}, Y::Array{ComplexF64,3})
    size(X) == size(Y) || error("Transfer tensors must have identical dimensions.")
    D = size(X, 1)
    d = size(X, 2)
    D2 = D * D
    E = zeros(ComplexF64, D2, D2)
    for s in 1:d
        E .+= kron(Matrix(view(X, :, s, :)), conj(Matrix(view(Y, :, s, :))))
    end
    E
end

function trace_kernel_from_transfer(Tenv::AbstractMatrix, D::Int)
    size(Tenv) == (D * D, D * D) ||
        error("Environment transfer matrix has incompatible size.")
    T4 = reshape(Tenv, D, D, D, D)
    # This is the column-major reshuffle satisfying
    # tr(kron(P, conj(Q)) * Tenv) = vec(P)^T * W * conj(vec(Q)).
    W4 = permutedims(T4, (4, 2, 3, 1))
    reshape(W4, D * D, D * D)
end

function block_feature_matrix(A::Array{ComplexF64,3},
                              B::Union{Nothing,Array{ComplexF64,3}},
                              block_pos::Int,
                              ell::Int,
                              d::Int)
    D = size(A, 1)
    block_pos in 0:ell || error("Block insertion position must be in 0:ell.")
    block_pos == 0 || B !== nothing ||
        error("A tangent tensor is required for a nonzero block insertion position.")

    left_dim = d^ell
    K = zeros(ComplexF64, left_dim, D * D)
    for config in 1:left_dim
        digits = index_to_digits(config, ell, d)
        P = Matrix{ComplexF64}(I, D, D)
        for j in 1:ell
            X = block_pos == j ? B : A
            P = P * Matrix(view(X, :, digits[j], :))
        end
        K[config, :] .= vec(P)
    end
    K
end

function _background_transfer_powers(EAA::Matrix{ComplexF64}, m::Int)
    D2 = size(EAA, 1)
    powers = Vector{Matrix{ComplexF64}}(undef, m + 1)
    powers[1] = Matrix{ComplexF64}(I, D2, D2)
    for q in 1:m
        powers[q + 1] = powers[q] * EAA
    end
    powers
end

function _environment_transfer_data(ket_data, bra_data, m::Int)
    EAA = site_transfer(ket_data.A, bra_data.A)
    EBA = ket_data.B === nothing ? nothing : site_transfer(ket_data.B, bra_data.A)
    EAB = bra_data.B === nothing ? nothing : site_transfer(ket_data.A, bra_data.B)
    EBB = (ket_data.B === nothing || bra_data.B === nothing) ?
          nothing : site_transfer(ket_data.B, bra_data.B)
    (m = m, powers = _background_transfer_powers(EAA, m),
     EBA = EBA, EAB = EAB, EBB = EBB)
end

function _environment_transfer(env_pos_ket::Int, env_pos_bra::Int, data)
    m = data.m
    0 <= env_pos_ket <= m || error("Ket environment insertion is outside the complement.")
    0 <= env_pos_bra <= m || error("Bra environment insertion is outside the complement.")
    powers = data.powers

    if env_pos_ket == 0 && env_pos_bra == 0
        return powers[m + 1]
    elseif env_pos_ket > 0 && env_pos_bra == 0
        data.EBA === nothing && error("Missing ket tangent transfer.")
        q = env_pos_ket
        return powers[q] * data.EBA * powers[m - q + 1]
    elseif env_pos_ket == 0 && env_pos_bra > 0
        data.EAB === nothing && error("Missing bra tangent transfer.")
        r = env_pos_bra
        return powers[r] * data.EAB * powers[m - r + 1]
    end

    data.EBB === nothing && error("Missing two-tangent transfer.")
    q = env_pos_ket
    r = env_pos_bra
    if q == r
        return powers[q] * data.EBB * powers[m - q + 1]
    elseif q < r
        return powers[q] * data.EBA * powers[r - q] * data.EAB * powers[m - r + 1]
    else
        return powers[r] * data.EAB * powers[q - r] * data.EBA * powers[m - q + 1]
    end
end

"""
    mixed_reduced_density_matrix_first_block(ket, bra, ell; d=2, normalize_states=true)

Return `Tr_complement |ket><bra|` on the first `ell` sites, contracting directly
from the puMPS tensor `A` and tangent insertion tensor `B`. For tangent states the
sum over insertion positions uses the same 1-based phase `cis(p*n)` as
`Vector(::PM.puMPSTvec)`, but this routine never materializes the dense vector.
"""
function mixed_reduced_density_matrix_first_block(
    ket::PuMPSCodeState,
    bra::PuMPSCodeState,
    ell::Int;
    d::Int = 2,
    normalize_states::Bool = true,
)::Matrix{ComplexF64}
    ell >= 1 || error("Block size ell must be at least 1.")
    ket_data = _direct_rdm_state_data(ket, ell, d)
    bra_data = _direct_rdm_state_data(bra, ell, d)
    N = ket_data.N
    N == bra_data.N || error("Ket and bra must have the same number of sites.")
    ell <= N || error("Block size ell cannot exceed system size.")
    ket_data.D == bra_data.D || error("Ket and bra bond dimensions must match.")
    ket_data.d == bra_data.d || error("Ket and bra physical dimensions must match.")

    D = ket_data.D
    left_dim = d^ell
    env_data = _environment_transfer_data(ket_data, bra_data, N - ell)
    W_cache = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    W_sum = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()

    for ta in ket_data.terms, tb in bra_data.terms
        env_key = (ta.env_pos, tb.env_pos)
        W = get!(W_cache, env_key) do
            Tenv = _environment_transfer(env_key[1], env_key[2], env_data)
            trace_kernel_from_transfer(Tenv, D)
        end
        block_key = (ta.block_pos, tb.block_pos)
        coeff = ta.coef * conj(tb.coef)
        if haskey(W_sum, block_key)
            W_sum[block_key] .+= coeff .* W
        else
            W_sum[block_key] = coeff .* W
        end
    end

    ket_blocks = Dict{Int, Matrix{ComplexF64}}()
    bra_blocks = Dict{Int, Matrix{ComplexF64}}()
    rho = zeros(ComplexF64, left_dim, left_dim)
    for ((block_pos_ket, block_pos_bra), Wacc) in W_sum
        Kket = get!(ket_blocks, block_pos_ket) do
            block_feature_matrix(ket_data.A, ket_data.B, block_pos_ket, ell, d)
        end
        Kbra = get!(bra_blocks, block_pos_bra) do
            block_feature_matrix(bra_data.A, bra_data.B, block_pos_bra, ell, d)
        end
        rho .+= Kket * Wacc * Kbra'
    end

    if normalize_states
        rho ./= _positive_state_norm(ket) * _positive_state_norm(bra)
    end
    ket === bra ? (rho + rho') ./ 2 : rho
end

function reduced_density_matrix_first_block_direct(state::PuMPSCodeState,
                                                   ell::Int;
                                                   d::Int = 2)
    mixed_reduced_density_matrix_first_block(state, state, ell; d=d)
end

function pair_mutual_information_values_direct(state_a::PuMPSCodeState,
                                               state_b::PuMPSCodeState,
                                               ell_values::AbstractVector{<:Integer};
                                               base::Real = 2,
                                               d::Int = 2)
    vals = Float64[]
    for ell in ell_values
        rho_aa = mixed_reduced_density_matrix_first_block(state_a, state_a, Int(ell); d=d)
        rho_ab = mixed_reduced_density_matrix_first_block(state_a, state_b, Int(ell); d=d)
        rho_ba = mixed_reduced_density_matrix_first_block(state_b, state_a, Int(ell); d=d)
        rho_bb = mixed_reduced_density_matrix_first_block(state_b, state_b, Int(ell); d=d)

        c = conj(tr(rho_ab))
        den2 = real(1 - abs2(c))
        if den2 < 0 && abs(den2) < 1e-10
            den2 = 0.0
        end
        den2 > 0 || error("The two code states are nearly linearly dependent.")
        den = sqrt(den2)

        rho_a_tildeb = (rho_ab - conj(c) .* rho_aa) ./ den
        rho_tildeb_a = (rho_ba - c .* rho_aa) ./ den
        rho_tildeb_tildeb = (
            rho_bb
            - c .* rho_ab
            - conj(c) .* rho_ba
            + abs2(c) .* rho_aa
        ) ./ den2

        rho_Q1R = 0.5 .* [rho_aa rho_a_tildeb; rho_tildeb_a rho_tildeb_tildeb]
        rho_Q1 = 0.5 .* (rho_aa + rho_tildeb_tildeb)
        rho_R = 0.5 .* ComplexF64[
            tr(rho_aa) tr(rho_a_tildeb);
            tr(rho_tildeb_a) tr(rho_tildeb_tildeb)
        ]

        rho_Q1R = (rho_Q1R + rho_Q1R') ./ 2
        rho_Q1 = (rho_Q1 + rho_Q1') ./ 2
        rho_R = (rho_R + rho_R') ./ 2
        push!(vals, entropy_from_rho(rho_Q1; base=base) +
                    entropy_from_rho(rho_R; base=base) -
                    entropy_from_rho(rho_Q1R; base=base))
    end
    vals
end

function pair_relative_entropy_values_direct(state_a::PuMPSCodeState,
                                             state_b::PuMPSCodeState,
                                             ell_values::AbstractVector{<:Integer};
                                             d::Int = 2)
    vals = Float64[]
    for ell in ell_values
        rho_a = reduced_density_matrix_first_block_direct(state_a, Int(ell); d=d)
        rho_b = reduced_density_matrix_first_block_direct(state_b, Int(ell); d=d)
        push!(vals, relative_entropy(rho_a, rho_b))
    end
    vals
end

function entanglement_rows_for_pumps_states_direct(state_I::PuMPSCodeState,
                                                   state_sigma::PuMPSCodeState,
                                                   state_epsilon::PuMPSCodeState,
                                                   N::Int;
                                                   ell_values::AbstractVector{<:Integer}=2:6,
                                                   base::Real=2,
                                                   d::Int=2)
    ell_values = collect(Int, ell_values)
    all(1 .<= ell_values .<= N) || error("All subsystem sizes must be between 1 and N.")

    I01 = pair_mutual_information_values_direct(state_I, state_sigma, ell_values; base=base, d=d)
    I02 = pair_mutual_information_values_direct(state_I, state_epsilon, ell_values; base=base, d=d)
    I12 = pair_mutual_information_values_direct(state_sigma, state_epsilon, ell_values; base=base, d=d)

    S10 = pair_relative_entropy_values_direct(state_sigma, state_I, ell_values; d=d)
    S20 = pair_relative_entropy_values_direct(state_epsilon, state_I, ell_values; d=d)
    S12 = pair_relative_entropy_values_direct(state_sigma, state_epsilon, ell_values; d=d)

    rows = NamedTuple[]
    for (idx, ell) in enumerate(ell_values)
        push!(rows, (
            N = N,
            ell = ell,
            x = ell / N,
            I01 = I01[idx],
            I02 = I02[idx],
            I12 = I12[idx],
            S10 = S10[idx],
            S20 = S20[idx],
            S12 = S12[idx],
        ))
    end
    rows
end

function save_entanglement_scan_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    open(path, "w") do io
        println(io, "N,D,ell,x,E0,E1,E2,gap_ratio,P0,P1,P2,I01,I02,I12,S10,S20,S12")
        for row in rows
            println(io, @sprintf("%d,%d,%d,%.16g,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e",
                                 row.N, row.D, row.ell, row.x,
                                 row.E0, row.E1, row.E2, row.gap_ratio,
                                 row.P0, row.P1, row.P2,
                                 row.I01, row.I02, row.I12,
                                 row.S10, row.S20, row.S12))
        end
    end
end

function entanglement_quality_fields(rel_data, N::Int, D::Int)
    idxs = rel_data.indices
    E0 = rel_data.energies[idxs[1]]
    E1 = rel_data.energies[idxs[2]]
    E2 = rel_data.energies[idxs[3]]
    gap = E1 - E0
    ratio = (E2 - E0) / gap
    (
        D = D,
        E0 = E0,
        E1 = E1,
        E2 = E2,
        gap_ratio = ratio,
        P0 = rel_data.parities[1],
        P1 = rel_data.parities[2],
        P2 = rel_data.parities[3],
    )
end

function run_entanglement_scan(; Ns::AbstractVector{<:Integer}=collect(12:2:24),
                               D::Int=ISING_D,
                               D_schedule::Union{Nothing,AbstractDict}=nothing,
                               ell_values::AbstractVector{<:Integer}=2:6,
                               outfile::AbstractString=joinpath(
                                   @__DIR__,
                                   D_schedule === nothing ?
                                   "ising_entanglement_scan_D$(D).csv" :
                                   "ising_entanglement_scan_variableD.csv",
                               ),
                               seed::Union{Nothing,Integer}=1234,
                               verbose::Bool=true,
                               method::Symbol=:direct)
    method in (:direct, :dense) ||
        error("Unknown entanglement scan method $method. Use :direct or :dense.")
    rows = NamedTuple[]
    for N in Ns
        D_N = D_schedule === nothing ? D : Int(D_schedule[Int(N)])
        verbose && println("=== entanglement scan: N=$N, D=$D_N ===")
        seed === nothing || Random.seed!(seed + N)
        rel_data = compute_lowest_states(N; D=D_N, verbose=verbose)
        idxs = rel_data.indices

        if method == :dense
            ψ0 = normalize_state_vector(Vector(rel_data.state))
            ψ1 = normalize_state_vector(Vector(rel_data.exs[idxs[2]]))
            ψ2 = normalize_state_vector(Vector(rel_data.exs[idxs[3]]))
            new_rows = entanglement_rows_for_states(ψ0, ψ1, ψ2, Int(N);
                                                    ell_values=ell_values)
        else
            state_I = rel_data.state
            state_sigma = rel_data.exs[idxs[2]]
            state_epsilon = rel_data.exs[idxs[3]]
            # Direct production path: do not call Vector(::PM.puMPSTvec) here.
            new_rows = entanglement_rows_for_pumps_states_direct(
                state_I, state_sigma, state_epsilon, Int(N);
                ell_values=ell_values,
            )
        end

        quality = entanglement_quality_fields(rel_data, Int(N), D_N)
        append!(rows, [merge((N = row.N,), quality, Base.structdiff(row, NamedTuple{(:N,)}))
                       for row in new_rows])
        save_entanglement_scan_csv(outfile, rows)
        verbose && println("saved partial results to $outfile")
    end
    rows
end

function run_entanglement_scan_variableD(;
                                         schedule::AbstractDict=CONSERVATIVE_ISING_D_SCHEDULE,
                                         Ns::AbstractVector{<:Integer}=sort(collect(keys(schedule))),
                                         ell_values::AbstractVector{<:Integer}=2:6,
                                         outfile::AbstractString=joinpath(@__DIR__, "ising_entanglement_scan_variableD.csv"),
                                         seed::Union{Nothing,Integer}=1234,
                                         verbose::Bool=true,
                                         method::Symbol=:direct)
    run_entanglement_scan(; Ns=Ns, D_schedule=schedule, ell_values=ell_values,
                          outfile=outfile, seed=seed, verbose=verbose,
                          method=method)
end

function ising_bit(idx0::Int, site::Int)
    (idx0 >> (site - 1)) & 1
end

function ising_z_eigenvalue(idx0::Int, site::Int)
    iszero(ising_bit(idx0, site)) ? 1.0 : -1.0
end

function flip_ising_site(idx0::Int, site::Int)
    idx0 ⊻ (1 << (site - 1))
end

function dense_ising_hamiltonian(N::Int; apbc::Bool=false)
    N >= 2 || error("N must be at least 2.")
    dim = 1 << N
    H = zeros(Float64, dim, dim)
    for idx0 in 0:dim-1
        col = idx0 + 1
        zsum = 0.0
        for j in 1:N
            zsum += ising_z_eigenvalue(idx0, j)
        end
        H[col, col] -= zsum
        for j in 1:N
            jp1 = j == N ? 1 : j + 1
            coeff = (apbc && j == N) ? 1.0 : -1.0
            flipped = flip_ising_site(flip_ising_site(idx0, j), jp1)
            H[flipped + 1, col] += coeff
        end
    end
    Hermitian(H)
end

function dense_parity_diagonal(N::Int)
    dim = 1 << N
    [isodd(count_ones(idx0)) ? -1.0 : 1.0 for idx0 in 0:dim-1]
end

function translate_index(idx0::Int, N::Int)
    shifted = 0
    for j in 1:N
        bit = ising_bit(idx0, j)
        jnew = j == N ? 1 : j + 1
        shifted |= bit << (jnew - 1)
    end
    shifted
end

function dense_translation(N::Int; twisted::Bool=false)
    dim = 1 << N
    T = zeros(ComplexF64, dim, dim)
    for idx0 in 0:dim-1
        shifted = translate_index(idx0, N)
        phase = twisted ? ising_z_eigenvalue(shifted, 1) : 1.0
        T[shifted + 1, idx0 + 1] = phase
    end
    T
end

function canonical_k_from_phase(phase::Number, N::Int)
    k = angle(phase) * N / (2π)
    mod(k + N / 2, N) - N / 2
end

function parity_expectation(ψ::AbstractVector{<:Complex}, N::Int)
    pdiag = dense_parity_diagonal(N)
    real(sum(abs2(ψ[i]) * pdiag[i] for i in eachindex(ψ)))
end

function joint_ising_spectrum(N::Int; apbc::Bool=false, energy_tol::Real=1e-8)
    H = dense_ising_hamiltonian(N; apbc=apbc)
    U = dense_translation(N; twisted=apbc)
    evals, evecs = eigen(H)

    entries = NamedTuple[]
    start = 1
    while start <= length(evals)
        stop = start
        while stop < length(evals) && abs(evals[stop + 1] - evals[start]) < energy_tol
            stop += 1
        end
        inds = start:stop
        V = Matrix{ComplexF64}(evecs[:, inds])
        Usub = V' * U * V
        uvals, uvecs = eigen(Usub)
        for a in eachindex(uvals)
            ψ = V * uvecs[:, a]
            ψ ./= norm(ψ)
            push!(entries, (
                energy = real(mean(evals[inds])),
                k = canonical_k_from_phase(uvals[a], N),
                parity = parity_expectation(ψ, N),
                vector = Vector{ComplexF64}(ψ),
                apbc = apbc,
            ))
        end
        start = stop + 1
    end
    sort(entries, by=x -> (x.energy, x.k))
end

function find_first_state(entries; parity::Union{Nothing,Int}=nothing,
                          k_target::Union{Nothing,Float64}=nothing,
                          energy_min::Real=-Inf,
                          k_tol::Real=1e-6,
                          parity_tol::Real=1e-6)
    for entry in entries
        entry.energy > energy_min || continue
        if parity !== nothing
            abs(entry.parity - parity) < parity_tol || continue
        end
        if k_target !== nothing
            abs(entry.k - k_target) < k_tol || continue
        end
        return entry
    end
    error("No state matched the requested filters.")
end

"""
    identify_ising_primaries_ed(N)

Small-system exact-diagonalization reference for the Ising primary states
needed in the current analysis. It identifies:

- I: PBC, even parity, k=0 ground state, Δ=0
- σ: PBC, odd parity, k=0, Δ=1/8
- ψ, ψbar: APBC, odd parity, k=±1/2, Δ=1/2

Scaling dimensions are reported in ε-gap units, i.e. using the PBC ε
primary as Δε=1 to remove the nonuniversal velocity.
"""
function identify_ising_primaries_ed(N::Int)
    pbc = joint_ising_spectrum(N; apbc=false)
    apbc = joint_ising_spectrum(N; apbc=true)

    I_state = find_first_state(pbc; parity=1, k_target=0.0)
    sigma_state = find_first_state(pbc; parity=-1, k_target=0.0,
                                   energy_min=I_state.energy + 1e-8)
    epsilon_state = find_first_state(pbc; parity=1, k_target=0.0,
                                     energy_min=I_state.energy + 1e-8)
    psi_state = find_first_state(apbc; parity=-1, k_target=0.5)
    psibar_state = find_first_state(apbc; parity=-1, k_target=-0.5)

    eps_gap = epsilon_state.energy - I_state.energy
    annotate(label, state, target_delta) = (
        label = label,
        energy = state.energy,
        k = state.k,
        parity = state.parity,
        delta = (state.energy - I_state.energy) / eps_gap,
        target_delta = target_delta,
        vector = state.vector,
        apbc = state.apbc,
    )

    (
        N = N,
        I = annotate(:I, I_state, 0.0),
        sigma = annotate(:sigma, sigma_state, 1 / 8),
        epsilon = annotate(:epsilon, epsilon_state, 1.0),
        psi = annotate(:psi, psi_state, 1 / 2),
        psibar = annotate(:psibar, psibar_state, 1 / 2),
    )
end

function save_primary_identification(path::AbstractString, entries)
    open(path, "w") do io
        println(io, "Ising primary identification from exact diagonalization")
        println(io, @sprintf("%6s %10s %7s %12s %12s %12s",
                             "N", "state", "BC", "k", "parity", "Δ (ε units)"))
        println(io, "-"^72)
        for entry in entries
            for state in (entry.I, entry.sigma, entry.epsilon, entry.psi, entry.psibar)
                bc = state.apbc ? "APBC" : "PBC"
                println(io, @sprintf("%6d %10s %7s %12.6f %12.6f %12.6f",
                                     entry.N, String(state.label), bc,
                                     state.k, state.parity, state.delta))
            end
        end
        println(io, "\nExpected primary data:")
        println(io, "  I:      PBC,  k=0,    parity=+1, Δ=0")
        println(io, "  sigma:  PBC,  k=0,    parity=-1, Δ=1/8")
        println(io, "  epsilon:PBC,  k=0,    parity=+1, Δ=1")
        println(io, "  psi:    APBC, k=+1/2, parity=-1, Δ=1/2")
        println(io, "  psibar: APBC, k=-1/2, parity=-1, Δ=1/2")
    end
end

function run_primary_identification(; Ns::Vector{Int}=[8, 10, 12],
                                    outfile::AbstractString=joinpath(@__DIR__, "ising_primary_identification.txt"))
    entries = [identify_ising_primaries_ed(N) for N in Ns]
    save_primary_identification(outfile, entries)
    entries
end

function run_mutual_information_analysis(; Ns::Vector{Int}=[12, 14],
                                         D::Int=ISING_D,
                                         outfile::AbstractString=joinpath(@__DIR__, "ising_I_Q1R_results.txt"))
    mi_entries = NamedTuple[]
    for N in Ns
        rel_data = compute_lowest_states(N; D=D, verbose=false)
        idxs = rel_data.indices

        ψ0 = Vector(rel_data.state)
        ψ1 = Vector(rel_data.exs[idxs[2]])
        ψ2 = Vector(rel_data.exs[idxs[3]])

        ℓ_max = min(6, N)
        info01 = analyze_I_Q1R_pair(ψ0, ψ1, N; ℓ_max=ℓ_max)
        info02 = analyze_I_Q1R_pair(ψ0, ψ2, N; ℓ_max=ℓ_max)
        info12 = analyze_I_Q1R_pair(ψ1, ψ2, N; ℓ_max=ℓ_max)

        push!(mi_entries, (
            N = N,
            pair01 = info01,
            pair02 = info02,
            pair12 = info12,
        ))
    end
    save_I_Q1R_results(outfile, mi_entries)
    mi_entries
end

############################################################
# Main loop over system sizes: pairs (0,1), (0,2), (1,2)
############################################################
if abspath(PROGRAM_FILE) == @__FILE__
    run_mutual_information_analysis()
end
