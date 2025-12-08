# Load the puMPS code from the src/ folder (one directory up from examples/)
include(joinpath(@__DIR__, "..", "src", "puMPS.jl"))
import .puMPS

using Printf
using LinearAlgebra
using PyPlot
using Statistics

const PM = puMPS

# Basic simulation parameters
const ISING_D =8
const ISING_MODEL = (hz = 1.0, hx = 0.0)

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

    ks = [ -1, 0, 1]
    num_states = [3, 6, 3]
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

Ns = [16,18,20]
ratio_info = ratio_scaling_analysis(Ns)
results_path = joinpath(@__DIR__, "ising_ratio_results.txt")
save_ratio_results(results_path, ratio_info)
save_conformal_tower_plot(ratio_info.raw, @__DIR__)

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

rel_entries = NamedTuple[]
for N in Ns
    rel_data = compute_lowest_states(N; D=ISING_D, verbose=false)
    idxs = rel_data.indices
    M0 = rel_data.state
    M1 = rel_data.exs[idxs[2]]
    M2 = rel_data.exs[idxs[3]]
    ℓ_max = min(6, N)
    rel_info = analyze_relative_entropy_Q1(M0, M1, M2; ℓ_max=ℓ_max, N=N,
                                           outfile=joinpath(@__DIR__, "Srel_Q1_vs_x_N$(N).png"))
    push!(rel_entries, (N=N, info=rel_info))
end
rel_results_path = joinpath(@__DIR__, "ising_relative_entropy.txt")
save_relative_entropy_results(rel_results_path, rel_entries)

function check_rdm_consistency(N::Int; ℓ_max::Int=2, D::Int=ISING_D)
    Hloc = ising_local_MPO(ComplexF64)
    M = PM.rand_puMPState(ComplexF64, 2, D, N)
    optimize_state!(M, Hloc)

    # brute-force vector (小 N 的时候用来验证)
    ψ = Vector(PM.puMPSTvec(M))  # 如果库里有类似的接口，用它拿到 full state

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
