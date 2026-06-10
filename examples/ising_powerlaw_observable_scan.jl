include(joinpath(@__DIR__, "ising_example.jl"))

using Printf
using LinearAlgebra
using Random

const POWERLAW_L_VALUES = [16, 18, 20, 22, 24]
const POWERLAW_ELL_VALUES = 2:8
const POWERLAW_BOND_DIM_SCHEDULE = Dict(
    16 => 14,
    18 => 16,
    20 => 16,
    22 => 18,
    24 => 18,
)

const POWERLAW_QUANTITIES = (
    small_avg_relent_X = "small_avg_relent_X",
    complement_proxy = "complement_proxy",
    mutual_information_XR = "mutual_information_XR",
)

const POWERLAW_RAW_FILENAMES = Dict(
    POWERLAW_QUANTITIES.small_avg_relent_X =>
        "ising_powerlaw_small_avg_relent_X.csv",
    POWERLAW_QUANTITIES.complement_proxy =>
        "ising_powerlaw_complement_proxy.csv",
    POWERLAW_QUANTITIES.mutual_information_XR =>
        "ising_powerlaw_mutual_information_XR.csv",
)

const POWERLAW_FIT_FILENAME = "ising_powerlaw_fit_summary.csv"
const POWERLAW_METHOD = "direct_puMPS_rdm"
const POWERLAW_D_CODE = 2

const POWERLAW_RAW_COLUMNS = [
    "L", "bond_dim", "ell", "x", "code_label", "state_0", "state_1",
    "D_code", "quantity", "value_bits",
    "E0", "E1", "E2", "gap_ratio", "P0", "P1", "P2",
    "overlap_re", "overlap_im", "overlap_abs",
    "entropy_base", "method", "warning",
]

const POWERLAW_FIT_COLUMNS = [
    "quantity", "code_label", "prefactor_c", "exponent", "intercept_log2",
    "n_points", "x_min", "x_max", "r2", "fit_rule",
]

function hermitian_part(rho::AbstractMatrix)
    Matrix{ComplexF64}((rho + rho') ./ 2)
end

function normalized_density_matrix_bits(rho::AbstractMatrix;
                                        trace_tol::Real=1e-9)
    rho_h = hermitian_part(rho)
    tr_rho = tr(rho_h)
    abs(imag(tr_rho)) <= trace_tol ||
        error("Density matrix trace has non-negligible imaginary part: $tr_rho")
    tr_real = real(tr_rho)
    abs(tr_real) > trace_tol || error("Density matrix has near-zero trace.")
    if abs(tr_real - 1.0) > trace_tol
        rho_h ./= tr_real
    end
    rho_h
end

function entropy_bits_from_rho(rho::AbstractMatrix;
                               base::Real=2,
                               psd_tol::Real=1e-9)
    rho_n = normalized_density_matrix_bits(rho)
    vals = eigvals(Hermitian(rho_n))
    lambdas = real.(vals)
    min_lambda = minimum(lambdas)
    min_lambda >= -psd_tol ||
        error("Density matrix is not positive semidefinite: minimum eigenvalue $min_lambda")
    lambdas .= max.(lambdas, 0.0)
    s = sum(lambdas)
    s > 0 || return 0.0
    lambdas ./= s
    nz = lambdas .> 0
    -sum(lambdas[nz] .* (log.(lambdas[nz]) ./ log(base)))
end

function matrix_log_hermitian_base(rho::AbstractMatrix;
                                   base::Real=2,
                                   eps_floor::Real=1e-14,
                                   psd_tol::Real=1e-9)
    rho_h = hermitian_part(rho)
    vals, vecs = eigen(Hermitian(rho_h))
    lambdas = real.(vals)
    min_lambda = minimum(lambdas)
    min_lambda >= -psd_tol ||
        error("Matrix logarithm input is not positive semidefinite: minimum eigenvalue $min_lambda")
    lambdas .= max.(lambdas, eps_floor)
    vecs * Diagonal(log.(lambdas) ./ log(base)) * vecs'
end

function relative_entropy_bits(rho::AbstractMatrix,
                               sigma::AbstractMatrix;
                               base::Real=2,
                               eps_floor::Real=1e-14)
    size(rho) == size(sigma) || error("Density matrices must have the same size.")
    rho_n = normalized_density_matrix_bits(rho)
    sigma_n = normalized_density_matrix_bits(sigma)
    log_rho = matrix_log_hermitian_base(rho_n; base=base, eps_floor=eps_floor)
    log_sigma = matrix_log_hermitian_base(sigma_n; base=base, eps_floor=eps_floor)
    real(tr(rho_n * (log_rho - log_sigma)))
end

function average_relative_entropy_bits(rho0::AbstractMatrix,
                                       rho1::AbstractMatrix;
                                       base::Real=2)
    rho0_n = normalized_density_matrix_bits(rho0)
    rho1_n = normalized_density_matrix_bits(rho1)
    rho_bar = normalized_density_matrix_bits((rho0_n + rho1_n) ./ 2)
    0.5 * (
        relative_entropy_bits(rho0_n, rho_bar; base=base) +
        relative_entropy_bits(rho1_n, rho_bar; base=base)
    )
end

function gram_schmidt_reduced_blocks(rho_aa::AbstractMatrix,
                                     rho_ab::AbstractMatrix,
                                     rho_ba::AbstractMatrix,
                                     rho_bb::AbstractMatrix;
                                     overlap::Union{Nothing,Complex}=nothing,
                                     dep_tol::Real=1e-12)
    c_ab = overlap === nothing ? conj(tr(rho_ab)) : ComplexF64(overlap)
    den2 = real(1 - abs2(c_ab))
    if den2 < 0 && abs(den2) < 1e-10
        den2 = 0.0
    end
    den2 > dep_tol || error("The two code states are nearly linearly dependent.")
    den = sqrt(den2)

    rho00 = normalized_density_matrix_bits(rho_aa)
    rho01 = (rho_ab .- conj(c_ab) .* rho_aa) ./ den
    rho10 = (rho_ba .- c_ab .* rho_aa) ./ den
    rho11 = (
        rho_bb
        .- c_ab .* rho_ab
        .- conj(c_ab) .* rho_ba
        .+ abs2(c_ab) .* rho_aa
    ) ./ den2
    rho11 = normalized_density_matrix_bits(rho11)
    mixed_mismatch = norm(rho10 - rho01')
    if mixed_mismatch > 1e-8
        @warn "Gram-Schmidt mixed reduced blocks are not adjoint within tolerance" mixed_mismatch
    end
    rho01 = (rho01 + rho10') ./ 2
    rho10 = rho01'

    (rho00=rho00, rho01=rho01, rho10=rho10, rho11=rho11,
     overlap=c_ab, den2=den2)
end

function rho_XR_from_code_blocks(blocks)
    rho_XR = 0.5 .* [blocks.rho00 blocks.rho01; blocks.rho10 blocks.rho11]
    hermitian_part(rho_XR)
end

function reference_density_from_code_blocks(blocks)
    rho_R = 0.5 .* ComplexF64[
        tr(blocks.rho00) tr(blocks.rho01);
        tr(blocks.rho10) tr(blocks.rho11)
    ]
    normalized_density_matrix_bits(rho_R)
end

function mutual_information_XR_from_code_blocks(blocks; base::Real=2)
    rho_X = normalized_density_matrix_bits((blocks.rho00 + blocks.rho11) ./ 2)
    rho_R = reference_density_from_code_blocks(blocks)
    rho_XR = normalized_density_matrix_bits(rho_XR_from_code_blocks(blocks))
    entropy_bits_from_rho(rho_X; base=base) +
        entropy_bits_from_rho(rho_R; base=base) -
        entropy_bits_from_rho(rho_XR; base=base)
end

function nonnegative_with_warning(value::Real, label::AbstractString;
                                  tol::Real=1e-10)
    if value >= 0
        return Float64(value), ""
    elseif value >= -tol
        return 0.0, "$(label)_clipped_roundoff_negative"
    else
        @warn "$label is significantly negative" value
        return Float64(value), "$(label)_negative"
    end
end

function powerlaw_observables_from_mixed_blocks(rho_aa::AbstractMatrix,
                                                rho_ab::AbstractMatrix,
                                                rho_ba::AbstractMatrix,
                                                rho_bb::AbstractMatrix;
                                                entropy_base::Real=2,
                                                overlap::Union{Nothing,Complex}=nothing)
    entropy_base == 2 || error("This task requires base-2 observables.")
    blocks = gram_schmidt_reduced_blocks(rho_aa, rho_ab, rho_ba, rho_bb;
                                         overlap=overlap)
    rho_bar = normalized_density_matrix_bits((blocks.rho00 + blocks.rho11) ./ 2)
    small_avg = 0.5 * (
        relative_entropy_bits(blocks.rho00, rho_bar; base=entropy_base) +
        relative_entropy_bits(blocks.rho11, rho_bar; base=entropy_base)
    )
    mutual_info = mutual_information_XR_from_code_blocks(blocks; base=entropy_base)
    complement = mutual_info - small_avg

    small_avg, warn_a = nonnegative_with_warning(small_avg, "small_avg_relent_X")
    mutual_info, warn_delta = nonnegative_with_warning(mutual_info, "mutual_information_XR")
    complement, warn_b = nonnegative_with_warning(complement, "complement_proxy")
    warning = join(filter(!isempty, [warn_a, warn_b, warn_delta]), ";")

    (small_avg_relent_X=small_avg,
     complement_proxy=complement,
     mutual_information_XR=mutual_info,
     overlap=blocks.overlap,
     warning=warning,
     rho_R=reference_density_from_code_blocks(blocks),
     blocks=blocks)
end

function dense_mixed_rdm_first_block(psi_a::AbstractVector,
                                     psi_b::AbstractVector,
                                     ell::Int,
                                     L::Int;
                                     d::Int=2)
    left_dim = d^ell
    right_dim = d^(L - ell)
    length(psi_a) == left_dim * right_dim || error("psi_a has incompatible length.")
    length(psi_b) == left_dim * right_dim || error("psi_b has incompatible length.")
    psi_a_n = Vector{ComplexF64}(psi_a) ./ norm(psi_a)
    psi_b_n = Vector{ComplexF64}(psi_b) ./ norm(psi_b)
    psi_a_mat = reshape(psi_a_n, left_dim, right_dim)
    psi_b_mat = reshape(psi_b_n, left_dim, right_dim)
    psi_a_mat * psi_b_mat'
end

function dense_mixed_rdm_subsystem(psi_a::AbstractVector,
                                   psi_b::AbstractVector,
                                   subset::Vector{Int},
                                   L::Int;
                                   d::Int=2)
    psi_a_n = Vector{ComplexF64}(psi_a) ./ norm(psi_a)
    psi_b_n = Vector{ComplexF64}(psi_b) ./ norm(psi_b)
    shape = ntuple(_ -> d, L)
    tensor_a = reshape(psi_a_n, shape)
    tensor_b = reshape(psi_b_n, shape)
    complement = setdiff(collect(1:L), subset)
    perm = vcat(subset, complement)
    mat_a = reshape(permutedims(tensor_a, perm), d^length(subset), :)
    mat_b = reshape(permutedims(tensor_b, perm), d^length(subset), :)
    mat_a * mat_b'
end

function dense_max_entangled_state(psi0::AbstractVector,
                                   psi1::AbstractVector)
    psi0_n = Vector{ComplexF64}(psi0) ./ norm(psi0)
    psi1_n = Vector{ComplexF64}(psi1) ./ norm(psi1)
    ket0 = ComplexF64[1, 0]
    ket1 = ComplexF64[0, 1]
    (kron(ket0, psi0_n) + kron(ket1, psi1_n)) ./ sqrt(2)
end

function csv_escape(x)
    s = string(x)
    occursin(r"[,\n\"]", s) || return s
    "\"" * replace(s, "\"" => "\"\"") * "\""
end

function write_csv(path::AbstractString, columns::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            vals = [getproperty(row, Symbol(col)) for col in columns]
            println(io, join(csv_escape.(vals), ","))
        end
    end
end

function read_simple_csv(path::AbstractString)
    isfile(path) || return NamedTuple[]
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]
    cols = split(lines[1], ",")
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ",")
        push!(rows, NamedTuple{Tuple(Symbol.(cols))}(Tuple(vals)))
    end
    rows
end

function raw_csv_path(output_dir::AbstractString, quantity::AbstractString)
    joinpath(output_dir, POWERLAW_RAW_FILENAMES[quantity])
end

function write_powerlaw_raw_csvs(output_dir::AbstractString, rows)
    mkpath(output_dir)
    for quantity in values(POWERLAW_QUANTITIES)
        qrows = [row for row in rows if row.quantity == quantity]
        sort!(qrows, by = r -> (r.L, r.ell, r.code_label))
        write_csv(raw_csv_path(output_dir, quantity), POWERLAW_RAW_COLUMNS, qrows)
    end
end

function write_powerlaw_fit_csv(output_dir::AbstractString, fit_rows)
    mkpath(output_dir)
    write_csv(joinpath(output_dir, POWERLAW_FIT_FILENAME),
              POWERLAW_FIT_COLUMNS,
              sort(collect(fit_rows), by = r -> (r.quantity, r.code_label)))
end

function existing_powerlaw_rows(output_dir::AbstractString)
    rows = NamedTuple[]
    for quantity in values(POWERLAW_QUANTITIES)
        path = raw_csv_path(output_dir, quantity)
        append!(rows, read_simple_csv(path))
    end
    rows
end

function completed_lengths(rows, ell_values)
    needed_ell = Set(Int.(ell_values))
    by_L_quantity_code = Dict{Tuple{Int,String,String}, Set{Int}}()
    for row in rows
        L = parse(Int, row.L)
        key = (L, row.quantity, row.code_label)
        get!(by_L_quantity_code, key, Set{Int}())
        push!(by_L_quantity_code[key], parse(Int, row.ell))
    end
    isempty(by_L_quantity_code) && return Set{Int}()
    complete = Set{Int}()
    for L in unique(key[1] for key in keys(by_L_quantity_code))
        ok = true
        for quantity in values(POWERLAW_QUANTITIES), code in ("I_sigma", "I_epsilon")
            ok &= get(by_L_quantity_code, (L, quantity, code), Set{Int}()) == needed_ell
        end
        ok && push!(complete, L)
    end
    complete
end

function parse_raw_rows_for_fits(output_dir::AbstractString)
    rows = NamedTuple[]
    for quantity in values(POWERLAW_QUANTITIES)
        for row in read_simple_csv(raw_csv_path(output_dir, quantity))
            push!(rows, (
                quantity = row.quantity,
                code_label = row.code_label,
                x = parse(Float64, row.x),
                value_bits = parse(Float64, row.value_bits),
            ))
        end
    end
    rows
end

function fit_power_law(rows;
                       quantity::AbstractString,
                       code_label::AbstractString,
                       fit_tol::Real=1e-14,
                       x_max_inclusive::Real=0.25)
    selected = [
        row for row in rows
        if row.quantity == quantity &&
           row.code_label == code_label &&
           row.x <= x_max_inclusive &&
           row.value_bits > fit_tol
    ]
    length(selected) >= 2 ||
        error("Not enough points to fit $quantity for $code_label.")
    xs = [row.x for row in selected]
    ys = [row.value_bits for row in selected]
    lx = log2.(xs)
    ly = log2.(ys)
    A = hcat(ones(length(lx)), lx)
    coeff = A \ ly
    intercept = coeff[1]
    exponent = coeff[2]
    pred = A * coeff
    ss_res = sum(abs2, ly .- pred)
    ss_tot = sum(abs2, ly .- mean(ly))
    r2 = ss_tot == 0 ? 1.0 : 1.0 - ss_res / ss_tot
    (
        quantity = quantity,
        code_label = code_label,
        prefactor_c = 2.0^intercept,
        exponent = exponent,
        intercept_log2 = intercept,
        n_points = length(selected),
        x_min = minimum(xs),
        x_max = maximum(xs),
        r2 = r2,
        fit_rule = "x <= $(x_max_inclusive) and value_bits > $(fit_tol)",
    )
end

function fit_powerlaw_observable_csvs(output_dir::AbstractString;
                                      fit_tol::Real=1e-14,
                                      x_max_inclusive::Real=0.25)
    raw_rows = parse_raw_rows_for_fits(output_dir)
    fit_rows = NamedTuple[]
    for quantity in values(POWERLAW_QUANTITIES), code_label in ("I_sigma", "I_epsilon")
        push!(fit_rows, fit_power_law(raw_rows;
                                      quantity=quantity,
                                      code_label=code_label,
                                      fit_tol=fit_tol,
                                      x_max_inclusive=x_max_inclusive))
    end
    write_powerlaw_fit_csv(output_dir, fit_rows)
    fit_rows
end

function make_powerlaw_raw_row(; L::Int, bond_dim::Int, ell::Int,
                               code_label::String,
                               state_0::String,
                               state_1::String,
                               quantity::String,
                               value_bits::Real,
                               quality,
                               overlap::Complex,
                               entropy_base::Real,
                               warning::String="")
    (
        L = L,
        bond_dim = bond_dim,
        ell = ell,
        x = ell / L,
        code_label = code_label,
        state_0 = state_0,
        state_1 = state_1,
        D_code = POWERLAW_D_CODE,
        quantity = quantity,
        value_bits = Float64(value_bits),
        E0 = quality.E0,
        E1 = quality.E1,
        E2 = quality.E2,
        gap_ratio = quality.gap_ratio,
        P0 = quality.P0,
        P1 = quality.P1,
        P2 = quality.P2,
        overlap_re = real(overlap),
        overlap_im = imag(overlap),
        overlap_abs = abs(overlap),
        entropy_base = Float64(entropy_base),
        method = POWERLAW_METHOD,
        warning = warning,
    )
end

function code_state_specs(state_I, state_sigma, state_epsilon)
    (
        (code_label="I_sigma", state_0="I", state_1="sigma",
         a=state_I, b=state_sigma),
        (code_label="I_epsilon", state_0="I", state_1="epsilon",
         a=state_I, b=state_epsilon),
    )
end

function compute_powerlaw_rows_for_states(state_I,
                                          state_sigma,
                                          state_epsilon,
                                          L::Int,
                                          bond_dim::Int,
                                          quality;
                                          ell_values=POWERLAW_ELL_VALUES,
                                          entropy_base::Real=2)
    entropy_base == 2 || error("This task requires entropy_base=2.")
    rows = NamedTuple[]
    rdm_cache = Dict{Tuple{String,String,Int}, Matrix{ComplexF64}}()
    state_lookup = Dict("I" => state_I, "sigma" => state_sigma, "epsilon" => state_epsilon)

    get_block(s0::String, s1::String, ell::Int) = get!(rdm_cache, (s0, s1, ell)) do
        mixed_reduced_density_matrix_first_block(state_lookup[s0], state_lookup[s1], ell)
    end

    for ell in Int.(ell_values)
        for spec in code_state_specs(state_I, state_sigma, state_epsilon)
            rho_aa = get_block(spec.state_0, spec.state_0, ell)
            rho_ab = get_block(spec.state_0, spec.state_1, ell)
            rho_ba = get_block(spec.state_1, spec.state_0, ell)
            rho_bb = get_block(spec.state_1, spec.state_1, ell)

            obs = powerlaw_observables_from_mixed_blocks(
                rho_aa, rho_ab, rho_ba, rho_bb;
                entropy_base=entropy_base,
            )
            push!(rows, make_powerlaw_raw_row(
                L=L, bond_dim=bond_dim, ell=ell,
                code_label=spec.code_label,
                state_0=spec.state_0,
                state_1=spec.state_1,
                quantity=POWERLAW_QUANTITIES.small_avg_relent_X,
                value_bits=obs.small_avg_relent_X,
                quality=quality,
                overlap=obs.overlap,
                entropy_base=entropy_base,
                warning=obs.warning,
            ))
            push!(rows, make_powerlaw_raw_row(
                L=L, bond_dim=bond_dim, ell=ell,
                code_label=spec.code_label,
                state_0=spec.state_0,
                state_1=spec.state_1,
                quantity=POWERLAW_QUANTITIES.complement_proxy,
                value_bits=obs.complement_proxy,
                quality=quality,
                overlap=obs.overlap,
                entropy_base=entropy_base,
                warning=obs.warning,
            ))
            push!(rows, make_powerlaw_raw_row(
                L=L, bond_dim=bond_dim, ell=ell,
                code_label=spec.code_label,
                state_0=spec.state_0,
                state_1=spec.state_1,
                quantity=POWERLAW_QUANTITIES.mutual_information_XR,
                value_bits=obs.mutual_information_XR,
                quality=quality,
                overlap=obs.overlap,
                entropy_base=entropy_base,
                warning=obs.warning,
            ))
            abs(obs.mutual_information_XR - obs.small_avg_relent_X - obs.complement_proxy) < 1e-8 ||
                error("Delta = A + B identity failed for $(spec.code_label), L=$L, ell=$ell")
        end
    end
    rows
end

function run_ising_powerlaw_observable_scan(;
    L_values = POWERLAW_L_VALUES,
    ell_values = POWERLAW_ELL_VALUES,
    bond_dim_schedule = POWERLAW_BOND_DIM_SCHEDULE,
    output_dir = @__DIR__,
    entropy_base = 2,
    force::Bool = false,
    verbose::Bool = true,
    seed::Union{Nothing,Integer} = 1234,
    fit_tol::Real = 1e-14,
    fit_x_max::Real = 0.25,
)
    entropy_base == 2 || error("This task requires entropy_base=2.")
    output_dir = String(output_dir)
    mkpath(output_dir)

    existing_rows = force ? NamedTuple[] : existing_powerlaw_rows(output_dir)
    complete_Ls = force ? Set{Int}() : completed_lengths(existing_rows, ell_values)
    rows = [
        row for row in existing_rows
        if parse(Int, row.L) in complete_Ls
    ]
    # Convert existing string rows to typed rows for rewriting.
    typed_rows = NamedTuple[]
    for row in rows
        push!(typed_rows, (
            L=parse(Int, row.L),
            bond_dim=parse(Int, row.bond_dim),
            ell=parse(Int, row.ell),
            x=parse(Float64, row.x),
            code_label=row.code_label,
            state_0=row.state_0,
            state_1=row.state_1,
            D_code=parse(Int, row.D_code),
            quantity=row.quantity,
            value_bits=parse(Float64, row.value_bits),
            E0=parse(Float64, row.E0),
            E1=parse(Float64, row.E1),
            E2=parse(Float64, row.E2),
            gap_ratio=parse(Float64, row.gap_ratio),
            P0=parse(Float64, row.P0),
            P1=parse(Float64, row.P1),
            P2=parse(Float64, row.P2),
            overlap_re=parse(Float64, row.overlap_re),
            overlap_im=parse(Float64, row.overlap_im),
            overlap_abs=parse(Float64, row.overlap_abs),
            entropy_base=parse(Float64, row.entropy_base),
            method=row.method,
            warning=(:warning in propertynames(row)) ? row.warning : "",
        ))
    end

    for L in Int.(L_values)
        if L in complete_Ls
            verbose && println("=== power-law scan: L=$L already complete, skipping ===")
            continue
        end
        bond_dim = Int(bond_dim_schedule[L])
        verbose && println("=== power-law scan: L=$L, bond_dim=$bond_dim ===")
        seed === nothing || Random.seed!(seed + L)
        rel_data = compute_lowest_states(L; D=bond_dim, verbose=verbose)
        idxs = rel_data.indices
        state_I = rel_data.state
        state_sigma = rel_data.exs[idxs[2]]
        state_epsilon = rel_data.exs[idxs[3]]
        quality = entanglement_quality_fields(rel_data, L, bond_dim)
        verbose && @printf("quality: E0=%.12f E1=%.12f E2=%.12f gap_ratio=%.6f P=(%.6f,%.6f,%.6f)\n",
                           quality.E0, quality.E1, quality.E2, quality.gap_ratio,
                           quality.P0, quality.P1, quality.P2)

        append!(typed_rows, compute_powerlaw_rows_for_states(
            state_I, state_sigma, state_epsilon, L, bond_dim, quality;
            ell_values=ell_values,
            entropy_base=entropy_base,
        ))
        write_powerlaw_raw_csvs(output_dir, typed_rows)
        verbose && println("saved partial power-law observable CSVs to $output_dir")
    end

    fit_rows = fit_powerlaw_observable_csvs(output_dir;
                                            fit_tol=fit_tol,
                                            x_max_inclusive=fit_x_max)
    verbose && println("saved fit summary to ", joinpath(output_dir, POWERLAW_FIT_FILENAME))
    (rows=typed_rows, fits=fit_rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ising_powerlaw_observable_scan()
end
