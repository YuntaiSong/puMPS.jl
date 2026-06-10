using LinearAlgebra
using Random
using Test

module IsingPowerlawObservableTest
include(joinpath(@__DIR__, "..", "examples", "ising_powerlaw_observable_scan.jl"))
end

const PW = IsingPowerlawObservableTest
const PWM = PW.PM

function _orthonormal_dense_pair(dim::Int)
    ψ0 = randn(ComplexF64, dim)
    ψ0 ./= norm(ψ0)
    ψ1 = randn(ComplexF64, dim)
    ψ1 .-= ψ0 .* dot(ψ0, ψ1)
    ψ1 ./= norm(ψ1)
    ψ0, ψ1
end

function _dense_mutual_information_XR(ψ0, ψ1, L::Int, ell::Int)
    ψQR = PW.dense_max_entangled_state(ψ0, ψ1)
    rho_X = PW.dense_mixed_rdm_subsystem(ψQR, ψQR, collect(1:ell), L + 1)
    rho_R = PW.dense_mixed_rdm_subsystem(ψQR, ψQR, [L + 1], L + 1)
    rho_XR = PW.dense_mixed_rdm_subsystem(ψQR, ψQR, vcat(collect(1:ell), L + 1), L + 1)
    PW.entropy_bits_from_rho(rho_X) + PW.entropy_bits_from_rho(rho_R) -
        PW.entropy_bits_from_rho(rho_XR)
end

function _random_tvec_powerlaw(M, k::Real)
    B = PWM.MPS.rand_MPSTensor(ComplexF64, PWM.phys_dim(M), PWM.bond_dim(M))
    PWM.puMPSTvec(M, B, Float64(k))
end

@testset "Ising power-law observables" begin
    @testset "base-2 entropy and relative entropy" begin
        rho_mix = ComplexF64[0.5 0; 0 0.5]
        rho_pure = ComplexF64[1 0; 0 0]
        @test isapprox(PW.entropy_bits_from_rho(rho_mix), 1.0; atol=1e-12)
        @test isapprox(PW.relative_entropy_bits(rho_pure, rho_mix), 1.0; atol=1e-12)
        @test isapprox(PW.relative_entropy(rho_pure, rho_mix), log(2); atol=1e-12)
    end

    @testset "average relative entropy equals Holevo information" begin
        Random.seed!(2102)
        L = 5
        ell = 2
        ψ0, ψ1 = _orthonormal_dense_pair(2^L)
        rho0 = PW.dense_mixed_rdm_first_block(ψ0, ψ0, ell, L)
        rho1 = PW.dense_mixed_rdm_first_block(ψ1, ψ1, ell, L)
        rho_bar = (rho0 + rho1) ./ 2
        A = PW.average_relative_entropy_bits(rho0, rho1)
        holevo = PW.entropy_bits_from_rho(rho_bar) -
                 0.5 * PW.entropy_bits_from_rho(rho0) -
                 0.5 * PW.entropy_bits_from_rho(rho1)
        @test isapprox(A, holevo; atol=1e-11, rtol=1e-11)
    end

    @testset "complement identity for dense states" begin
        Random.seed!(2103)
        L = 5
        ell = 2
        ψ0, ψ1 = _orthonormal_dense_pair(2^L)
        rho0_X = PW.dense_mixed_rdm_first_block(ψ0, ψ0, ell, L)
        rho1_X = PW.dense_mixed_rdm_first_block(ψ1, ψ1, ell, L)
        rho0_Y = PW.dense_mixed_rdm_subsystem(ψ0, ψ0, collect((ell + 1):L), L)
        rho1_Y = PW.dense_mixed_rdm_subsystem(ψ1, ψ1, collect((ell + 1):L), L)
        A_X = PW.average_relative_entropy_bits(rho0_X, rho1_X)
        A_Y = PW.average_relative_entropy_bits(rho0_Y, rho1_Y)
        I_XR = _dense_mutual_information_XR(ψ0, ψ1, L, ell)
        @test isapprox(I_XR, 1 + A_X - A_Y; atol=1e-11, rtol=1e-11)
        @test isapprox(I_XR - A_X, 1 - A_Y; atol=1e-11, rtol=1e-11)
    end

    @testset "direct puMPS RDM equals dense-vector RDM" begin
        Random.seed!(2104)
        L = 6
        bond_dim = 2
        ell = 2
        M = PWM.rand_puMPState(ComplexF64, 2, bond_dim, L)
        T1 = _random_tvec_powerlaw(M, 0.0)
        T2 = _random_tvec_powerlaw(M, 1.0)
        for (a, b) in ((M, M), (M, T1), (T1, M), (T1, T1), (T1, T2))
            rho_direct = PW.mixed_reduced_density_matrix_first_block(a, b, ell)
            rho_dense = PW.dense_mixed_rdm_first_block(Vector(a), Vector(b), ell, L)
            @test isapprox(rho_direct, rho_dense; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "mixed-block Hermiticity and trace consistency" begin
        Random.seed!(2105)
        L = 6
        M = PWM.rand_puMPState(ComplexF64, 2, 2, L)
        T = _random_tvec_powerlaw(M, 0.0)
        rho_aa = PW.mixed_reduced_density_matrix_first_block(M, M, 2)
        rho_ab = PW.mixed_reduced_density_matrix_first_block(M, T, 2)
        rho_ba = PW.mixed_reduced_density_matrix_first_block(T, M, 2)
        rho_bb = PW.mixed_reduced_density_matrix_first_block(T, T, 2)
        @test isapprox(rho_ba, rho_ab'; atol=1e-8, rtol=1e-8)
        @test isapprox(real(tr(rho_aa)), 1.0; atol=1e-10)
        @test isapprox(real(tr(rho_bb)), 1.0; atol=1e-10)
        @test minimum(eigvals(Hermitian(rho_aa))) > -1e-9
        @test minimum(eigvals(Hermitian(rho_bb))) > -1e-9
    end

    @testset "block rho_XR matches dense maximally entangled calculation" begin
        Random.seed!(2106)
        L = 5
        ell = 2
        ψ0, ψ1 = _orthonormal_dense_pair(2^L)
        blocks = (
            rho00 = PW.dense_mixed_rdm_first_block(ψ0, ψ0, ell, L),
            rho01 = PW.dense_mixed_rdm_first_block(ψ0, ψ1, ell, L),
            rho10 = PW.dense_mixed_rdm_first_block(ψ1, ψ0, ell, L),
            rho11 = PW.dense_mixed_rdm_first_block(ψ1, ψ1, ell, L),
        )
        rho_XR_blocks = PW.rho_XR_from_code_blocks(blocks)
        ψQR = PW.dense_max_entangled_state(ψ0, ψ1)
        rho_XR_dense = PW.dense_mixed_rdm_subsystem(
            ψQR, ψQR, vcat(collect(1:ell), L + 1), L + 1)
        @test isapprox(rho_XR_blocks, rho_XR_dense; atol=1e-11, rtol=1e-11)
        I_blocks = PW.mutual_information_XR_from_code_blocks(blocks)
        I_dense = _dense_mutual_information_XR(ψ0, ψ1, L, ell)
        @test isapprox(I_blocks, I_dense; atol=1e-11, rtol=1e-11)
    end

    @testset "Gram-Schmidt reduced-block formulas" begin
        Random.seed!(2107)
        L = 5
        ell = 2
        ψa, ψorth = _orthonormal_dense_pair(2^L)
        c = 0.31 + 0.27im
        den = sqrt(1 - abs2(c))
        ψb = c .* ψa .+ den .* ψorth
        blocks = PW.gram_schmidt_reduced_blocks(
            PW.dense_mixed_rdm_first_block(ψa, ψa, ell, L),
            PW.dense_mixed_rdm_first_block(ψa, ψb, ell, L),
            PW.dense_mixed_rdm_first_block(ψb, ψa, ell, L),
            PW.dense_mixed_rdm_first_block(ψb, ψb, ell, L);
            overlap=dot(ψa, ψb),
        )
        ψtilde = (ψb .- c .* ψa) ./ den
        @test isapprox(blocks.rho01, PW.dense_mixed_rdm_first_block(ψa, ψtilde, ell, L);
                       atol=1e-11, rtol=1e-11)
        @test isapprox(blocks.rho10, PW.dense_mixed_rdm_first_block(ψtilde, ψa, ell, L);
                       atol=1e-11, rtol=1e-11)
        @test isapprox(blocks.rho11, PW.dense_mixed_rdm_first_block(ψtilde, ψtilde, ell, L);
                       atol=1e-11, rtol=1e-11)
    end

    @testset "phase invariance" begin
        Random.seed!(2108)
        L = 5
        ell = 2
        ψ0, ψ1 = _orthonormal_dense_pair(2^L)
        θ = 0.73
        function obs_for_pair(a, b)
            PW.powerlaw_observables_from_mixed_blocks(
                PW.dense_mixed_rdm_first_block(a, a, ell, L),
                PW.dense_mixed_rdm_first_block(a, b, ell, L),
                PW.dense_mixed_rdm_first_block(b, a, ell, L),
                PW.dense_mixed_rdm_first_block(b, b, ell, L),
            )
        end
        obs = obs_for_pair(ψ0, ψ1)
        obs_phase = obs_for_pair(ψ0, exp(1im * θ) .* ψ1)
        @test isapprox(obs.small_avg_relent_X, obs_phase.small_avg_relent_X;
                       atol=1e-11, rtol=1e-11)
        @test isapprox(obs.complement_proxy, obs_phase.complement_proxy;
                       atol=1e-11, rtol=1e-11)
        @test isapprox(obs.mutual_information_XR, obs_phase.mutual_information_XR;
                       atol=1e-11, rtol=1e-11)
    end

    @testset "power-law fitter on synthetic data" begin
        c = 1.7
        p = 2.25
        rows = [
            (quantity=PW.POWERLAW_QUANTITIES.small_avg_relent_X,
             code_label="I_sigma", x=1 / 8, value_bits=c * (1 / 8)^p),
            (quantity=PW.POWERLAW_QUANTITIES.small_avg_relent_X,
             code_label="I_sigma", x=1 / 4, value_bits=c * (1 / 4)^p),
            (quantity=PW.POWERLAW_QUANTITIES.small_avg_relent_X,
             code_label="I_sigma", x=0.25, value_bits=c * 0.25^p),
            (quantity=PW.POWERLAW_QUANTITIES.small_avg_relent_X,
             code_label="I_sigma", x=0.3, value_bits=c * 0.3^p),
            (quantity=PW.POWERLAW_QUANTITIES.small_avg_relent_X,
             code_label="I_sigma", x=1 / 2, value_bits=c * (1 / 2)^p),
            (quantity=PW.POWERLAW_QUANTITIES.small_avg_relent_X,
             code_label="I_sigma", x=1 / 16, value_bits=0.0),
        ]
        fit = PW.fit_power_law(rows;
                               quantity=PW.POWERLAW_QUANTITIES.small_avg_relent_X,
                               code_label="I_sigma")
        @test isapprox(fit.prefactor_c, c; atol=1e-12, rtol=1e-12)
        @test isapprox(fit.exponent, p; atol=1e-12, rtol=1e-12)
        @test fit.n_points == 3
        @test isapprox(fit.x_max, 0.25; atol=0, rtol=0)
        @test occursin("x <= 0.25", fit.fit_rule)
    end

    @testset "CSV schema and code-subspace exclusion" begin
        mktempdir() do dir
            quality = (E0=0.0, E1=1.0, E2=2.0, gap_ratio=2.0,
                       P0=1.0, P1=-1.0, P2=1.0)
            rows = NamedTuple[]
            for (L, ell) in ((6, 1), (8, 2)), code in ("I_sigma", "I_epsilon")
                A = 0.1 * ell / L
                B = 0.2 * ell / L
                for (quantity, value) in (
                    (PW.POWERLAW_QUANTITIES.small_avg_relent_X, A),
                    (PW.POWERLAW_QUANTITIES.complement_proxy, B),
                    (PW.POWERLAW_QUANTITIES.mutual_information_XR, A + B),
                )
                    push!(rows, PW.make_powerlaw_raw_row(
                        L=L, bond_dim=2, ell=ell, code_label=code,
                        state_0="I", state_1=code == "I_sigma" ? "sigma" : "epsilon",
                        quantity=quantity, value_bits=value,
                        quality=quality, overlap=0.0 + 0.0im, entropy_base=2,
                    ))
                end
            end
            PW.write_powerlaw_raw_csvs(dir, rows)
            PW.fit_powerlaw_observable_csvs(dir)
            for quantity in values(PW.POWERLAW_QUANTITIES)
                path = PW.raw_csv_path(dir, quantity)
                @test isfile(path)
                header = split(readline(path), ",")
                @test all(col in header for col in PW.POWERLAW_RAW_COLUMNS)
                csv_rows = PW.read_simple_csv(path)
                @test all(row.code_label in ("I_sigma", "I_epsilon") for row in csv_rows)
                @test all(row.code_label != "sigma_epsilon" for row in csv_rows)
                @test all(parse(Int, row.D_code) == 2 for row in csv_rows)
                @test all(parse(Float64, row.entropy_base) == 2.0 for row in csv_rows)
            end
            @test isfile(joinpath(dir, PW.POWERLAW_FIT_FILENAME))

            raw = PW.existing_powerlaw_rows(dir)
            by_key = Dict{Tuple{Int,Int,String,String},Float64}()
            for row in raw
                by_key[(parse(Int, row.L), parse(Int, row.ell),
                        row.code_label, row.quantity)] = parse(Float64, row.value_bits)
            end
            for (L, ell) in ((6, 1), (8, 2)), code in ("I_sigma", "I_epsilon")
                A = by_key[(L, ell, code, PW.POWERLAW_QUANTITIES.small_avg_relent_X)]
                B = by_key[(L, ell, code, PW.POWERLAW_QUANTITIES.complement_proxy)]
                Δ = by_key[(L, ell, code, PW.POWERLAW_QUANTITIES.mutual_information_XR)]
                @test isapprox(Δ, A + B; atol=1e-14, rtol=1e-14)
            end
        end
    end

    @testset "production path does not call dense tangent-vector conversion" begin
        source = read(joinpath(@__DIR__, "..", "examples",
                               "ising_powerlaw_observable_scan.jl"), String)
        rows_body = match(r"function compute_powerlaw_rows_for_states[\s\S]*?function run_ising_powerlaw_observable_scan", source)
        scan_body = match(r"function run_ising_powerlaw_observable_scan[\s\S]*?if abspath", source)
        @test rows_body !== nothing
        @test scan_body !== nothing
        @test !occursin("Vector(", rows_body.match)
        @test !occursin("Vector(", scan_body.match)
        @test occursin("mixed_reduced_density_matrix_first_block", rows_body.match)
    end
end
