using LinearAlgebra
using Random
using Test

module IsingExampleDirectRDMTest
include(joinpath(@__DIR__, "..", "examples", "ising_example.jl"))
end

const EX = IsingExampleDirectRDMTest
const EPM = EX.PM

function _random_tvec(M, k::Real)
    B = EPM.MPS.rand_MPSTensor(ComplexF64, EPM.phys_dim(M), EPM.bond_dim(M))
    EPM.puMPSTvec(M, B, Float64(k))
end

function _dense_mixed_rdm(a, b, ell::Int, N::Int; d::Int=2)
    psi_a = EX.normalize_state_vector(Vector(a))
    psi_b = EX.normalize_state_vector(Vector(b))
    psi_a_mat = reshape(psi_a, d^ell, d^(N - ell))
    psi_b_mat = reshape(psi_b, d^ell, d^(N - ell))
    psi_a_mat * psi_b_mat'
end

@testset "direct puMPS reduced density matrices" begin
    @testset "trace-kernel reshuffling" begin
        Random.seed!(1101)
        for D in (2, 3)
            P = randn(ComplexF64, D, D)
            Q = randn(ComplexF64, D, D)
            Tenv = randn(ComplexF64, D * D, D * D)
            lhs = tr(kron(P, conj(Q)) * Tenv)
            rp = reshape(vec(P), 1, :)
            rq = reshape(vec(Q), 1, :)
            rhs = (rp * EX.trace_kernel_from_transfer(Tenv, D) * rq')[1]
            @test isapprox(lhs, rhs; atol=1e-10, rtol=1e-10)
        end
    end

    @testset "ground-state direct RDM vs dense vector" begin
        Random.seed!(1102)
        N = 6
        D = 3
        M = EPM.rand_puMPState(ComplexF64, 2, D, N)
        psi = EX.normalize_state_vector(Vector(M))
        for ell in 1:3
            rho_direct = EX.mixed_reduced_density_matrix_first_block(M, M, ell)
            rho_dense = EX.reduced_density_matrix_from_vector(psi, ell, N)
            @test isapprox(rho_direct, rho_dense; atol=1e-9, rtol=1e-9)
            @test isapprox(real(tr(rho_direct)), 1.0; atol=1e-10)
            @test norm(rho_direct - rho_direct') < 1e-10
            @test minimum(eigvals(Hermitian(rho_direct))) > -1e-9
        end
    end

    @testset "tangent diagonal RDM vs dense vector" begin
        Random.seed!(1103)
        N = 6
        D = 2
        for k in (0.0, 1.0)
            M = EPM.rand_puMPState(ComplexF64, 2, D, N)
            Tvec = _random_tvec(M, k)
            psi = EX.normalize_state_vector(Vector(Tvec))
            for ell in 1:2
                rho_direct = EX.mixed_reduced_density_matrix_first_block(Tvec, Tvec, ell)
                rho_dense = EX.reduced_density_matrix_from_vector(psi, ell, N)
                @test isapprox(rho_direct, rho_dense; atol=1e-8, rtol=1e-8)
            end
        end
    end

    @testset "mixed off-diagonal RDMs vs dense vector" begin
        Random.seed!(1104)
        N = 6
        D = 2
        M = EPM.rand_puMPState(ComplexF64, 2, D, N)
        Tvec1 = _random_tvec(M, 0.0)
        Tvec2 = _random_tvec(M, 1.0)
        for (a, b) in ((M, Tvec1), (Tvec1, Tvec2))
            psi_a = EX.normalize_state_vector(Vector(a))
            psi_b = EX.normalize_state_vector(Vector(b))
            for ell in 1:2
                rho_ab_direct = EX.mixed_reduced_density_matrix_first_block(a, b, ell)
                rho_ba_direct = EX.mixed_reduced_density_matrix_first_block(b, a, ell)
                rho_ab_dense = _dense_mixed_rdm(a, b, ell, N)
                @test isapprox(rho_ab_direct, rho_ab_dense; atol=1e-8, rtol=1e-8)
                @test isapprox(rho_ba_direct, rho_ab_direct'; atol=1e-8, rtol=1e-8)
                c_dense = dot(psi_a, psi_b)
                c_direct = conj(tr(rho_ab_direct))
                @test isapprox(c_direct, c_dense; atol=1e-8, rtol=1e-8)
            end
        end
    end

    @testset "direct mutual information vs dense purification" begin
        Random.seed!(1105)
        N = 6
        D = 2
        M = EPM.rand_puMPState(ComplexF64, 2, D, N)
        Tvec1 = _random_tvec(M, 0.0)
        Tvec2 = _random_tvec(M, 1.0)
        for (a, b) in ((M, Tvec1), (Tvec1, Tvec2))
            psi_a = EX.normalize_state_vector(Vector(a))
            psi_b = EX.normalize_state_vector(Vector(b))
            for ell in 1:2
                I_direct = EX.pair_mutual_information_values_direct(a, b, [ell]; base=2)[1]
                I_dense = EX.pair_mutual_information_values(psi_a, psi_b, N, [ell]; base=2)[1]
                @test isapprox(I_direct, I_dense; atol=1e-8, rtol=1e-8)
            end
        end
    end

    @testset "direct relative entropy vs dense-vector RDMs" begin
        Random.seed!(1106)
        N = 6
        D = 2
        M = EPM.rand_puMPState(ComplexF64, 2, D, N)
        Tvec1 = _random_tvec(M, 0.0)
        Tvec2 = _random_tvec(M, 1.0)
        for (a, b) in ((M, Tvec1), (Tvec1, Tvec2))
            psi_a = EX.normalize_state_vector(Vector(a))
            psi_b = EX.normalize_state_vector(Vector(b))
            for ell in 1:2
                rho_a_direct = EX.reduced_density_matrix_first_block_direct(a, ell)
                rho_b_direct = EX.reduced_density_matrix_first_block_direct(b, ell)
                S_direct = EX.relative_entropy(rho_a_direct, rho_b_direct)
                rho_a_dense = EX.reduced_density_matrix_from_vector(psi_a, ell, N)
                rho_b_dense = EX.reduced_density_matrix_from_vector(psi_b, ell, N)
                S_dense = EX.relative_entropy(rho_a_dense, rho_b_dense)
                @test isapprox(S_direct, S_dense; atol=1e-8, rtol=1e-8)
            end
        end
    end

    @testset "direct entanglement rows accept compressed states" begin
        Random.seed!(1107)
        N = 6
        D = 2
        M = EPM.rand_puMPState(ComplexF64, 2, D, N)
        Tvec1 = _random_tvec(M, 0.0)
        Tvec2 = _random_tvec(M, 1.0)
        rows = EX.entanglement_rows_for_pumps_states_direct(M, Tvec1, Tvec2, N;
                                                            ell_values=1:2)
        @test length(rows) == 2
        @test all(isfinite(row.I01) && isfinite(row.I02) && isfinite(row.I12)
                  for row in rows)
        @test all(isfinite(row.S10) && isfinite(row.S20) && isfinite(row.S12)
                  for row in rows)
    end
end
