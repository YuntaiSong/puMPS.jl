# Load the puMPS code from the src/ folder (one directory up from examples/)
include(joinpath(@__DIR__, "..", "src", "puMPS.jl"))
import .puMPS

using PyPlot

const PM = puMPS

delta = 0.5

D = 8
N = 32

M = PM.rand_puMPState(ComplexF64, 2, D, N)

H = PM.ANNNI_local_MPO(ComplexF64, delta1=delta, delta2=delta)

PM.vumps_opt!(M, H, 1e-6, maxitr=5) #Pre-optimization using DMRG-like method
PM.minimize_energy_local!(M, H, 100, step=0.1)

println("Computing excitations!")

ks_tocompute = [-2,-1,0,1,2]
num_states = [5,4,7,4,5]

ens, ks, exs = PM.excitations!(M, PM.ANNNI_PBC_MPO_split(ComplexF64, delta1=delta, delta2=delta), ks_tocompute, num_states)

H1 = PM.Hn_in_basis(M, PM.ANNNI_Hn_MPO_split(ComplexF64, 1, N, delta1=delta, delta2=delta), exs, ks_tocompute)
H2 = PM.Hn_in_basis(M, PM.ANNNI_Hn_MPO_split(ComplexF64, 2, N, delta1=delta, delta2=delta), exs, ks_tocompute)

ind1 = argmin(real.(ens))
indT = argmax(abs.(H2[:,ind1]))

en1 = ens[ind1]
enT = ens[indT]

fac = 2.0 / (enT-en1)

c = 2*abs2(H2[indT,ind1] * fac)
@show real(c)

plot(ks, real(ens), "o")
show()
