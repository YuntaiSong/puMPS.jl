export ABlocks, BBlocks, build_A_from_blocks, build_B_from_blocks,
       z2_block_dims, z2_bond_representation, z2_allowed_indices

"""
    ABlocks

Block decomposition for Z₂-even puMPS tensors.
"""
struct ABlocks{T}
    A_up_ee::Matrix{T}
    A_up_oo::Matrix{T}
    A_dn_eo::Matrix{T}
    A_dn_oe::Matrix{T}
end

"""
    BBlocks

Container holding all potential blocks of a Z₂-definite tangent tensor.
"""
struct BBlocks{T}
    B_up_ee::Matrix{T}
    B_up_oo::Matrix{T}
    B_up_eo::Matrix{T}
    B_up_oe::Matrix{T}
    B_dn_ee::Matrix{T}
    B_dn_oo::Matrix{T}
    B_dn_eo::Matrix{T}
    B_dn_oe::Matrix{T}
end

function z2_block_dims(D::Integer)
    D_even = D ÷ 2
    D_odd = D - D_even
    D_even > 0 && D_odd > 0 || throw(ArgumentError("Z₂ block structure requires bond dimension ≥ 2, got $D"))
    D_even, D_odd
end

function build_A_from_blocks(blocks::ABlocks{T})::MPSTensor{T} where {T}
    A_up_ee, A_up_oo = blocks.A_up_ee, blocks.A_up_oo
    A_dn_eo, A_dn_oe = blocks.A_dn_eo, blocks.A_dn_oe

    D_e, D_o = size(A_up_ee, 1), size(A_up_oo, 1)
    D = D_e + D_o
    A = zeros(T, D, 2, D)

    A[1:D_e, 1, 1:D_e] .= A_up_ee
    A[D_e+1:D, 1, D_e+1:D] .= A_up_oo
    A[1:D_e, 2, D_e+1:D] .= A_dn_eo
    A[D_e+1:D, 2, 1:D_e] .= A_dn_oe
    A
end

function build_B_from_blocks(blocks::BBlocks{T}, parity::Int)::MPSTensor{T} where {T}
    parity in (-1, 1) || throw(ArgumentError("parity must be ±1, got $parity"))

    D_e = size(blocks.B_up_ee, 1)
    D_o = size(blocks.B_up_oo, 1)
    D = D_e + D_o
    B = zeros(T, D, 2, D)

    if parity == 1
        B[1:D_e, 1, 1:D_e] .= blocks.B_up_ee
        B[D_e+1:D, 1, D_e+1:D] .= blocks.B_up_oo
        B[1:D_e, 2, D_e+1:D] .= blocks.B_dn_eo
        B[D_e+1:D, 2, 1:D_e] .= blocks.B_dn_oe
    else
        B[1:D_e, 1, D_e+1:D] .= blocks.B_up_eo
        B[D_e+1:D, 1, 1:D_e] .= blocks.B_up_oe
        B[1:D_e, 2, 1:D_e] .= blocks.B_dn_ee
        B[D_e+1:D, 2, D_e+1:D] .= blocks.B_dn_oo
    end

    B
end

function z2_bond_representation(D::Integer, ::Type{T}=Float64) where {T}
    D_e, D_o = z2_block_dims(D)
    Diagonal(vcat(ones(T, D_e), -ones(T, D_o)))
end

z2_bond_representation(M::puMPState{T}) where {T} =
    z2_bond_representation(bond_dim(M), T)

function z2_allowed_indices(D::Integer, parity::Int; d::Integer=2)
    parity in (-1, 1) || throw(ArgumentError("parity must be ±1, got $parity"))
    D_e, D_o = z2_block_dims(D)
    mask = falses(D, d, D)
    if parity == 1
        mask[1:D_e, 1, 1:D_e] .= true
        mask[D_e+1:D, 1, D_e+1:D] .= true
        mask[1:D_e, 2, D_e+1:D] .= true
        mask[D_e+1:D, 2, 1:D_e] .= true
    else
        mask[1:D_e, 1, D_e+1:D] .= true
        mask[D_e+1:D, 1, 1:D_e] .= true
        mask[1:D_e, 2, 1:D_e] .= true
        mask[D_e+1:D, 2, D_e+1:D] .= true
    end
    LI = LinearIndices(mask)
    inds = Vector{Int}(undef, count(mask))
    idx = 1
    for CI in CartesianIndices(mask)
        if mask[CI]
            inds[idx] = LI[CI]
            idx += 1
        end
    end
    inds
end
