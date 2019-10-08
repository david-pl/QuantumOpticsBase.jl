import Base: ==, *, /, +, -, Broadcast
import SparseArrays: sparse

"""
    sparse(op::AbstractOperator)

Convert an arbitrary operator into a [`SparseOperator`](@ref).
"""
sparse(a::AbstractOperator) = throw(ArgumentError("Direct conversion from $(typeof(a)) not implemented. Use sparse(dense(op)) instead."))
sparse(a::SparseOperator) = copy(a)
sparse(a::Operator) = Operator(a.basis_l, a.basis_r, sparse(a.data))

# Arithmetic operations
function ptrace(op::SparseOperator, indices::Vector{Int})
    check_ptrace_arguments(op, indices)
    shape = [op.basis_l.shape; op.basis_r.shape]
    data = ptrace(op.data, shape, indices)
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    Operator(b_l, b_r, data)
end

function expect(op::SparseOperator{B,B}, state::Ket{B}) where B<:Basis
    state.data' * op.data * state.data
end


function expect(op::SparseOperator{B1,B2}, state::DenseOperator{B2,B2}) where {B1<:Basis,B2<:Basis}
    result = ComplexF64(0.)
    @inbounds for colindex = 1:op.data.n
        for i=op.data.colptr[colindex]:op.data.colptr[colindex+1]-1
            result += op.data.nzval[i]*state.data[colindex, op.data.rowval[i]]
        end
    end
    result
end

function permutesystems(rho::SparseOperator{B1,B2}, perm::Vector{Int}) where {B1<:CompositeBasis,B2<:CompositeBasis}
    @assert length(rho.basis_l.bases) == length(rho.basis_r.bases) == length(perm)
    @assert isperm(perm)
    shape = [rho.basis_l.shape; rho.basis_r.shape]
    data = permutedims(rho.data, shape, [perm; perm .+ length(perm)])
    Operator(permutesystems(rho.basis_l, perm), permutesystems(rho.basis_r, perm), data)
end

identityoperator(::Type{T}, b1::Basis, b2::Basis) where {T<:SparseOperator} = Operator(b1, b2, sparse(ComplexF64(1)*I, length(b1), length(b2)))
identityoperator(::Type{T}, b1::Basis, b2::Basis) where {T<:DataOperator} = Operator(b1, b2, sparse(ComplexF64(1)*I, length(b1), length(b2)))
identityoperator(b1::Basis, b2::Basis) = identityoperator(SparseOperator, b1, b2)
identityoperator(b::Basis) = identityoperator(b, b)

"""
    diagonaloperator(b::Basis)

Create a diagonal operator of type [`SparseOperator`](@ref).
"""
function diagonaloperator(b::Basis, diag::Vector{T}) where T <: Number
  @assert 1 <= length(diag) <= prod(b.shape)
  Operator(b, sparse(Diagonal(convert(Vector{ComplexF64}, diag))))
end


# Fast in-place multiplication implementations
gemm!(alpha, M::SparseOperator{B1,B2}, b::DenseOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = gemm!(convert(ComplexF64, alpha), M.data, b.data, convert(ComplexF64, beta), result.data)
gemm!(alpha, a::DenseOperator{B1,B2}, M::SparseOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = gemm!(convert(ComplexF64, alpha), a.data, M.data, convert(ComplexF64, beta), result.data)
gemv!(alpha, M::SparseOperator{B1,B2}, b::Ket{B2}, beta, result::Ket{B1}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), M.data, b.data, convert(ComplexF64, beta), result.data)
gemv!(alpha, b::Bra{B1}, M::SparseOperator{B1,B2}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), b.data, M.data, convert(ComplexF64, beta), result.data)

# TODO: adjoints

# Broadcasting
struct SparseOperatorStyle{BL<:Basis,BR<:Basis} <: DataOperatorStyle{BL,BR} end
Broadcast.BroadcastStyle(::Type{<:SparseOperator{BL,BR}}) where {BL<:Basis,BR<:Basis} = SparseOperatorStyle{BL,BR}()
Broadcast.BroadcastStyle(::DenseOperatorStyle{B1,B2}, ::SparseOperatorStyle{B1,B2}) where {B1<:Basis,B2<:Basis} = DenseOperatorStyle{B1,B2}()
Broadcast.BroadcastStyle(::DenseOperatorStyle{B1,B2}, ::SparseOperatorStyle{B3,B4}) where {B1<:Basis,B2<:Basis,B3<:Basis,B4<:Basis} = throw(IncompatibleBases())
Broadcast.BroadcastStyle(::SparseOperatorStyle{B1,B2}, ::SparseOperatorStyle{B3,B4}) where {B1<:Basis,B2<:Basis,B3<:Basis,B4<:Basis} = throw(IncompatibleBases())

@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL<:Basis,BR<:Basis,Style<:SparseOperatorStyle{BL,BR},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bl,br = find_basis(bcf.args)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    return Operator{BL,BR}(bl, br, copy(bc_))
end
