import Base: ==, *, /, +, -, Broadcast
import SparseArrays: sparse

"""
    SparseOperator(b1[, b2, data])

Sparse array implementation of Operator.

The matrix is stored as the julia built-in type `SparseMatrixCSC`
in the `data` field.
"""
mutable struct SparseOperator{BL<:Basis,BR<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}} <: DataOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    data::T
    function SparseOperator{BL,BR,T}(b1::Basis, b2::Basis, data::T) where {BL<:Basis,BR<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}}
        if length(b1) != size(data, 1) || length(b2) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(b1, b2, data)
    end
end

SparseOperator(b1::BL, b2::BR, data::T) where {BL<:Basis,BR<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}} = SparseOperator{BL,BR,T}(b1, b2, data)
SparseOperator{BL,BR}(b1::BL, b2::BR, data::T) where {BL<:Basis,BR<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}} = SparseOperator{BL,BR,T}(b1, b2, data)
SparseOperator(b1::Basis, b2::Basis, data) = SparseOperator(b1, b2, convert(SparseMatrixCSC{ComplexF64,Int}, data))
SparseOperator(b::Basis, data::SparseMatrixCSC{ComplexF64, Int}) = SparseOperator(b, b, data)
SparseOperator(b::Basis, data::Matrix{ComplexF64}) = SparseOperator(b, sparse(data))
SparseOperator(op::DenseOperator) = SparseOperator(op.basis_l, op.basis_r, sparse(op.data))

SparseOperator(b1::Basis, b2::Basis) = SparseOperator(b1, b2, spzeros(ComplexF64, length(b1), length(b2)))
SparseOperator{BL,BR}(b1::BL, b2::BR) where {BL<:Basis,BR<:Basis} = SparseOperator{BL,BR}(b1, b2, spzeros(ComplexF64, length(b1), length(b2)))
SparseOperator(b::Basis) = SparseOperator(b, b)

Base.copy(x::SparseOperator) = SparseOperator(x.basis_l, x.basis_r, copy(x.data))
dense(a::SparseOperator) = DenseOperator(a.basis_l, a.basis_r, Matrix(a.data))

"""
    sparse(op::AbstractOperator)

Convert an arbitrary operator into a [`SparseOperator`](@ref).
"""
sparse(a::AbstractOperator) = throw(ArgumentError("Direct conversion from $(typeof(a)) not implemented. Use sparse(full(op)) instead."))
sparse(a::SparseOperator) = copy(a)
sparse(a::DenseOperator) = SparseOperator(a.basis_l, a.basis_r, sparse(a.data))

==(x::SparseOperator, y::SparseOperator) = false
==(x::T, y::T) where T<:SparseOperator = (x.data == y.data)


# Arithmetic operations
+(a::SparseOperator, b::SparseOperator) = throw(IncompatibleBases())
+(a::T, b::T) where T<:SparseOperator = T(a.basis_l, a.basis_r, a.data+b.data)
+(a::SparseOperator, b::DenseOperator) = throw(IncompatibleBases())
+(a::SparseOperator{B1,B2}, b::DenseOperator{B1,B2}) where {B1<:Basis,B2<:Basis} = DenseOperator{B1,B2}(a.basis_l, a.basis_r, a.data+b.data)
+(a::DenseOperator, b::SparseOperator) = throw(IncompatibleBases())
+(a::DenseOperator{B1,B2}, b::SparseOperator{B1,B2}) where {B1<:Basis,B2<:Basis} = DenseOperator{B1,B2}(a.basis_l, a.basis_r, a.data+b.data)

-(a::T) where T<:SparseOperator = T(a.basis_l, a.basis_r, -a.data)
-(a::SparseOperator, b::SparseOperator) = throw(IncompatibleBases())
-(a::T, b::T) where T<:SparseOperator = SparseOperator(a.basis_l, a.basis_r, a.data-b.data)
-(a::SparseOperator, b::DenseOperator) = throw(IncompatibleBases())
-(a::SparseOperator{B1,B2}, b::DenseOperator{B1,B2}) where {B1<:Basis,B2<:Basis} = DenseOperator{B1,B2}(a.basis_l, a.basis_r, a.data-b.data)
-(a::DenseOperator, b::SparseOperator) = throw(IncompatibleBases())
-(a::DenseOperator{B1,B2}, b::SparseOperator{B1,B2}) where {B1<:Basis,B2<:Basis} = DenseOperator{B1,B2}(a.basis_l, a.basis_r, a.data-b.data)

*(a::SparseOperator{B1,B2}, b::SparseOperator{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = SparseOperator{B1,B3}(a.basis_l, b.basis_r, a.data*b.data)
*(a::SparseOperator, b::SparseOperator) = throw(IncompatibleBases())
*(a::T, b::Number) where T<:SparseOperator = T(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::T) where T<:SparseOperator = T(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::T, b::Number) where T<:SparseOperator = T(a.basis_l, a.basis_r, a.data/complex(b))

dagger(x::SparseOperator) = SparseOperator(x.basis_r, x.basis_l, x.data')
ishermitian(A::SparseOperator) = false
ishermitian(A::SparseOperator{B,B}) where B<:Basis = ishermitian(A.data)

tensor(a::SparseOperator, b::SparseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))
tensor(a::DenseOperator, b::SparseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))
tensor(a::SparseOperator, b::DenseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))

tr(op::SparseOperator{B,B}) where B<:Basis = tr(op.data)
tr(op::SparseOperator) = throw(IncompatibleBases())

conj(op::T) where T<:SparseOperator = T(op.basis_l, op.basis_r, conj(op.data))
conj!(op::SparseOperator) = conj!(op.data)

transpose(op::SparseOperator{BL,BR,T}) where {BL<:Basis,BR<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}} = SparseOperator{BR,BL,T}(op.basis_r, op.basis_l, T(transpose(op.data)))

function ptrace(op::SparseOperator, indices::Vector{Int})
    check_ptrace_arguments(op, indices)
    shape = [op.basis_l.shape; op.basis_r.shape]
    data = ptrace(op.data, shape, indices)
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    SparseOperator(b_l, b_r, data)
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
    SparseOperator(permutesystems(rho.basis_l, perm), permutesystems(rho.basis_r, perm), data)
end

identityoperator(::Type{T}, b1::Basis, b2::Basis) where {T<:SparseOperator} = SparseOperator(b1, b2, sparse(ComplexF64(1)*I, length(b1), length(b2)))
identityoperator(::Type{T}, b1::Basis, b2::Basis) where {T<:DataOperator} = SparseOperator(b1, b2, sparse(ComplexF64(1)*I, length(b1), length(b2)))
identityoperator(b1::Basis, b2::Basis) = identityoperator(SparseOperator, b1, b2)
identityoperator(b::Basis) = identityoperator(b, b)

"""
    diagonaloperator(b::Basis)

Create a diagonal operator of type [`SparseOperator`](@ref).
"""
function diagonaloperator(b::Basis, diag::Vector{T}) where T <: Number
  @assert 1 <= length(diag) <= prod(b.shape)
  SparseOperator(b, sparse(Diagonal(convert(Vector{ComplexF64}, diag))))
end


# Fast in-place multiplication implementations
mul!(result::DenseOperator{B1,B3},M::SparseOperator{B1,B2},b::DenseOperator{B2,B3},alpha,beta) where {B1<:Basis,B2<:Basis,B3<:Basis} = mul!(result.data,M.data,b.data,alpha,beta)
mul!(result::DenseOperator{B1,B3},a::DenseOperator{B1,B2},M::SparseOperator{B2,B3},alpha,beta) where {B1<:Basis,B2<:Basis,B3<:Basis} = mul!(result.data,a.data,M.data,alpha,beta)
mul!(result::Ket{B1},M::SparseOperator{B1,B2},b::Ket{B2},alpha,beta) where {B1<:Basis,B2<:Basis} = mul!(result.data,M.data,b.data,alpha,beta)
mul!(result::Bra{B2},b::Bra{B1},M::SparseOperator{B1,B2},alpha,beta) where {B1<:Basis,B2<:Basis} = mul!(result.data,b.data,M.data,alpha,beta)

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
    return SparseOperator{BL,BR}(bl, br, copy(bc_))
end
