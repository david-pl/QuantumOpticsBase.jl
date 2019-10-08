import Base: ==, +, -, *, /, Broadcast
using Base.Cartesian

Operator{BL,BR}(b1::BL, b2::BR, data::T) where {BL<:Basis,BR<:Basis,T<:Matrix{ComplexF64}} = Operator{BL,BR,T}(b1, b2, data)
Operator(b1::BL, b2::BR, data::T) where {BL<:Basis,BR<:Basis,T<:Matrix{ComplexF64}} = Operator{BL,BR,T}(b1, b2, data)
Operator(b1::Basis, b2::Basis) = Operator(b1, b2, zeros(ComplexF64, length(b1), length(b2)))
Operator(b::Basis) = Operator(b, b)

Base.copy(x::T) where T<:Operator = T(x.basis_l, x.basis_r, copy(x.data))

"""
    dense(op::AbstractOperator)

Convert an arbitrary Operator into a [`DenseOperator`](@ref).
"""
dense(x::Operator) = Operator(x.basis_l,x.basis_r,Matrix(x.data))
# dense(x::Operator{BL,BR,T}) where {BL,BR,T<:Adjoint} = Operator(x.basis_r,x.basis_l,Matrix(x.data)')
# dense(x::DenseOperator) = copy(x)

==(x::Operator{BL,BR}, y::Operator{BL,BR}) where {BL,BR} = x.data == y.data

# Arithmetic operations
+(a::Operator{BL,BR}, b::Operator{BL,BR}) where {BL,BR} = Operator(a.basis_l, a.basis_r, a.data+b.data)
# +(a::Operator{B1,B2}, b::Operator{B3,B4}) where {B1,B2,B3,B4} = throw(IncompatibleBases())

-(a::T) where T<:Operator = T(a.basis_l, a.basis_r, -a.data)
-(a::Operator{BL,BR}, b::Operator{BL,BR}) where {BL,BR} = Operator(a.basis_l, a.basis_r, a.data-b.data)
# -(a::Operator, b::Operator) = throw(IncompatibleBases())

*(a::Operator{BL,BR}, b::Ket{BR}) where {BL<:Basis,BR<:Basis} = Ket{BL}(a.basis_l, a.data*b.data)
# *(a::Operator, b::Ket) = throw(IncompatibleBases())
*(a::Bra{BL}, b::Operator{BL,BR}) where {BL<:Basis,BR<:Basis} = Bra{BR}(b.basis_r, transpose(b.data)*a.data)
# *(a::Bra, b::Operator) = throw(IncompatibleBases())
*(a::Operator{B1,B2}, b::Operator{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = Operator(a.basis_l, b.basis_r, a.data*b.data)
# *(a::Operator, b::Operator) = throw(IncompatibleBases())
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, b .* a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, a .* b.data)


# function *(op1::AbstractOperator{B1,B2}, op2::DenseOperator{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
#     result = Operator(op1.basis_l, op2.basis_r)
#     gemm!(Complex(1.), op1, op2, Complex(0.), result)
#     return result
# end
# function *(op1::DenseOperator{B1,B2}, op2::AbstractOperator{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
#     result = Operator{B1,B3}(op1.basis_l, op2.basis_r)
#     gemm!(Complex(1.), op1, op2, Complex(0.), result)
#     return result
# end
function *(op::AbstractOperator{BL,BR}, psi::Ket{BR}) where {BL<:Basis,BR<:Basis}
    result = Ket{BL}(op.basis_l)
    gemv!(Complex(1.), op, psi, Complex(0.), result)
    return result
end
function *(psi::Bra{BL}, op::AbstractOperator{BL,BR}) where {BL<:Basis,BR<:Basis}
    result = Bra{BR}(op.basis_r)
    gemv!(Complex(1.), psi, op, Complex(0.), result)
    return result
end

/(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, a.data ./ b)


dagger(x::Operator) = Operator(x.basis_r, x.basis_l, x.data')

ishermitian(A::Operator) = false
ishermitian(A::Operator{B,B}) where B<:Basis = ishermitian(A.data)

tensor(a::DataOperator, b::DataOperator) = Operator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))

conj(a::Operator) = Operator(a.basis_l, a.basis_r, conj(a.data))
conj!(a::Operator) = conj!(a.data)

transpose(op::Operator) = Operator(op.basis_r, op.basis_l, transpose(op.data))

"""
    tensor(x::Ket, y::Bra)

Outer product ``|x⟩⟨y|`` of the given states.
"""
tensor(a::Ket, b::Bra) = Operator(a.basis, b.basis, reshape(kron(b.data, a.data), prod(a.basis.shape), prod(b.basis.shape)))


tr(op::DataOperator{B,B}) where B<:Basis = tr(op.data)

function ptrace(a::DataOperator, indices::Vector{Int})
    check_ptrace_arguments(a, indices)
    rank = length(a.basis_l.shape)
    result = _ptrace(Val{rank}, a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return Operator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end

function ptrace(psi::Ket, indices::Vector{Int})
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_ket(Val{rank}, psi.data, b.shape, indices)
    return Operator(b_, b_, result)
end
function ptrace(psi::Bra, indices::Vector{Int})
    check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_bra(Val{rank}, psi.data, b.shape, indices)
    return Operator(b_, b_, result)
end

normalize!(op::DenseOperator) = (rmul!(op.data, 1.0/tr(op)); nothing)

function expect(op::DenseOperator{B,B}, state::Ket{B}) where B<:Basis
    state.data' * op.data * state.data
end

function expect(op::DenseOperator{B1,B2}, state::AbstractOperator{B2,B2}) where {B1<:Basis,B2<:Basis}
    result = ComplexF64(0.)
    @inbounds for i=1:size(op.data, 1), j=1:size(op.data,2)
        result += op.data[i,j]*state.data[j,i]
    end
    result
end

function exp(op::T) where {B<:Basis,T<:DenseOperator{B,B}}
    return Operator(op.basis_l, op.basis_r, exp(op.data))
end

function permutesystems(a::DenseOperator{B1,B2}, perm::Vector{Int}) where {B1<:CompositeBasis,B2<:CompositeBasis}
    @assert length(a.basis_l.bases) == length(a.basis_r.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(a.data, [a.basis_l.shape; a.basis_r.shape]...)
    data = permutedims(data, [perm; perm .+ length(perm)])
    data = reshape(data, length(a.basis_l), length(a.basis_r))
    Operator(permutesystems(a.basis_l, perm), permutesystems(a.basis_r, perm), data)
end

identityoperator(::Type{T}, b1::Basis, b2::Basis) where {T<:DenseOperator} = Operator(b1, b2, Matrix{ComplexF64}(I, length(b1), length(b2)))

"""
    projector(a::Ket, b::Bra)

Projection operator ``|a⟩⟨b|``.
"""
projector(a::Ket, b::Bra) = tensor(a, b)
"""
    projector(a::Ket)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Ket) = tensor(a, dagger(a))
"""
    projector(a::Bra)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Bra) = tensor(dagger(a), a)

"""
    dm(a::StateVector)

Create density matrix ``|a⟩⟨a|``. Same as `projector(a)`.
"""
dm(x::Ket) = tensor(x, dagger(x))
dm(x::Bra) = tensor(dagger(x), x)


# Partial trace implementation for dense operators.
function _strides(shape::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[1] = 1
    for m=2:N
        S[m] = S[m-1]*shape[m-1]
    end
    return S
end

# Dense operator version
@generated function _ptrace(::Type{Val{RANK}}, a::Matrix{ComplexF64},
                            shape_l::Vector{Int}, shape_r::Vector{Int},
                            indices::Vector{Int}) where RANK
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = copy(shape_l)
        result_shape_l[indices] .= 1
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = copy(shape_r)
        result_shape_r[indices] .= 1
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(ComplexF64, N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

@generated function _ptrace_ket(::Type{Val{RANK}}, a::Vector{ComplexF64},
                            shape::Vector{Int}, indices::Vector{Int}) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        result_shape[indices] .= 1
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(ComplexF64, N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0]*conj(a[Ir_0])
            end
        end
        return result
    end
end

@generated function _ptrace_bra(::Type{Val{RANK}}, a::Vector{ComplexF64},
                            shape::Vector{Int}, indices::Vector{Int}) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        result_shape[indices] .= 1
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(ComplexF64, N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += conj(a[Il_0])*a[Ir_0]
            end
        end
        return result
    end
end

# Fast in-place multiplication
gemm!(alpha, a::Matrix{ComplexF64}, b::Matrix{ComplexF64}, beta, result::Matrix{ComplexF64}) = BLAS.gemm!('N', 'N', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
gemv!(alpha, a::Matrix{ComplexF64}, b::Vector{ComplexF64}, beta, result::Vector{ComplexF64}) = BLAS.gemv!('N', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
gemv!(alpha, a::Vector{ComplexF64}, b::Matrix{ComplexF64}, beta, result::Vector{ComplexF64}) = BLAS.gemv!('T', convert(ComplexF64, alpha), b, a, convert(ComplexF64, beta), result)

gemm!(alpha, a::Adjoint{ComplexF64,<:Matrix}, b::Adjoint{ComplexF64,<:Matrix}, beta, result::Matrix{ComplexF64}) = BLAS.gemm!('C', 'C', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
gemm!(alpha, a::Adjoint{ComplexF64,<:Matrix}, b::Matrix{ComplexF64}, beta, result::Matrix{ComplexF64}) = BLAS.gemm!('C', 'N', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
gemm!(alpha, a::Matrix{ComplexF64}, b::Adjoint{ComplexF64,<:Matrix}, beta, result::Matrix{ComplexF64}) = BLAS.gemm!('N', 'C', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
gemv!(alpha, a::Adjoint{ComplexF64,<:Matrix}, b::Vector{ComplexF64}, beta, result::Vector{ComplexF64}) = BLAS.gemv!('C', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
gemv!(alpha, a::Vector{ComplexF64}, b::Adjoint{ComplexF64,<:Matrix}, beta, result::Vector{ComplexF64}) = BLAS.gemv!('T', convert(ComplexF64, alpha), b, a, convert(ComplexF64, beta), result)

gemm!(alpha, a::DenseOperator{B1,B2}, b::DenseOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = gemm!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)
gemv!(alpha, a::DenseOperator{B1,B2}, b::Ket{B2}, beta, result::Ket{B1}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)
gemv!(alpha, a::Bra{B1}, b::DenseOperator{B1,B2}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)


# Multiplication for Operators in terms of their gemv! implementation
function gemm!(alpha, M::AbstractOperator{B1,B2}, b::DenseOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    for i=1:size(b.data, 2)
        bket = Ket(b.basis_l, b.data[:,i])
        resultket = Ket(M.basis_l, result.data[:,i])
        gemv!(alpha, M, bket, beta, resultket)
        result.data[:,i] = resultket.data
    end
end

function gemm!(alpha, b::DenseOperator{B1,B2}, M::AbstractOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    for i=1:size(b.data, 1)
        bbra = Bra(b.basis_r, vec(b.data[i,:]))
        resultbra = Bra(M.basis_r, vec(result.data[i,:]))
        gemv!(alpha, bbra, M, beta, resultbra)
        result.data[i,:] = resultbra.data
    end
end

# Broadcasting
Base.size(A::DataOperator) = size(A.data)
@inline Base.axes(A::DataOperator) = axes(A.data)
Base.broadcastable(A::DataOperator) = A

# Custom broadcasting styles
abstract type DataOperatorStyle{BL<:Basis,BR<:Basis} <: Broadcast.BroadcastStyle end
struct DenseOperatorStyle{BL<:Basis,BR<:Basis} <: DataOperatorStyle{BL,BR} end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:DenseOperator{BL,BR}}) where {BL<:Basis,BR<:Basis} = DenseOperatorStyle{BL,BR}()
Broadcast.BroadcastStyle(::DenseOperatorStyle{B1,B2}, ::DenseOperatorStyle{B3,B4}) where {B1<:Basis,B2<:Basis,B3<:Basis,B4<:Basis} = throw(IncompatibleBases())

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL<:Basis,BR<:Basis,Style<:DenseOperatorStyle{BL,BR},Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    bl,br = find_basis(bcf.args)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    # TODO: remove convert
    return Operator{BL,BR}(bl, br, convert(Matrix{ComplexF64}, copy(bc_)))
end
find_basis(a::DataOperator, rest) = (a.basis_l, a.basis_r)

const BasicMathFunc = Union{typeof(+),typeof(-),typeof(*)}
function Broadcasted_restrict_f(f::BasicMathFunc, args::Tuple{Vararg{<:DataOperator}}, axes)
    args_ = Tuple(a.data for a=args)
    return Broadcast.Broadcasted(f, args_, axes)
end
function Broadcasted_restrict_f(f, args::Tuple{Vararg{<:DataOperator}}, axes)
    throw(error("Cannot broadcast function `$f` on type `$(eltype(args))`"))
end

# In-place broadcasting
@inline function Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL<:Basis,BR<:Basis,Style<:DataOperatorStyle{BL,BR},Axes,F,Args}
    axes(dest) == axes(bc) || Base.Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && isa(bc.args, Tuple{<:DataOperator{BL,BR}}) # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    # Get the underlying data fields of operators and broadcast them as arrays
    bcf = Broadcast.flatten(bc)
    bc_ = Broadcasted_restrict_f(bcf.f, bcf.args, axes(bcf))
    copyto!(dest.data, bc_)
    return dest
end
@inline Base.copyto!(A::DataOperator{BL,BR},B::DataOperator{BL,BR}) where {BL<:Basis,BR<:Basis} = (copyto!(A.data,B.data); A)
# @inline Base.copyto!(dest::DataOperator{BL,BR}, bc::Broadcast.Broadcasted{Style,Axes,F,Args}) where {BL<:Basis,BR<:Basis,Style<:DataOperatorStyle,Axes,F,Args} =
#     throw(IncompatibleBases())
