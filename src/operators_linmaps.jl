import LinearMaps
using LinearMaps: LinearMap

const LinMapOp{B1,B2} = Operator{B1,B2,<:LinearMap}
dense(op::LinMapOp) = op*Operator(op.basis_r, op.basis_r, Matrix{ComplexF64}(I, length(op.basis_r), length(op.basis_r)))
dagger(op::LinMapOp) = transform(op.basis_r, op.basis_l)

function gemv!(alpha, M::Operator{B1,B2,<:LinearMap}, b::Ket{B2}, beta, result::Ket{B1}) where {B1<:Basis,B2<:Basis}
    LinearMaps.mul!(result.data,M.data,b.data,alpha,beta)
    return nothing
end

function gemv!(alpha, b::Bra{B1}, M::Operator{B1,B2,<:LinearMap}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis}
    if beta==0
        M.data.fc(result.data,b.data)
        rmul!(result.data,alpha)
    else
        psi_ = Bra(M.basis_r,b.data)
        M.data.fc(psi_.data,b.data)
        rmul!(psi_.data,alpha)
        rmul!(result.data,beta)
        result.data .+= psi_.data
    end
    return nothing
end

function gemm!(alpha, A::Operator{B1,B2,<:LinearMap}, B::Operator{B2,B3}, beta, result::Operator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    LinearMaps.mul!(result.data,A.data,B.data,alpha,beta)
    nothing
end

# Multiplication for Operators in terms of their gemv! implementation
# TODO: better Matrix-LinearMap multiplication here
function gemm!(alpha, b::Operator{B1,B2}, M::LinMapOp{B2,B3}, beta, result::Operator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    @views @inbounds for i=1:size(b.data, 1)
        bbra = Bra(b.basis_r, vec(b.data[i,:]))
        resultbra = Bra(M.basis_r, vec(result.data[i,:]))
        gemv!(alpha, bbra, M, beta, resultbra)
        result.data[i,:] = resultbra.data
    end
end


function *(M::Operator{B1,B2,<:LinearMap},b::Ket{B2}) where {B1,B2}
    result = Ket(M.basis_l)
    gemv!(1.0,M,b,0.0,result)
    return result
end
function *(b::Bra{B1},M::Operator{B1,B2,<:LinearMap}) where {B1,B2}
    result = Bra(M.basis_r)
    gemv!(1.0,b,M,0.0,result)
    return result
end

function *(M::Operator{B1,B2,<:LinearMap},b::Operator{B2,B3,T}) where {B1,B2,B3,T}
    result = Operator{B1,B3,T}(M.basis_l,b.basis_r,similar(b.data))
    gemm!(1.0,M,b,0.0,result)
    return result
end
function *(b::Operator{B3,B1,T}, M::Operator{B1,B2,<:LinearMap}) where {B1,B2,B3,T}
    result = Operator{B3,B2,T}(b.basis_l,M.basis_r,similar(b.data))
    gemm!(1.0,b,M,0.0,result)
    return result
end

# TODO: Replace usage of LazyArrays altogether?
function LazyOperator(f::typeof(+),a::Operator{B1,B2},b::LinMapOp{B1,B2}) where {B1,B2}
    vec_ket = Ket(b.basis_r)
    result_ket = Ket(b.basis_l)
    function func_r(result,vec)
        copyto!(vec_ket.data,vec)
        # b*vec
        gemv!(1.0,b,vec_ket,0.0,result_ket)
        # a*vec + b*vec
        gemv!(1.0,a,vec_ket,1.0,result_ket)
        copyto!(result,result_ket.data)
    end

    vec_bra = Bra(b.basis_l)
    result_bra = Bra(b.basis_r)
    function func_l(result,vec)
        copyto!(vec_bra.data,vec)
        # vec'*b
        gemv!(1.0,vec_bra,b,0.0,result_bra)
        # vec'*a + vec'*b
        gemv!(1.0,vec_bra,a,1.0,result_bra)
        copyto!(result,result_bra.data)
    end

    dtype = promote_type(eltype(a.data),eltype(b.data))
    data = LinearMap{dtype}(func_r, func_l, length(b.basis_l), length(b.basis_r); ismutating=true)
    return Operator(b.basis_l, b.basis_r, data)
end

function LazyOperator(f::typeof(-),a::Operator{B1,B2},b::LinMapOp{B1,B2}) where {B1,B2}
    vec_ket = Ket(b.basis_r)
    result_ket = Ket(b.basis_l)
    function func_r(result,vec)
        copyto!(vec_ket.data,vec)
        # b*vec
        gemv!(-1.0,b,vec_ket,0.0,result_ket)
        # a*vec + b*vec
        gemv!(1.0,a,vec_ket,1.0,result_ket)
        copyto!(result,result_ket.data)
    end

    vec_bra = Bra(b.basis_l)
    result_bra = Bra(b.basis_r)
    function func_l(result,vec)
        copyto!(vec_bra.data,vec)
        # vec'*b
        gemv!(-1.0,vec_bra,b,0.0,result_bra)
        # vec'*a + vec'*b
        gemv!(1.0,vec_bra,a,1.0,result_bra)
        copyto!(result,result_bra.data)
    end

    dtype = promote_type(eltype(a.data),eltype(b.data))
    data = LinearMap{dtype}(func_r, func_l, length(b.basis_l), length(b.basis_r); ismutating=true)
    return Operator(b.basis_l, b.basis_r, data)
end

function LazyOperator(f::typeof(*),a::Operator{B1,B2},b::LinMapOp{B2,B3}) where {B1,B2,B3}
    vec_ket = Ket(b.basis_r)
    result_ket = Ket(a.basis_l)
    function func_r(result,vec)
        copyto!(vec_ket.data,vec)
        # b*vec
        gemv!(1.0,b,vec_ket,0.0,result_ket)
        # a*vec + b*vec
        gemv!(1.0,a,result_ket,0.0,result_ket)
        copyto!(result,result_ket.data)
    end

    vec_bra = Bra(b.basis_l)
    result_bra = Bra(b.basis_r)
    function func_l(result,vec)
        copyto!(vec_bra.data,vec)
        # vec'*b
        gemv!(1.0,vec_bra,b,0.0,result_bra)
        # vec'*a + vec'*b
        gemv!(1.0,result_bra,a,0.0,result_bra)
        copyto!(result,result_bra.data)
    end

    dtype = promote_type(eltype(a.data),eltype(b.data))
    data = LinearMap{dtype}(func_r, func_l, length(a.basis_l), length(b.basis_r); ismutating=true)
    return Operator(b.basis_l, b.basis_r, data)
end
