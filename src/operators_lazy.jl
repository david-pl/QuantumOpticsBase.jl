import LazyArrays

const AddFunc = Union{typeof(+),typeof(-)}
const LazyOperator{BL,BR} = Operator{BL,BR,<:LazyArrays.LazyArray}

function ApplyOperator(f::AddFunc,A::Operator{BL,BR},B::Operator{BL,BR}) where {BL,BR}
    data = LazyArrays.ApplyArray(f,A.data,B.data)
    return Operator(A.basis_l, A.basis_r, data)
end
function ApplyOperator(f::typeof(*),A::Operator{B1,B2},B::Operator{B2,B3}) where {B1,B2,B3}
    data = LazyArrays.ApplyArray(f,A.data,B.data)
    return Operator(A.basis_l,B.basis_r,data)
end
function ApplyOperator(f::typeof(tensor),A::Operator,B::Operator)
    data = LazyArrays.ApplyArray(kron,B.data,A.data)
    basis_l = tensor(A.basis_l,B.basis_l)
    basis_r = tensor(A.basis_r,B.basis_r)
    return Operator(basis_l,basis_r,data)
end


# gemm!(alpha, a::DenseOperator{B1,B2}, b::DenseOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = gemm!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)
function gemv!(alpha::Number, a::LazyOperator{B1,B2}, b::Ket{B2}, beta::Number, result::Ket{B1}) where {B1<:Basis,B2<:Basis}
    # result.data .= LazyArrays.@~ alpha*a.data*b.data + beta*result.data
    LazyArrays.materialize!(LazyArrays.MulAdd(alpha,a.data,b.data,beta,result.data))
    return nothing
end
function lazy_gemv!(alpha::Number, a::LazyOperator{B1,B2}, b::Ket{B2}, beta::Number, result::Ket{B1}) where {B1,B2}
    return LazyArrays.MulAdd(alpha,a.data,b.data,beta,result.data)
end
# gemv!(alpha, a::Bra{B1}, b::DenseOperator{B1,B2}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)

# function gemv!(alpha, A::LazyArrays.LazyArray, b::Vector, beta, result::Vector)
#     result .= LazyArrays.@~ alpha*A*b + beta*result
# end
