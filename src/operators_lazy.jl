import LazyArrays

const AddFunc = Union{typeof(+),typeof(-)}
const LazyOperatorType{BL,BR} = Operator{BL,BR,<:LazyArrays.LazyArray}

const ApplyArrayOperator{BL,BR} = Union{LazyOperatorType{BL,BR},MatrixOperator{BL,BR}}

function LazyOperator(f::AddFunc,A::ApplyArrayOperator{BL,BR},B::ApplyArrayOperator{BL,BR}) where {BL,BR}
    data = LazyArrays.ApplyArray(f,A.data,B.data)
    return Operator(A.basis_l, A.basis_r, data)
end
function LazyOperator(f::AddFunc,ops::Tuple{Vararg{<:ApplyArrayOperator{BL,BR}}}) where {BL,BR}
    data = LazyArrays.ApplyArray(f,(o.data for o=ops)...)
    return Operator(ops[1].basis_l, ops[1].basis_r, data)
end

function LazyOperator(f::typeof(*),A::ApplyArrayOperator{B1,B2},B::ApplyArrayOperator{B2,B3}) where {B1,B2,B3}
    data = LazyArrays.ApplyArray(f,A.data,B.data)
    return Operator(A.basis_l,B.basis_r,data)
end
function LazyOperator(f::typeof(*),ops::Tuple{Vararg{<:ApplyArrayOperator}})
    for i=1:length(ops)-1
        check_multiplicable(ops[i],ops[i+1])
    end
    data = LazyArrays.ApplyArray(f,(o.data for o=ops)...)
    return Operator(ops[1].basis_l,ops[end].basis_r,data)
end

function LazyOperator(f::typeof(tensor),A::ApplyArrayOperator,B::ApplyArrayOperator)
    data = LazyArrays.ApplyArray(kron,B.data,A.data)
    basis_l = tensor(A.basis_l,B.basis_l)
    basis_r = tensor(A.basis_r,B.basis_r)
    return Operator(basis_l,basis_r,data)
end
function LazyOperator(f::typeof(tensor),ops::Tuple{Vararg{<:ApplyArrayOperator}})
    data = LazyArrays.ApplyArray(kron,(o.data for o=reverse(ops))...)
    basis_l = tensor((o.basis_l for o=ops)...)
    basis_r = tensor((o.basis_r for o=ops)...)
    return Operator(basis_l,basis_r,data)
end

# LazyOperator(f,ops...) = LazyOperator(f,ops)

# Old syntax
LazyProduct(args...) = LazyOperator(*,args...)
LazySum(args...) = LazyOperator(+,args...)
LazyTensor(args...) = LazyOperator(tensor,args...)


# gemm!(alpha, a::DenseOperator{B1,B2}, b::DenseOperator{B2,B3}, beta, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = gemm!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)
function gemv!(alpha::Number, a::LazyOperatorType{B1,B2}, b::Ket{B2}, beta::Number, result::Ket{B1}) where {B1<:Basis,B2<:Basis}
    # result.data .= LazyArrays.@~ alpha*a.data*b.data + beta*result.data
    LazyArrays.materialize!(lazy_gemv!(alpha,a,b,beta,result))
    return nothing
end
function lazy_gemv!(alpha::Number, a::LazyOperatorType{B1,B2}, b::Ket{B2}, beta::Number, result::Ket{B1}) where {B1,B2}
    return LazyArrays.MulAdd(alpha,a.data,b.data,beta,result.data)
end
# gemv!(alpha, a::Bra{B1}, b::DenseOperator{B1,B2}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis} = gemv!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)

# function gemv!(alpha, A::LazyArrays.LazyArray, b::Vector, beta, result::Vector)
#     result .= LazyArrays.@~ alpha*A*b + beta*result
# end
