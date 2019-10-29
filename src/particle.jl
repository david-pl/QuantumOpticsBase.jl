import Base: position
using FFTW
import LinearMaps
using LinearMaps: LinearMap

"""
    PositionBasis(xmin, xmax, Npoints)
    PositionBasis(b::MomentumBasis)

Basis for a particle in real space.

For simplicity periodic boundaries are assumed which means that
the rightmost point defined by `xmax` is not included in the basis
but is defined to be the same as `xmin`.

When a [`MomentumBasis`](@ref) is given as argument the exact values
of ``x_{min}`` and ``x_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dp`` and ``\\pi/dp`` with ``dp=(p_{max}-p_{min})/N``.
"""
struct PositionBasis{T,X1,X2} <: Basis
    shape::Vector{T}
    xmin::Float64
    xmax::Float64
    N::T
    function PositionBasis{X1,X2}(xmin::Real, xmax::Real, N::T) where {X1,X2,T<:Int}
        @assert isa(X1, Real) && isa(X2, Real)
        new{T,X1,X2}([N], xmin, xmax, N)
    end
end
PositionBasis(xmin::Real, xmax::Real, N::Int) = PositionBasis{xmin,xmax}(xmin,xmax,N)

"""
    MomentumBasis(pmin, pmax, Npoints)
    MomentumBasis(b::PositionBasis)

Basis for a particle in momentum space.

For simplicity periodic boundaries are assumed which means that
`pmax` is not included in the basis but is defined to be the same as `pmin`.

When a [`PositionBasis`](@ref) is given as argument the exact values
of ``p_{min}`` and ``p_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dx`` and ``\\pi/dx`` with ``dx=(x_{max}-x_{min})/N``.
"""
struct MomentumBasis{P1,P2} <: Basis
    shape::Vector{Int}
    pmin::Float64
    pmax::Float64
    N::Int
    function MomentumBasis{P1,P2}(pmin::Real, pmax::Real, N::Int) where {P1,P2}
        @assert isa(P1, Real) && isa(P2, Real)
        new([N], pmin, pmax, N)
    end
end
MomentumBasis(pmin::Real, pmax::Real, N::Int) = MomentumBasis{pmin,pmax}(pmin, pmax, N)

PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))

==(b1::PositionBasis, b2::PositionBasis) = b1.xmin==b2.xmin && b1.xmax==b2.xmax && b1.N==b2.N
==(b1::MomentumBasis, b2::MomentumBasis) = b1.pmin==b2.pmin && b1.pmax==b2.pmax && b1.N==b2.N


"""
    gaussianstate(b::PositionBasis, x0, p0, sigma)
    gaussianstate(b::MomentumBasis, x0, p0, sigma)

Create a Gaussian state around `x0` and` p0` with width `sigma`.

In real space the gaussian state is defined as

```math
\\Psi(x) = \\frac{1}{\\pi^{1/4}\\sqrt{\\sigma}}
            e^{i p_0 (x-\\frac{x_0}{2}) - \\frac{(x-x_0)^2}{2 \\sigma^2}}
```

and is connected to the momentum space definition

```math
\\Psi(p) = \\frac{\\sqrt{\\sigma}}{\\pi^{1/4}}
            e^{-i x_0 (p-\\frac{p_0}{2}) - \\frac{1}{2}(p-p_0)^2 \\sigma^2}
```

via a Fourier-transformation

```math
\\Psi(p) = \\frac{1}{\\sqrt{2\\pi}}
            \\int_{-\\infty}^{\\infty} e^{-ipx}\\Psi(x) \\mathrm{d}x
```

The state has the properties

* ``⟨p⟩ = p_0``
* ``⟨x⟩ = x_0``
* ``\\mathrm{Var}(x) = \\frac{σ^2}{2}``
* ``\\mathrm{Var}(p) = \\frac{1}{2 σ^2}``

Due to the numerically necessary discretization additional scaling
factors ``\\sqrt{Δx}`` and ``\\sqrt{Δp}`` are used so that
``\\langle x_i|Ψ\\rangle = \\sqrt{Δ x} Ψ(x_i)`` and ``\\langle p_i|Ψ\\rangle = \\sqrt{Δ p} Ψ(p_i)`` so
that the resulting Ket state is normalized.
"""
function gaussianstate(b::PositionBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dx = spacing(b)
    alpha = 1.0/(pi^(1/4)*sqrt(sigma))*sqrt(dx)
    x = b.xmin
    for i=1:b.N
        psi.data[i] = alpha*exp(1im*p0*(x-x0/2) - (x-x0)^2/(2*sigma^2))
        x += dx
    end
    return psi
end

function gaussianstate(b::MomentumBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dp = spacing(b)
    alpha = sqrt(sigma)/pi^(1/4)*sqrt(dp)
    p = b.pmin
    for i=1:b.N
        psi.data[i] = alpha*exp(-1im*x0*(p-p0/2) - (p-p0)^2/2*sigma^2)
        p += dp
    end
    return psi
end


"""
    spacing(b::PositionBasis)

Difference between two adjacent points of the real space basis.
"""
spacing(b::PositionBasis) = (b.xmax - b.xmin)/b.N
"""
    spacing(b::MomentumBasis)

Momentum difference between two adjacent points of the momentum basis.
"""
spacing(b::MomentumBasis) = (b.pmax - b.pmin)/b.N

"""
    samplepoints(b::PositionBasis)

x values of the real space basis.
"""
samplepoints(b::PositionBasis) = (dx = spacing(b); Float64[b.xmin + i*dx for i=0:b.N-1])
"""
    samplepoints(b::MomentumBasis)

p values of the momentum basis.
"""
samplepoints(b::MomentumBasis) = (dp = spacing(b); Float64[b.pmin + i*dp for i=0:b.N-1])

"""
    position(b::PositionBasis)

Position operator in real space.
"""
position(b::PositionBasis) = Operator(b, sparse(Diagonal(complex(samplepoints(b)))))


"""
    position(b:MomentumBasis)

Position operator in momentum space.
"""
function position(b::MomentumBasis)
    b_pos = PositionBasis(b)
    transform(b, b_pos)*dense(position(b_pos))*transform(b_pos, b)
end

"""
    momentum(b:MomentumBasis)

Momentum operator in momentum space.
"""
momentum(b::MomentumBasis) = Operator(b, sparse(Diagonal(complex(samplepoints(b)))))

"""
    momentum(b::PositionBasis)

Momentum operator in real space.
"""
function momentum(b::PositionBasis)
    b_mom = MomentumBasis(b)
    transform(b, b_mom)*dense(momentum(b_mom))*transform(b_mom, b)
end

"""
    potentialoperator(b::PositionBasis, V(x))

Operator representing a potential ``V(x)`` in real space.
"""
function potentialoperator(b::PositionBasis, V::Function)
    x = samplepoints(b)
    diagonaloperator(b, V.(x))
end

"""
    potentialoperator(b::MomentumBasis, V(x))

Operator representing a potential ``V(x)`` in momentum space.
"""
function potentialoperator(b::MomentumBasis, V::Function)
    b_pos = PositionBasis(b)
    transform(b, b_pos)*dense(potentialoperator(b_pos, V))*transform(b_pos, b)
end

"""
    potentialoperator(b::CompositeBasis, V(x, y, z, ...))

Operator representing a potential ``V`` in more than one dimension.

# Arguments
* `b`: Composite basis consisting purely either of `PositionBasis` or
    `MomentumBasis`. Note, that calling this with a composite basis in
    momentum space might consume a large amount of memory.
* `V`: Function describing the potential. ATTENTION: The number of arguments
    accepted by `V` must match the spatial dimension. Furthermore, the order
    of the arguments has to match that of the order of the tensor product of
    bases (e.g. if `b=bx⊗by⊗bz`, then `V(x,y,z)`).
"""
function potentialoperator(b::CompositeBasis, V::Function)
    if isa(b.bases[1], PositionBasis)
        potentialoperator_position(b, V)
    elseif isa(b.bases[1], MomentumBasis)
        potentialoperator_momentum(b, V)
    else
        throw(IncompatibleBases())
    end
end
function potentialoperator_position(b::CompositeBasis, V::Function)
    for base=b.bases
        @assert isa(base, PositionBasis)
    end

    points = [samplepoints(b1) for b1=b.bases]
    dims = length.(points)
    n = length(b.bases)
    data = Array{ComplexF64}(undef, dims...)
    @inbounds for i=1:length(data)
        index = Tuple(CartesianIndices(data)[i])
        args = (points[j][index[j]] for j=1:n)
        data[i] = V(args...)
    end

    diagonaloperator(b, data[:])
end
function potentialoperator_momentum(b::CompositeBasis, V::Function)
    bases_pos = []
    for base=b.bases
        @assert isa(base, MomentumBasis)
        push!(bases_pos, PositionBasis(base))
    end
    b_pos = tensor(bases_pos...)
    transform(b, b_pos)*dense(potentialoperator_position(b_pos, V))*transform(b_pos, b)
end

# """
#     FFTOperator
#
# Abstract type for all implementations of FFT operators.
# """
# abstract type FFTOperator{BL<:Basis, BR<:Basis, T} <: AbstractOperator{BL,BR} end
#
# """
#     FFTOperators
#
# Operator performing a fast fourier transformation when multiplied with a state
# that is a Ket or an Operator.
# """
# mutable struct FFTOperators{BL<:Basis,BR<:Basis,T<:Array{ComplexF64},P1,P2,P3,P4} <: FFTOperator{BL, BR, T}
#     basis_l::BL
#     basis_r::BR
#     fft_l!::P1
#     fft_r!::P2
#     fft_l2!::P3
#     fft_r2!::P4
#     mul_before::T
#     mul_after::T
#     function FFTOperators(b1::BL, b2::BR,
#         fft_l!::P1,
#         fft_r!::P2,
#         fft_l2!::P3,
#         fft_r2!::P4,
#         mul_before::T,
#         mul_after::T) where {BL<:Basis,BR<:Basis,T,P1,P2,P3,P4}
#         new{BL,BR,T,P1,P2,P3,P4}(b1, b2, fft_l!, fft_r!, fft_l2!, fft_r2!, mul_before, mul_after)
#     end
# end
#
# """
#     FFTKets
#
# Operator that can only perform fast fourier transformations on Kets.
# This is much more memory efficient when only working with Kets.
# """
# mutable struct FFTKets{BL<:Basis,BR<:Basis,T<:Array{ComplexF64},P1,P2} <: FFTOperator{BL, BR, T}
#     basis_l::BL
#     basis_r::BR
#     fft_l!::P1
#     fft_r!::P2
#     mul_before::T
#     mul_after::T
#     function FFTKets(b1::BL, b2::BR,
#         fft_l!::P1,
#         fft_r!::P2,
#         mul_before::T,
#         mul_after::T) where {BL<:Basis,BR<:Basis, T, P1, P2}
#         new{BL, BR, T, P1, P2}(b1, b2, fft_l!, fft_r!, mul_before, mul_after)
#     end
# end

"""
    transform(b1::MomentumBasis, b2::PositionBasis)
    transform(b1::PositionBasis, b2::MomentumBasis)

Transformation operator between position basis and momentum basis.
"""
function transform(basis_l::MomentumBasis, basis_r::PositionBasis; ket_only::Bool=false)
    Lx = (basis_r.xmax - basis_r.xmin)
    dp = spacing(basis_l)
    dx = spacing(basis_r)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    mul_before = exp.(-1im*basis_l.pmin*(samplepoints(basis_r) .- basis_r.xmin))
    mul_after = exp.(-1im*basis_r.xmin*samplepoints(basis_l))/sqrt(basis_r.N)

    N::Int = length(basis_r)
    x = Vector{ComplexF64}(undef, N)
    fft_r! = plan_fft!(x)
    function _fft_mul!(result,vec)
        @inbounds for i=1:N
            result[i] = mul_before[i] * vec[i]
        end
        # TODO: remove unnecessary reshaping
        fft_r! * reshape(result, size(mul_before))
        @inbounds for i=1:N
            result[i] *= mul_after[i]
        end
        return result
    end

    M::Int = length(basis_l)
    y = Vector{ComplexF64}(undef, M)
    fft_l! = plan_bfft!(y)
    function _bfft_mul!(result,vec)
        @inbounds for i=1:M
            result[i] = conj(mul_after[i]) * conj(vec[i])
        end
        # TODO: remove unnecessary reshaping
        fft_l! * reshape(result, size(mul_after))
        @inbounds for i=1:M
            result[i] = conj(result[i]) * mul_before[i]
        end
        return result
    end

    data = LinearMap{ComplexF64}(_fft_mul!, _bfft_mul!, N, M; ismutating=true)
    return Operator(basis_l,basis_r,data)
end

"""
    transform(b1::CompositeBasis, b2::CompositeBasis)

Transformation operator between two composite bases. Each of the bases
has to contain bases of type PositionBasis and the other one a corresponding
MomentumBasis.
"""
function transform(basis_l::PositionBasis, basis_r::MomentumBasis; ket_only::Bool=false)
    Lx = (basis_l.xmax - basis_l.xmin)
    dp = spacing(basis_r)
    dx = spacing(basis_l)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    mul_before = exp.(1im*basis_l.xmin*(samplepoints(basis_r) .- basis_r.pmin))
    mul_after = exp.(1im*basis_r.pmin*samplepoints(basis_l))/sqrt(basis_r.N)

    N::Int = length(basis_r)
    x = Vector{ComplexF64}(undef, N)
    fft_r! = plan_bfft!(x)
    function _fft_mul!(result,vec)
        @inbounds for i=1:N
            result[i] = mul_before[i] * vec[i]
        end
        # TODO: remove unnecessary reshaping
        fft_r! * reshape(result, size(mul_before))
        @inbounds for i=1:N
            result[i] *= mul_after[i]
        end
        return result
    end

    M::Int = length(basis_l)
    y = Vector{ComplexF64}(undef, M)
    fft_l! = plan_fft!(y)
    function _bfft_mul!(result,vec)
        @inbounds for i=1:M
            result[i] = conj(mul_after[i]) * conj(vec[i])
        end
        # TODO: remove unnecessary reshaping
        fft_l! * reshape(result, size(mul_after))
        @inbounds for i=1:M
            result[i] = conj(result[i]) * mul_before[i]
        end
        return result
    end

    data = LinearMap{ComplexF64}(_fft_mul!, _bfft_mul!, N, M; ismutating=true)
    return Operator(basis_l,basis_r,data)
end

# function transform(basis_l::CompositeBasis, basis_r::CompositeBasis; ket_only::Bool=false, index::Vector{Int}=Int[])
#     @assert length(basis_l.bases) == length(basis_r.bases)
#     if length(index) == 0
#         check_pos = [isa.(basis_l.bases, PositionBasis)...]
#         check_mom = [isa.(basis_l.bases, MomentumBasis)...]
#         if any(check_pos) && !any(check_mom)
#             index = [1:length(basis_l.bases);][check_pos]
#         elseif any(check_mom) && !any(check_pos)
#             index = [1:length(basis_l.bases);][check_mom]
#         else
#             throw(IncompatibleBases())
#         end
#     end
#     if all(isa.(basis_l.bases[index], PositionBasis))
#         @assert all(isa.(basis_r.bases[index], MomentumBasis))
#         transform_xp(basis_l, basis_r, index; ket_only=ket_only)
#     elseif all(isa.(basis_l.bases[index], MomentumBasis))
#         @assert all(isa.(basis_r.bases[index], PositionBasis))
#         transform_px(basis_l, basis_r, index; ket_only=ket_only)
#     else
#         throw(IncompatibleBases())
#     end
# end
#
# function transform_xp(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Vector{Int}; ket_only::Bool=false)
#     n = length(basis_l.bases)
#     Lx = [(b.xmax - b.xmin) for b=basis_l.bases[index]]
#     dp = [spacing(b) for b=basis_r.bases[index]]
#     dx = [spacing(b) for b=basis_l.bases[index]]
#     N = [length(b) for b=basis_l.bases]
#     for i=1:n
#         if N[i] != length(basis_r.bases[i])
#             throw(IncompatibleBases())
#         end
#     end
#     for i=1:length(index)
#         if abs(2*pi/dp[i] - Lx[i])/Lx[i] > 1e-12
#             throw(IncompatibleBases())
#         end
#     end
#
#     if index[1] == 1
#         mul_before = exp.(1im*basis_l.bases[1].xmin*(samplepoints(basis_r.bases[1]) .- basis_r.bases[1].pmin))
#         mul_after = exp.(1im*basis_r.bases[1].pmin*samplepoints(basis_l.bases[1]))/sqrt(basis_r.bases[1].N)
#     else
#         mul_before = ones(N[1])
#         mul_after = ones(N[1])
#     end
#     for i=2:n
#         if any(i .== index)
#             mul_before = kron(exp.(1im*basis_l.bases[i].xmin*(samplepoints(basis_r.bases[i]) .- basis_r.bases[i].pmin)), mul_before)
#             mul_after = kron(exp.(1im*basis_r.bases[i].pmin*samplepoints(basis_l.bases[i]))/sqrt(basis_r.bases[i].N), mul_after)
#         else
#             mul_before = kron(ones(N[i]), mul_before)
#             mul_after = kron(ones(N[i]), mul_after)
#         end
#     end
#     mul_before = reshape(mul_before, (N...,))
#     mul_after = reshape(mul_after, (N...,))
#
#     x = Array{ComplexF64}(undef, N...)
#     if ket_only
#         FFTKets(basis_l, basis_r, plan_fft!(x, index), plan_bfft!(x, index), mul_before, mul_after)
#     else
#         A = Array{ComplexF64}(undef, [N; N]...)
#         FFTOperators(basis_l, basis_r, plan_fft!(x, index), plan_bfft!(x, index), plan_fft!(A, [n + 1:2n;][index]), plan_bfft!(A, [1:n;][index]), mul_before, mul_after)
#     end
# end
#
# function transform_px(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Vector{Int}; ket_only::Bool=false)
#     n = length(basis_l.bases)
#     Lx = [(b.xmax - b.xmin) for b=basis_r.bases[index]]
#     dp = [spacing(b) for b=basis_l.bases[index]]
#     dx = [spacing(b) for b=basis_r.bases[index]]
#     N = [length(b) for b=basis_l.bases]
#     for i=1:n
#         if N[i] != length(basis_r.bases[i])
#             throw(IncompatibleBases())
#         end
#     end
#     for i=1:length(index)
#         if abs(2*pi/dp[i] - Lx[i])/Lx[i] > 1e-12
#             throw(IncompatibleBases())
#         end
#     end
#
#     if index[1] == 1
#         mul_before = exp.(-1im*basis_l.bases[1].pmin*(samplepoints(basis_r.bases[1]) .- basis_r.bases[1].xmin))
#         mul_after = exp.(-1im*basis_r.bases[1].xmin*samplepoints(basis_l.bases[1]))/sqrt(N[1])
#     else
#         mul_before = ones(N[1])
#         mul_after = ones(N[1])
#     end
#     for i=2:n
#         if i in index
#             mul_before = kron(exp.(-1im*basis_l.bases[i].pmin*(samplepoints(basis_r.bases[i]) .- basis_r.bases[i].xmin)), mul_before)
#             mul_after = kron(exp.(-1im*basis_r.bases[i].xmin*samplepoints(basis_l.bases[i]))/sqrt(N[i]), mul_after)
#         else
#             mul_before = kron(ones(N[i]), mul_before)
#             mul_after = kron(ones(N[i]), mul_after)
#         end
#     end
#     mul_before = reshape(mul_before, (N...,))
#     mul_after = reshape(mul_after, (N...,))
#
#     x = Array{ComplexF64}(undef, N...)
#     if ket_only
#         FFTKets(basis_l, basis_r, plan_bfft!(x, index), plan_fft!(x, index), mul_before, mul_after)
#     else
#         A = Array{ComplexF64}(undef, [N; N]...)
#         FFTOperators(basis_l, basis_r, plan_bfft!(x, index), plan_fft!(x, index), plan_bfft!(A, [n + 1:2n;][index]), plan_fft!(A, [1:n;][index]), mul_before, mul_after)
#     end
# end

const LinMapOp{B1,B2} = Operator{B1,B2,<:LinearMap}
dense(op::LinMapOp) = op*Operator(op.basis_r, op.basis_r, Matrix{ComplexF64}(I, length(op.basis_r), length(op.basis_r)))

dagger(op::LinMapOp) = transform(op.basis_r, op.basis_l)

# tensor(A::FFTOperators, B::FFTOperators) = transform(tensor(A.basis_l, B.basis_l), tensor(A.basis_r, B.basis_r))
# tensor(A::FFTKets, B::FFTKets) = transform(tensor(A.basis_l, B.basis_l), tensor(A.basis_r, B.basis_r); ket_only=true)

function gemv!(alpha, M::Operator{B1,B2,<:LinearMap}, b::Ket{B2}, beta, result::Ket{B1}) where {B1<:Basis,B2<:Basis}
    # if beta==0
    #     M.data.f(result.data,b.data)
    #     rmul!(result.data,alpha)
    # else
    #     psi_ = Ket(M.basis_l, copy(b.data))
    #     M.data.f(psi_.data,b.data)
    #     rmul!(psi_.data, alpha)
    #     rmul!(result.data, beta)
    #     result.data .+= psi_.data
    # end
    LinearMaps.mul!(result.data,M.data,b.data,alpha,beta)
    return nothing
end

function gemv!(alpha, b::Bra{B1}, M::Operator{B1,B2,<:LinearMap}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis}
    # TODO: better handling of adjoint
    LinearMaps.mul!(result.data,M.data',b.data,alpha,beta)
    # conj!(b.data)
    # if beta==0
    #     M.data.fc(result.data,b.data)
    #     rmul!(result.data,alpha)
    # else
    #     psi_ = Bra(M.basis_r, copy(b.data))
    #     M.data.fc(psi_.data,b.data)
    #     rmul!(psi_.data,alpha)
    #     rmul!(result.data,beta)
    #     result.data .+= psi_.data
    # end
    return nothing
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

# function gemv!(alpha_, b::Bra{B1}, M::FFTOperator{B1,B2}, beta_, result::Bra{B2}) where {B1<:Basis,B2<:Basis}
#     alpha = convert(ComplexF64, alpha_)
#     beta = convert(ComplexF64, beta_)
#     N::Int = length(M.basis_l)
    # if beta==Complex(0.)
    #     @inbounds for i=1:N
    #         result.data[i] = conj(M.mul_after[i]) * conj(b.data[i])
    #     end
    #     M.fft_l! * reshape(result.data, size(M.mul_after))
    #     @inbounds for i=1:N
    #         result.data[i] = conj(result.data[i]) * M.mul_before[i] * alpha
    #     end
    # else
    #     psi_ = Bra(M.basis_r, conj(b.data))
    #     @inbounds for i=1:N
    #         psi_.data[i] *= conj(M.mul_after[i])
    #     end
    #     M.fft_l! * reshape(psi_.data, size(M.mul_after))
    #     @inbounds for i=1:N
    #         result.data[i] = beta*result.data[i] + alpha * conj(psi_.data[i]) * M.mul_before[i]
    #     end
    # end
#     nothing
# end

# function gemm!(alpha, A::Operator{B1,B2}, B::Operator{B2,B3,<:LinearMap}, beta, result::Operator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
#     # LinearMaps.mul!(result.data,B.data',transpose(A.data),alpha,beta)
#     # if beta==0
#     #     A_ = adjoint(A)
#     #     result_ = adjoint(result)
#     #     @inbounds @views for k=1:length(A.basis_l)
#     #         B.data.fc(result_.data[:,k],A_.data[:,k])
#     #     end
#     #     # conj!(result.data)
#     #     rmul!(result.data, alpha)
#     # else
#     #     op_ = Operator(result.basis_l,result.basis_r,similar(result.data))
#     #     @inbounds @views for k=1:length(A.basis_l)
#     #         B.data.fc(op_.data[:,k],A.data[:,k])
#     #     end
#     #     conj!(op_.data)
#     #     rmul!(op_.data, alpha)
#     #     rmul!(result.data, beta)
#     #     result.data .+= op_.data
#     # end
#
#     # result.data .= transpose(result.data)
#     # result.data .= adjoint(result.data)
#     nothing
# end
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


# function gemm!(alpha_, A::DenseOperator{B1,B2}, B::FFTOperators{B2,B3}, beta_, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
#     alpha = convert(ComplexF64, alpha_)
#     beta = convert(ComplexF64, beta_)
#     if beta != Complex(0.)
#         data = Matrix{ComplexF64}(undef, size(result.data, 1), size(result.data, 2))
#     else
#         data = result.data
#     end
#     copyto!(data, A.data)
#     @inbounds for j=1:length(B.mul_after), i=1:length(B.mul_after)
#         data[i, j] *= B.mul_after[j]
#     end
#     conj!(data)
#     n = size(B.mul_after)
#     B.fft_l2! * reshape(data, n..., n...)
#     conj!(data)
#     N = prod(n)
#     @inbounds for j=1:N, i=1:N
#         data[i, j] *= B.mul_before[j]
#     end
#     if alpha != Complex(1.)
#         lmul!(alpha, data)
#     end
#     if beta != Complex(0.)
#         rmul!(result.data, beta)
#         result.data += data
#     end
#     nothing
# end
#
# function gemm!(alpha_, A::FFTOperators{B1,B2}, B::DenseOperator{B2,B3}, beta_, result::DenseOperator{B1,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis}
#     alpha = convert(ComplexF64, alpha_)
#     beta = convert(ComplexF64, beta_)
#     if beta != Complex(0.)
#         data = Matrix{ComplexF64}(undef, size(result.data, 1), size(result.data, 2))
#     else
#         data = result.data
#     end
#     copyto!(data, B.data)
#     @inbounds for j=1:length(A.mul_before), i=1:length(A.mul_before)
#         data[i, j] *= A.mul_before[i]
#     end
#     n = size(A.mul_before)
#     A.fft_r2! * reshape(data, n...,n...)
#     N = prod(n)
#     @inbounds for j=1:N, i=1:N
#         data[i, j] *= A.mul_after[i]
#     end
#     if alpha != Complex(1.)
#         lmul!(alpha, data)
#     end
#     if beta != Complex(0.)
#         rmul!(result.data, beta)
#         result.data += data
#     end
#     nothing
# end
