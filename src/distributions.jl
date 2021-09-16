struct VarianceComponents{Q <: AbstractMatrix{Float64},
                          R <: AbstractMatrix{Float64},
                          S <: AbstractMatrix{Float64},
                          T <: AbstractMatrix{Float64}}
    Ψ::Q
    Σ::R
    L::S
    Λ::T
end


function treeDimension(tsm::TraitSimulationModel)
    return get_dimension(tsm.treeModel)
end

function dataDimension(tsm::TraitSimulationModel)
    return isnothing(tsm.extensionModel) ?
            treeDimension(tsm) : dataDimension(tsm.extensionModel)
end

function get_loadings(tsm::TraitSimulationModel)
    if isnothing(tsm.extensionModel)
        return I
    end
    return get_loadings(tsm.extensionModel)
end

function get_residual_variance(tsm::TraitSimulationModel)
    if isnothing(tsm.extensionModel)
        return Diagonal(zeros(dataDimension(tsm)))
    end
    return get_residual_variance(tsm.extensionModel)
end

function tree_variance(tsm::TraitSimulationModel; pss::Float64 = Inf)
    Ψ = vcv(tsm.treeModel.tree, tsm.taxa)
    n = size(Ψ, 1)
    Ψ .+= 1.0 / pss * ones(n, n)
    return Hermitian(Ψ)
end


function data_distribution(tsm::TraitSimulationModel; pss::Float64 = Inf)
    comps = varianceComponents(tsm)
    Σ = data_variance(comps)
    n = length(tsm.taxa)
    μ = vec(ones(n) * (tsm.treeModel.μ' * comps.L))

    return MvNormal(μ, Symmetric(Σ))
end

function factor_marginal_distribution(tsm::TraitSimulationModel; pss::Float64 = Inf)
    Ψ = tree_variance(tsm, pss = pss)
    Σ = tsm.treeModel.Σ
    return Hermitian(kron(Σ, Ψ))
end


function varianceComponents(tsm::TraitSimulationModel; pss::Float64 = Inf)
    Ψ = tree_variance(tsm, pss = pss)
    Σ = tsm.treeModel.Σ
    L = get_loadings(tsm)
    Λ = get_residual_variance(tsm)
    return VarianceComponents(Ψ, Σ, L, Λ)
end

function factor_variance(vc::VarianceComponents)
    return Hermitian(kron(vc.Σ, vc.Ψ))
end

function factor_data_covariance(vc::VarianceComponents)
    return kron(vc.Σ * vc.L, vc.Ψ)
end

function data_variance(vc::VarianceComponents)
    return Hermitian(kron(vc.L' * vc.Σ * vc.L, vc.Ψ) .+
        kron(vc.Λ, Diagonal(ones(size(vc.Ψ, 1)))))
end

function dimensions(tsm::TraitSimulationModel)
    n = length(tsm.taxa)
    k, p = size(get_loadings(tsm))
    dim = n * (k + p)
    f_inds = 1:(n * k)
    y_inds = 1:(n * p)
    y_inds = y_inds .+ (n * k)
    return (factor_indices = f_inds, data_indices = y_inds, dimension = dim)
end


function factor_and_data_joint_distribution(tsm::TraitSimulationModel; pss::Float64 = Inf)
    comps = varianceComponents(tsm)
    n = length(tsm.taxa)

    f_inds, y_inds, dim = dimensions(tsm)

    V = zeros(dim, dim)
    V[f_inds, f_inds] .= factor_variance(comps)
    V[f_inds, y_inds] .= factor_data_covariance(comps)
    V[y_inds, f_inds] .= (@view V[f_inds, y_inds])'
    V[y_inds, y_inds] .= data_variance(comps)

    μ_fac = ones(n) * tsm.treeModel.μ'
    μ_data = μ_fac * comps.L

    μ = vcat(vec(μ_fac), vec(μ_data))
    return MvNormal(μ, Symmetric(V))
end

function factor_conditional_distribution(tsm::TraitSimulationModel,
        data::AbstractArray{Float64}; pss::Float64 = Inf)
    @unpack μ, Σ = factor_and_data_joint_distribution(tsm, pss = pss)
    f_inds, y_inds, dim = dimensions(tsm)
    Vff = @view Σ[f_inds, f_inds]
    Vyf = @view Σ[y_inds, f_inds]
    Vyy = @view Σ[y_inds, y_inds]
    μf = @view μ[f_inds]
    μy = @view μ[y_inds]
    y = vec(data)

    Pyy = inv(Hermitian(Vyy))

    cμ = μf + Vyf' * (Pyy * (y - μy))
    cΣ = Symmetric(Vff) - Symmetric(Vyf' * Pyy * Vyf)

    return MvNormal(cμ, cΣ)
end












