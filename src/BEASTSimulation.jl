module BEASTSimulation

export TreeDiffusionModel,
       NoExtensionModel,
       MergedExtension,
       StackedExtension,
       ResidualVarianceModel,
       LatentFactorModel,
       TraitSimulationModel,
       simulate,
       get_newick,
       randomFactorSimulationModel,
       split_indices

using PhyloNetworks, LinearAlgebra, LinearAlgebra.BLAS, DataFrames, Distributions
using UnPack
using BeastUtils.MatrixUtils, BEASTTreeUtils

function force_hermitian(X::AbstractMatrix{Float64})
    return Hermitian(X)
end

function force_hermitian(X::Diagonal{Float64})
    return X
end

function force_hermitian(X::Hermitian{Float64})
    return X
end

import Distributions.MvNormal
function MvNormal(μ::AbstractVector{Float64}, Σ::Hermitian{Float64})
    return MvNormal(μ, Symmetric(Σ))
end


abstract type ModelExtension end

struct TreeDiffusionModel{T <: Union{Diagonal{Float64}, Hermitian{Float64}},
                          S <: AbstractVector{Float64}}
    tree::HybridNetwork
    Σ::T
    μ::S

    function TreeDiffusionModel(tree::HybridNetwork, Σ::AbstractMatrix{Float64},
                μ::AbstractVector{Float64})
        Σ = force_hermitian(Σ)
        if size(Σ, 1) != length(μ)
            throw(DimensionMismatch("variance is of dimension $(size(Σ)), " *
                    "but mean is of dimension $(length(μ))."))
        end

        return new{typeof(Σ), typeof(μ)}(tree, Σ, μ)
    end
end

function TreeDiffusionModel(tree::HybridNetwork,
                            Σ::AbstractArray{Float64, 2})
    p = size(Σ, 1)
    μ = zeros(p)
    return TreeDiffusionModel(tree, Σ, μ)
end

function TreeDiffusionModel(newick::String,
                            Σ::AbstractArray{Float64, 2})
    tree = readTopology(newick)
    return TreeDiffusionModel(tree, Σ)
end

function TreeDiffusionModel(tree::Union{String, HybridNetwork}, p::Int)
    return TreeDiffusionModel(tree, Diagonal(ones(p)))
end

struct MergedExtension <: ModelExtension
    models::Vector{<:ModelExtension}
end

struct NoExtensionModel <: ModelExtension
    k::Int

    function NoExtensionModel(k::Int)
        if k < 1
            error("dimensionality must be at least 1")
        end

        return new(k)
    end
end

struct StackedExtension <: ModelExtension
    models::Vector{<:ModelExtension}

    function StackedExtension(models::Vector{<:ModelExtension})
        n = length(models)
        for i = 1:(n - 1)
            @assert !(typeof(models[i]) <: LatentFactorModel)
            @assert dataDimension(models[i]) == treeDimension(models[i + 1])
        end
        return new(models)
    end

end



struct ResidualVarianceModel{T <: AbstractMatrix{Float64}} <: ModelExtension
    Γ::T # residual variance

    function ResidualVarianceModel(Γ::AbstractArray{Float64, 2})
        if !issquare(Γ)
            error("Γ must be square.")
        end
        Γ = force_hermitian(Γ)
        return new{typeof(Γ)}(Γ)
    end

end

struct LatentFactorModel{S <: AbstractMatrix{Float64},
                         T <: AbstractMatrix{Float64}} <: ModelExtension
    L::S # loadings matrix
    Λ::T # residual variance

    function LatentFactorModel(L::AbstractMatrix{Float64},
                                Λ::AbstractMatrix{Float64})

        k, p = size(L)
        if !issquare(Λ)
            error("Residual variance Λ must be square.")
        end
        if size(Λ) != (p, p)
            q, s = size(Λ)
            error("Marices are non-conformable." *
                    " Matrix L has $p columns while matrix Λ is $q x $s.")
        end

        Λ = force_hermitian(Λ)

        S = typeof(L)
        T = typeof(Λ)

        return new{S, T}(L, Λ)
    end
end

function LatentFactorModel(L::AbstractMatrix{Float64})
    p = size(L, 2)
    Λ = Diagonal(ones(p))
    return LatentFactorModel(L, Λ)
end

function treeDimension(x::NoExtensionModel)
    return x.k
end

function treeDimension(x::LatentFactorModel)
    return size(x.L, 1)
end

function treeDimension(x::ResidualVarianceModel)
    return size(x.Γ, 1)
end

function treeDimension(x::StackedExtension)
    return treeDimension(x.models[1])
end

function dataDimension(x::StackedExtension)
    return dataDimension(x.models[end])
end

function dataDimension(x::NoExtensionModel)
    return x.k
end

function dataDimension(x::LatentFactorModel)
    return size(x.L, 2)
end

function dataDimension(x::ResidualVarianceModel)
    return treeDimension(x)
end

function get_loadings(x::LatentFactorModel)
    return x.L
end

function get_loadings(x::ResidualVarianceModel)
    return Diagonal(ones(treeDimension(x)))
end

function get_loadings(x::NoExtensionModel)
    return Diagonal(ones(treeDimension(x)))
end

function get_loadings(x::MergedExtension)
    tree_dims = [treeDimension(m) for m in x.models]
    data_dims = [dataDimension(m) for m in x.models]
    k = sum(tree_dims)
    p = sum(data_dims)
    L = zeros(k, p)
    k_offset = 0
    p_offset = 0

    for i = 1:length(x.models)
         L[(k_offset + 1):(k_offset + tree_dims[i]),
          (p_offset + 1):(p_offset + data_dims[i])] .= get_loadings(x.models[i])
        k_offset += tree_dims[i]
        p_offset += data_dims[i]
    end
    return L
end

function get_loadings(x::StackedExtension)
    return get_loadings(x.models[end])
end

function get_residual_variance(x::StackedExtension)
    n = length(x.models)
    V = get_residual_variance(x.models[1])

    for i = 2:n
        L = get_loadings(x.models[i])
        V = L' * V * L + get_residual_variance(x.models[i])
    end

    return V
end


function get_residual_variance(x::LatentFactorModel)
    return x.Λ
end

function get_residual_variance(x::ResidualVarianceModel)
    return x.Γ
end

function get_residual_variance(x::NoExtensionModel)
    return zeros(x.k, x.k)
end

function get_residual_variance(x::MergedExtension)
    data_dims = [dataDimension(m) for m in x.models]
    p = sum(data_dims)
    V = zeros(p, p)
    offset = 0

    for i = 1:length(x.models)
        inds = (offset + 1):(offset + data_dims[i])
        V[inds, inds] .= get_residual_variance(x.models[i])
        offset += data_dims[i]
    end
    return V
end


function randomLFM(k::Int, p::Int;
                    L_dist::UnivariateDistribution = Normal(0, 1),
                    Λ_dist::UnivariateDistribution = Gamma(2, 0.25))
    L = rand(L_dist, k, p)
    λ = rand(Λ_dist, p)
    return LatentFactorModel(L, Diagonal(λ))
end

function randomFactorSimulationModel(n::Int, k::Int, p::Int)
    tree = BEASTTreeUtils.rtree(n, ultrametric = true)
    standardize_height!(tree)
    lfm = randomLFM(k, p)
    tsm = TraitSimulationModel(tipLabels(tree), tree, lfm)
    return tsm
end

# function merge_models(models::ModelExtension...)
#     n = length(models)
#     tree_dims = treeDimension.(models)
#     data_dims = dataDimension.(models)
#     K = sum(tree_dims)
#     P = sum(data_dims)
#     L = zeros(K, P)
#     Λ = zeros(P, P)
#     k_offset = 0
#     p_offset = 0
#     for i = 1:n
#         k_end = k_offset + tree_dims[i]
#         p_end = p_offset + data_dims[i]
#         k_rng = (k_offset + 1):k_end
#         p_rng = (p_offset + 1):p_end
#         L[k_rng, p_rng] .= get_loadings(models[i])
#         Λ[p_rng, p_rng] .= get_residual_variance(models[i])
#         k_offset = k_end
#         p_offset = p_end
#     end

#     return LatentFactorModel(L, Λ)
# end

function merge_models(models::ModelExtension...)
    return MergedExtension(collect(models))
end

function split_indices(mext::MergedExtension)
    data_dims = [dataDimension(m) for m in mext.models]
    n = length(mext.models)

    inds = Vector{UnitRange{Int}}(undef, n)
    offset = 0
    for i = 1:n
        new_offset = offset + data_dims[i]
        inds[i] = (offset + 1):new_offset
        offset = new_offset
    end
    return inds
end

function split_indices(dims::Vector{Int})
    n = length(dims)

    inds = Vector{UnitRange{Int}}(undef, n)
    offset = 0
    for i = 1:n
        new_offset = offset + dims[i]
        inds[i] = (offset + 1):new_offset
        offset = new_offset
    end
    return inds
end



mutable struct TraitSimulationModel
    taxa::AbstractArray{T, 1} where T <: AbstractString
    treeModel::TreeDiffusionModel
    extensionModel::Union{Nothing, ModelExtension}

    function TraitSimulationModel(taxa::AbstractArray{T, 1} where T <: AbstractString,
                                  treeModel::TreeDiffusionModel,
                                  extensionModel::ModelExtension)
        return new(taxa, treeModel, extensionModel)
    end

    function TraitSimulationModel(taxa::AbstractArray{T, 1} where T <: AbstractString,
                                  treeModel::TreeDiffusionModel)
        return new(taxa, treeModel, nothing)
    end

    function TraitSimulationModel(taxa::AbstractArray{<:AbstractString, 1},
                                  tree::Union{String, HybridNetwork},
                                  extensionModel::ModelExtension)
        treeModel = TreeDiffusionModel(tree, treeDimension(extensionModel))
        return new(taxa, treeModel, extensionModel)
    end

    function TraitSimulationModel(tree::Union{String, HybridNetwork},
                                  extensionModel::ModelExtension)
        treeModel = TreeDiffusionModel(tree, treeDimension(extensionModel))
        return new(tipLabels(tree), treeModel, extensionModel)
    end
end

function get_tree(tsm::TraitSimulationModel)
    return tsm.treeModel.tree
end

function get_newick(tsm::TraitSimulationModel)
    return writeTopology(get_tree(tsm))
end



function get_dimension(tdm::TreeDiffusionModel)
    return length(tdm.μ)
end

function simulate_on_tree(model::TreeDiffusionModel,
                        taxa::AbstractArray{T, 1}) where T <: AbstractString
    params = PhyloNetworks.ParamsMultiBM(model.μ, Matrix(model.Σ))
    trait_sim = PhyloNetworks.simulate(model.tree, params)
    sim_taxa = trait_sim.M.tipNames
    perm = indexin(taxa, sim_taxa)
    @assert sim_taxa[perm] == taxa #TODO remove

    n = length(taxa)
    p = get_dimension(model)

    sim_Y = trait_sim[:Tips]
    Y = sim_Y'[perm, :]

    return Y
end

function add_extension(data::AbstractMatrix{Float64}, nem::NoExtensionModel)
    return data
end

function add_extension(data::AbstractMatrix{Float64}, rvm::ResidualVarianceModel)

    L_chol = cholesky(rvm.Γ).L

    n, p = size(data)

    for i = 1:n
        # yi = @view data[i, :]
        # gemm!('N', 'N', 1.0, L_chol, randn(p), 1.0, yi)
        data[i, :] .+= L_chol * randn(p)
    end
    return data
end

function add_extension(data::AbstractMatrix{Float64}, lfm::LatentFactorModel)
    n, k = size(data)
    p = size(lfm.L, 2)
    Y = data * lfm.L

    res_dist = MvNormal(zeros(p), lfm.Λ)

    for i = 1:n
        Y[i, :] .+= rand(res_dist)
    end

    return Y
end

function add_extension(data::Matrix{Float64}, me::MergedExtension)
    n, k = size(data)
    models = me.models
    ks = [treeDimension(m) for m in models]
    ps = [dataDimension(m) for m in models]
    p = sum(ps)
    Y = zeros(n, p)

    data_offset = 0
    tree_offset = 0
    for i = 1:length(models)

        data_sub = @view data[:, (tree_offset + 1):(tree_offset + ks[i])]
        Y_sub = add_extension(data_sub, models[i])
        Y[:, (data_offset + 1):(data_offset + ps[i])] .= Y_sub

        data_offset += ps[i]
        tree_offset += ks[i]
    end
    return Y
end


import PhyloNetworks: simulate
function simulate(tsm::TraitSimulationModel)

    data = simulate_on_tree(tsm.treeModel, tsm.taxa)

    if !isnothing(tsm.extensionModel)
        data = add_extension(data, tsm.extensionModel)
    end

    return data

end

include("distributions.jl")


end
