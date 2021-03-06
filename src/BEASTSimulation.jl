module BEASTSimulation

export TreeDiffusionModel,
       ResidualVarianceModel,
       LatentFactorModel,
       TraitSimulationModel,
       simulate,
       get_newick,
       randomFactorSimulationModel

using PhyloNetworks, LinearAlgebra, LinearAlgebra.BLAS, DataFrames, Distributions
using BeastUtils.MatrixUtils, BeastUtils.TreeUtils

abstract type ModelExtension end

struct TreeDiffusionModel
    tree::HybridNetwork
    Σ::AbstractArray{Float64, 2}
    μ::AbstractArray{Float64, 1}

    function TreeDiffusionModel(tree::HybridNetwork,
                                Σ::AbstractArray{Float64, 2})
        p, q = size(Σ)
        if p != q
            error("Σ must be square.")
        end
        μ = zeros(p)
        return new(tree, Σ, μ)
    end

    function TreeDiffusionModel(newick::String,
                                Σ::AbstractArray{Float64, 2})
        tree = readTopology(newick)
        return TreeDiffusionModel(tree, Σ)
    end

    function TreeDiffusionModel(tree::Union{String, HybridNetwork}, p::Int)
        return TreeDiffusionModel(tree, Diagonal(ones(p)))
    end

end

struct ResidualVarianceModel <: ModelExtension
    Γ::AbstractArray{Float64, 2} # residual variance

    function ResidualVarianceModel(Γ::AbstractArray{Float64, 2})
        if !issquare(Γ)
            error("Γ must be square.")
        end

        return new(Γ)
    end

end

struct LatentFactorModel <: ModelExtension
    L::AbstractArray{Float64, 2} # loadings matrix
    Λ::Diagonal{Float64} # residual variance

    function LatentFactorModel(L::AbstractArray{Float64, 2},
                                Λ::Diagonal{Float64})

        k, p = size(L)
        if size(Λ, 1) != p
            q = size(Λ, 1)
            error("Marices are non-conformable." *
                    " Matrix L has $p columns while matrix Λ is $q x $q.")
        end

        return new(L, Λ)
    end

end

function treeDimension(x::LatentFactorModel)
    return size(x.L, 1)
end

function treeDimension(x::ResidualVarianceModel)
    return size(x.Γ, 1)
end

function randomLFM(k::Int, p::Int;
                    L_dist::UnivariateDistribution = Normal(0, 1),
                    Λ_dist::UnivariateDistribution = Gamma(2, 0.25))
    L = rand(L_dist, k, p)
    λ = rand(Λ_dist, p)
    return LatentFactorModel(L, Diagonal(λ))
end

function randomFactorSimulationModel(n::Int, k::Int, p::Int)
    tree = TreeUtils.rtree(n, ultrametric = true)
    standardize_height!(tree)
    lfm = randomLFM(k, p)
    tsm = TraitSimulationModel(tipLabels(tree), tree, lfm)
    return tsm
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

function add_extension(data::Matrix{Float64}, rvm::ResidualVarianceModel)

    L_chol = cholesky(rvm.Γ).L

    n, p = size(data)

    for i = 1:n
        # yi = @view data[i, :]
        # gemm!('N', 'N', 1.0, L_chol, randn(p), 1.0, yi)
        data[i, :] .+= L_chol * randn(p)
    end
    return data
end

function add_extension(data::Matrix{Float64}, lfm::LatentFactorModel)
    n, k = size(data)
    p = size(lfm.L, 2)
    Y = data * lfm.L
    chol_vec = sqrt.(lfm.Λ.diag) # Λ is diagonal

    for i = 1:n
        Y[i, :] .+= chol_vec .* randn(p)
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




end
