abstract ClassificationModel

type MultiAdversarialModel <: ClassificationModel
  w::Vector{Float64}
  alpha::Vector{Float64}
  constraints::Vector{Tuple{Integer, Vector{Integer}}}
  n_class::Int
  game_value_01::Float64
  game_value_augmented::Float64
  train_adv_loss::Float64
  train_01_loss::Float64
end

type KernelMultiAdversarialModel <: ClassificationModel
  kernel::Symbol
  kernel_params::Vector{Float64}
  alpha::Vector{Float64}
  constraints::Vector{Tuple{Integer, Vector{Integer}}}
  n_class::Int
  game_value_01::Float64
  game_value_augmented::Float64
  train_adv_loss::Float64
  train_01_loss::Float64
end
