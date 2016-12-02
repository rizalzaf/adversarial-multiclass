
function standardize(data::Matrix)
  m,n = size(data)
  standardized = zeros(m,n)
  mean_vector = zeros(n)
  std_vector = zeros(n)
  for i = 1:n
    mean_vector[i] = mean(data[:,i])
    std_vector[i] = std(data[:,i])
    if std_vector[i] != 0
      standardized[:,i] = (data[:,i]-mean_vector[i])./std_vector[i]
    elseif mean_vector[i] < 1 && mean_vector[i] >=0
      standardized[:,i] = data[:,i]
    else
      standardized[:,i] = 1
    end
  end
  return standardized::Matrix{Float64}, mean_vector::Vector{Float64}, std_vector::Vector{Float64}
end

function standardize(data::Matrix, mean_vector::Vector{Float64}, std_vector::Vector{Float64})
  m,n = size(data)
  standardized = zeros(m,n)
  for i = 1:n
    if std_vector[i] != 0
      standardized[:,i] = (data[:,i]-mean_vector[i]) ./ std_vector[i]
    elseif mean_vector[i] < 1 && mean_vector[i] >=0
      standardized[:,i] = data[:,i]
    else
      standardized[:,i] = 1
    end
  end
  return standardized
end


function normalize(X::Matrix)
  # normalize to 0 1
  r_nrm = 1.0   # range
  shift = 0.0
  X_max = maximum(X, 1)
  X_min = minimum(X, 1)
  X_nrm = (r_nrm * broadcast(-, X, X_min) ./ broadcast(-, X_max, X_min)) + shift

  return X_nrm
end

function k_fold(n::Int, k::Int)
  idx = randperm(n)

  # allocate folds
  folds = Vector[]
  n_f = round(Int, floor(n/k))
  add_f = n % k
  j = 1
  for i=1:k
    if i <= add_f
      push!(folds, idx[j:j+n_f])
      j += n_f+1
    else
      push!(folds, idx[j:j+n_f-1])
      j += n_f
    end
  end

  return folds::Vector{Vector}
end

function jaakkola_heuristic(X::Matrix, y::Vector)
  n = size(X,1)
  Xt = X'   # transpose is more efficient
  min_dist = zeros(n)
  for i=1:n
    ds = Inf
    for j=1:n
      if y[i] != y[j]
        d = norm(view(Xt,:,i) - view(Xt,:,j))
        if d < ds
          ds = d
        end
      end
    end
    min_dist[i] = ds
  end

  sigma = median(min_dist)
  gamma = 0.5 / (sigma * sigma)

  return gamma
end

function count_constraints(model::MultiAdversarialModel, y_train::Vector)
  w = model.w
  alpha = model.alpha
  cs = model.constraints

  n_train = length(y_train)

  # via dec or full
  if cs[1][1] == 1 && cs[1][2] == [y_train[1]]
    optim = :dec
    cs_added = cs[n_train+1:end]
  else
    optim = :full
    cs_added = cs
  end

  n_cs_added = length(cs_added)
  const_num = zeros(n_train)
  for i=1:n_train
    const_num[i] = length(find(x::Tuple{Integer, Vector} -> x[1] == i, cs_added))
  end
  max_cs_added = maximum(const_num)

  zero_thereshold = 1e-3
  active_const_num = zeros(n_train)
  for i=1:n_train
    idx = find(x::Tuple{Integer, Vector} -> x[1] == i, cs_added)
    active_const_num[i] = sum(alpha[idx] .> zero_thereshold)
  end

  n_cs_active = sum(active_const_num)
  max_cs_active = maximum(active_const_num)
  n_sv = sum(active_const_num .> 0)

  return n_cs_added, max_cs_added, n_cs_active, max_cs_active, n_sv
end

function count_constraints(model::KernelMultiAdversarialModel, y_train::Vector)
  alpha = model.alpha
  cs = model.constraints

  n_train = length(y_train)

  # via dec or full
  if cs[1][1] == 1 && cs[1][2] == [y_train[1]]
    optim = :dec
    cs_added = cs[n_train+1:end]
  else
    optim = :full
    cs_added = cs
  end

  n_cs_added = length(cs_added)
  const_num = zeros(n_train)
  for i=1:n_train
    const_num[i] = length(find(x::Tuple{Integer, Vector} -> x[1] == i, cs_added))
  end
  max_cs_added = maximum(const_num)

  zero_thereshold = 1e-3
  active_const_num = zeros(n_train)
  for i=1:n_train
    idx = find(x::Tuple{Integer, Vector} -> x[1] == i, cs_added)
    active_const_num[i] = sum(alpha[idx] .> zero_thereshold)
  end

  n_cs_active = sum(active_const_num)
  max_cs_active = maximum(active_const_num)
  n_sv = sum(active_const_num .> 0)

  return n_cs_added, max_cs_added, n_cs_active, max_cs_active, n_sv
end
