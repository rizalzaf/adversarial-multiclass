
function fg_one!(w::Vector, g::Vector, i::Int64, X1::Matrix, y::Vector, C::Real, n::Int64, n_c::Int64, idmi::Vector)
  m = length(g)
  halfnorm = 0.5 * dot(w, w)

  psis = psi_list(w, X1, y, i, n_c, idmi)
  psis_id, xi = best_psis(psis)      # most violated constraints

  dxi = calc_dconst((i,psis_id), X1, y, n_c, idmi)

  for i=1:m
    g[i] = w[i] +  C * dxi[i]
  end

  return halfnorm + C * xi
end

# train adversarial method using SGD
function train_adv_sgd(X::Matrix, y::Vector, C::Real=1.0;
  step::Float64=0.1, use_adagrad::Bool=false,
  ftol::Real=1e-6, grtol::Real=1e-6,
  show_trace::Bool=true, max_iter::Int=1000)

  n = length(y)
  # add one
  X1 = [ones(n) X]'   # transpose
  m = size(X1, 1)

  # number of class
  n_c = maximum(y)
  n_f = n_c * m   # number of features

  # parameters. init with zero
  w = rand(n_f) - 0.5
  grad = zeros(n_f)

  # prepare saved vars
  idmi = map(i -> idi(m, i), collect(1:n_c))

  # adagrad
  # rate is something you need to set beforehand
  rate = step
  square_g = zeros(n_f)   # for storing historical square of grads

  f_prev = Inf
  iter = 0

  while true
    iter = iter + 1

    ids = randperm(n)
    sum_f = 0.0

    for i in ids

      f = fg_one!(w, grad, i, X1, y, C, n, n_c, idmi)
      sum_f += f
      if show_trace
        println("iter : ", iter, ", sample : ", i, ", C : ", C, ", f : ", f, ", abs grad : ", mean(abs(grad)))
      end

      if use_adagrad
        # adagrad
        square_g += grad .^ 2
        w = w - (rate ./ sqrt(square_g)) .* grad
      else
        w = w - step * grad
      end

    end

    # discount step
    step = step * 0.95

    if iter >= max_iter
      if show_trace println("maximum iteration reached!!") end
      break
    end

    if mean(abs(grad)) < grtol
      if show_trace println("gradient breaks!!") end
      break
    end

    f = sum_f / n
    if abs(f_prev - f) < ftol
      if show_trace println("function breaks!!") end
      break
    end
    f_prev = f

  end

  # finalizing losses
  gv_aug = zeros(n)
  gv_01 = zeros(n)
  l_adv = zeros(n)
  l_01 = zeros(n)
  for i=1:n
    psis = psi_list(w, X1, y, i, n_c, idmi)
    psis_id, val = best_psis(psis)      # most violated constraints
    n_ps = length(psis_id)

    gv_aug[i] = val

    # compute probs
    p_hat = zeros(n_c)
    p_check = zeros(n_c)
    for j=1:n_c
      if j in psis_id
        p_hat[j] = ( (n_ps-1.0)*psis[j] - sum(psis[psis_id[psis_id .!= j]]) + 1.0 ) / n_ps
        p_check[j] = 1.0 / n_ps
      else
        p_hat[j] = 0.0
        p_check[j] = 0.0
      end
    end

    C01 = 1 - eye(n_c)    # 01 loss matrix
    v = p_hat' * C01 * p_check   # the result is vector size 1 not a number
    gv_01[i] = v[1]     #

    ## training loss
    l_adv[i] = 1.0 - p_hat[y[i]]
    l_01[i] = 1.0 - round(Int, indmax(p_hat) == y[i])
  end

  game_value_01 = mean(gv_01)
  game_value_augmented = mean(gv_aug)

  # create model
  adv_model = MultiAdversarialModel(w, zeros(0), Tuple{Integer, Vector}[], n_c, game_value_01, game_value_augmented, mean(l_adv), mean(l_01))

  return adv_model::MultiAdversarialModel
end
