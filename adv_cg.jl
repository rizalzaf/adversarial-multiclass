using Gurobi
using Mosek

# train adversarial method using constraint generation
function train_adv_cg(X::Matrix, y::Vector, C::Real=1.0;
  perturb::Real=0.0, tol::Real=1e-6, psdtol::Real=1e-6, obj_reltol::Real=0.0,
  log::Real=0, n_thread::Int=0, solver::Symbol=:gurobi, verbose::Bool=true)

  n = length(y)
  # add one
  X1 = [ones(n) X]'   # transpose
  m = size(X1, 1)

  # number of class
  n_c = maximum(y)
  n_f = n_c * m   # number of features

  # parameters. init with zero
  w = zeros(n_f)

  alpha = zeros(n)

  # array of tuple
  constraints = Tuple{Integer, Vector}[]

  # prepare saved vars
  idmi = map(i -> idi(m, i), collect(1:n_c))

  # precompute xi dot xj
  K = [( i >= j ? dot(view(X1,:,i), view(X1,:,j)) : 0.0 )::Float64 for i=1:n, j=1:n]
  K = [( i >= j ? K[i,j] : K[j,i] )::Float64 for i=1:n, j=1:n]

  if solver == :gurobi
    # gurobi solver
    # gurobi environtment
    env = Gurobi.Env()
    # Method : 0=primal simplex, 1=dual simplex, 2=barrier  ; default for QP: barrier
    # Threads : default = 0 (use all threads)
    setparams!(env, PSDTol=psdtol, LogToConsole=log, Method=2, Threads=n_thread)
  elseif solver == :mosek
    # mosek environtment
    env = makeenv()
  end

  # params for Gurobi
  Q = zeros(0,0)
  nu = zeros(0)
  A = spzeros(n,0)        # sparse matrix
  b = ones(n) * C

  Q_prev = zeros(0,0)
  nu_prev = zeros(0)
  A_prev = spzeros(n,0)   # sparse matrix

  # additional for w
  L = zeros(n_f,0)
  L_prev = zeros(n_f,0)

  iter = 0
  dual_obj = 0.0
  dual_obj_prev = -Inf

  while true
    iter += 1

    if verbose
      println("Iteration : ", iter)
      tic();
    end

    # previous constraints
    const_prev = copy(constraints)

    ## add to constraints
    const_added = Tuple{Integer, Vector}[]

    # find constraint for each sample
    for i=1:n
      psis = psi_list(w, X1, y, i, n_c, idmi)
      psis_id, val = best_psis(psis)      # most violated constraints

      # current xi_i
      id_i = find(x -> x[1] == i, constraints)
      xi_i_list =  map(x -> x[2], constraints[id_i])
      max_xi_i = 0
      for j = 1:length(xi_i_list)
        a = calc_const(psis::Vector, xi_i_list[j])
        if a > max_xi_i
          max_xi_i = a
        end
      end

      if val > max_xi_i
        cs = (i, sort!(psis_id))
        if findfirst(size(constraints) .== size(collect(cs))) == 0 || findfirst(constraints .== cs) == 0
          push!(constraints, cs)
          push!(const_added, cs)
        end
      end
    end

    # if no constraints added
    if length(const_added) == 0
      break
    end

    n_const = length(constraints)

    #### Start QP ###

    if verbose
      println(">> Start QP")
      toc();
      tic();
    end

    n_prev = length(const_prev)
    n_added = length(const_added)

    Q_aug = [ ( calc_dot((const_prev[i][1], const_prev[i][2], const_added[j][1], const_added[j][2]), K, y) )::Float64
          for i=1:n_prev, j=1:n_added]

    Q_aug_diag = [
          ( i >= j ? calc_dot((const_added[i][1], const_added[i][2], const_added[j][1], const_added[j][2]), K, y) : 0.0 )::Float64
          for i=1:n_added, j=1:n_added]
    Q_aug_diag = [ (i >= j ? Q_aug_diag[i,j] : Q_aug_diag[j,i] )::Float64 for i=1:n_added, j=1:n_added ]

    Q = [
          ( (i <= n_prev && j <= n_prev) ? Q_prev[i, j] : ( i <= n_prev ? Q_aug[i, j-n_prev] :
          ( j <= n_prev ? Q_aug[j, i-n_prev] : Q_aug_diag[i-n_prev, j-n_prev] ) ) )::Float64
          for i=1:n_const, j=1:n_const]

    nu_aug = [ ( calc_cconst(const_added[i][2]) )::Float64 for i=1:n_added ]
    nu = [ ( i <= n_prev ? nu_prev[i] : nu_aug[i-n_prev] )::Float64 for i=1:n_const]

    A_aug = spzeros(n, n_added)
    for j = 1:n_added
      A_aug[const_added[j][1], j] = 1.0
    end
    A = [A_prev A_aug]

    ## add perturbation
    for i=1:n_const
      Q[i,i] = Q[i,i] + perturb
    end

    if verbose
      toc();
      tic();
    end

    if solver == :gurobi

      if verbose println(">> Optim :: Gurobi") end

      ## init model
      model = gurobi_model(env,
                 sense = :minimize,
                 H = Q,
                 f = -nu,
                 A = A,
                 b = b,
                 lb = zeros(n_const)
                 )
      # Print the model to check correctness
      # print(model)

      # Solve with Gurobi
      Gurobi.optimize(model)


      if verbose
        toc();
        println("<< End QP")
      end

      dual_obj = -get_objval(model)
      # Solution
      if verbose println("Objective value: ", dual_obj) end

      # get alpha
      alpha = get_solution(model)

      if verbose println("n constraints = ", length(constraints)) end

      ### end QP ###

    elseif solver == :mosek

      if verbose println(">> Optim :: Mosek") end

      task = maketask(env)

      # set params
      putintparam(task, Mosek.MSK_IPAR_LOG, 1)
      putintparam(task, Mosek.MSK_IPAR_LOG_CHECK_CONVEXITY, 1)
      putdouparam(task, Mosek.MSK_DPAR_CHECK_CONVEXITY_REL_TOL, psdtol)

      # variables
      appendvars(task, n_const)
      # bound on var
      for i::Int32 = 1:n_const
        putbound(task, Mosek.MSK_ACC_VAR, i, Mosek.MSK_BK_RA, 0.0, C)
      end

      # objective
      putobjsense(task, Mosek.MSK_OBJECTIVE_SENSE_MINIMIZE)
      qi = zeros(Int32, (n_const * (n_const+1)) ÷ 2 )
      qj = zeros(Int32, (n_const * (n_const+1)) ÷ 2 )
      qv = zeros(Float64, (n_const * (n_const+1)) ÷ 2 )
      ix = 1
      for j::Int32 = 1:n_const
        for i::Int32 = j:n_const
          qi[ix] = i
          qj[ix] = j
          qv[ix] = Q[i,j]
          ix += 1
        end
      end
      putqobj(task, qi, qj, qv)

      putclist(task, collect(1:n_const), -nu)

      # constraints
      ## sparse array
      appendcons(task, n)
      for i::Int32 = 1:n_const
        id_nz = A[:,i].nzind
        putacol(task, i, id_nz, ones(length(id_nz)))
      end

      for i::Int32 = 1:n
        putbound(task, Mosek.MSK_ACC_CON, i, Mosek.MSK_BK_RA, 0.0, C)
      end

      if verbose
        toc(); tic();
      end

      Mosek.optimize(task)

      if verbose
        toc();
        println("<< End QP")
      end

      # Solution
      dual_obj, _ = getsolutioninfo(task, Mosek.MSK_SOL_ITR)
      if verbose println("Objective value: ", -dual_obj) end

      # get alpha
      alpha = getxx(task, Mosek.MSK_SOL_ITR)

      if verbose println("n constraints = ", length(constraints)) end
    end

    L_aug = zeros(n_f, n_added)
    for i=1:n_added
      L_aug[:, i] = calc_dconst(const_added[i], X1, y, n_c, idmi)
    end
    L = [L_prev L_aug]

    # recover w
    w = zeros(n_f)
    for i = 1:n_const
      w -= alpha[i] * L[:, i]
    end

    if obj_reltol > 0.0
      if (dual_obj - dual_obj_prev) / dual_obj_prev < obj_reltol && iter > 1
        if verbose println(">> Iteration STOPPED | Objective relative tolerance : ", obj_reltol) end
        break
      end
    end

    if verbose println() end

    Q_prev = Q
    nu_prev = nu
    A_prev = A
    L_prev = L
    dual_obj_prev = dual_obj
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
  adv_model = MultiAdversarialModel(w, alpha, constraints, n_c, game_value_01, game_value_augmented, mean(l_adv), mean(l_01))

  return adv_model::MultiAdversarialModel
end


function predict_adv(model::MultiAdversarialModel, X_test::Matrix)

  w = model.w
  n_c = model.n_class
  n = size(X_test, 1)

  X1 = [ones(n) X_test]'   # transpose
  m = size(X1, 1)

  # prepare saved vars
  idmi = map(i -> idi(m, i), collect(1:n_c))

  prob = zeros(n, n_c)
  pred = zeros(n)
  for i=1:n
    psis = psi_list(w, X1, i, n_c, idmi)
    psis_id, val = best_psis(psis)      # most violated constraints
    n_ps = length(psis_id)

    for j=1:n_c
      if j in psis_id
        prob[i,j] = ( (n_ps-1.0)*psis[j] - sum(psis[psis_id[psis_id .!= j]]) + 1.0 ) / n_ps
      else
        prob[i,j] = 0.0
      end
    end

    pred[i] = indmax(psis)

  end

  return prob::Matrix{Float64}, pred::Vector{Float64}
end

function test_adv(model::MultiAdversarialModel, X_test::Matrix, y_test::Vector)
  n = size(X_test, 1)

  y_prob, y_pred = predict_adv(model, X_test)

  # calculate testing loss
  losses = zeros(n)
  losses01 = zeros(n)
  for i=1:n
    losses[i] = 1.0 - y_prob[i, y_test[i]]
    losses01[i] = 1.0 - round(Int, y_pred[i] == y_test[i])
  end

  loss = sum(losses) / n
  loss01 = sum(losses01) / n

  return loss::Float64, losses::Vector{Float64}, loss01::Float64, losses01::Vector{Float64},
   y_prob::Matrix{Float64}, y_pred::Vector{Float64}

end
