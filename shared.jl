
function best_psis(psis::Vector)
  k = length(psis)
  idx = sortperm(psis, rev=true)    # sort, get the indices

  max_id = 1
  max_val = psis[idx[1]]
  for j=2:k
    v = ( sum(psis[idx[1:j]]) + j - 1 ) / j
    if v >= max_val
      max_val = v
      max_id = j
    else
      break
    end
  end

  psi_id = idx[1:max_id]

  return psi_id::Vector{Int}, max_val::Float64
end

function calc_const(psi_list::Vector, psi_id::Vector)
  j = length(psi_id)
  ret = ( sum(psi_list[psi_id]) + j - 1 ) / j
  return ret::Float64
end


function calc_cconst(psi_id::Vector)
  n_psi = length(psi_id)
  ret = (n_psi - 1 ) / n_psi
  return ret::Float64
end


## get featrure indeces for class i
function idi(m::Integer, i::Integer)
  return ((i-1)*m+1 : i*m)::UnitRange{Int64}
end

function psi_list(w::Vector, X::Matrix, y::Vector, i::Integer, c::Integer, idmi::Vector)
  psis = zeros(c)
  yi = y[i]
  for j=1:c
    if j != yi
      v1 = dot(w[idmi[j]], view(X, :, i))
      v2 = dot(w[idmi[yi]], -view(X, :, i))
      psis[j] = v1 + v2
    end
  end

  return psis::Vector{Float64}
end

## no y. for prediction
function psi_list(w::Vector, X::Matrix, i::Integer, c::Integer, idmi::Vector)
  psis = zeros(c)
  for j=1:c
    val = dot(w[idmi[j]], view(X, :, i))
    psis[j] = val
  end

  return psis::Vector{Float64}
end

# in terms of dual variable (useful for kernel methods)
function psi_list_dual(alpha::Vector, LPsi::Matrix, i::Integer, c::Integer)
  LPsi_i = view(LPsi, (i-1)*c+1 : i*c, :)
  psis = -(LPsi_i * alpha)
  return psis::Vector{Float64}
end

# in terms of dual variable (useful for kernel methods) dec
function psi_list_dual(sLa::Vector, i::Integer, c::Integer)
  psis = -(sLa[(i-1)*c+1 : i*c])
  return psis::Vector{Float64}
end

function eta_list(w::Vector, X::Matrix, i::Integer, c::Integer, idmi::Vector)
  etas = zeros(c)
  for j=1:c
    etas[j] = dot(w[idmi[j]], view(X, :, i))
  end

  return etas::Vector{Float64}
end

function calc_dot(key::Tuple{Integer,Vector,Integer,Vector}, K_ij::Float64, y::Vector)

  i = key[1]
  j = key[3]
  psi_i = key[2]
  psi_j = key[4]
  li = length(psi_i)
  lj = length(psi_j)
  yi = y[i]
  yj = y[j]

  mult = 0.0

  inii = yi in psi_i
  inij = yi in psi_j
  inji = yj in psi_i
  injj = yj in psi_j

  if yi == yj
    mult += (li - round(Int, inii)) * (lj - round(Int, injj))
  else
    if inij
      mult -= (li - round(Int, inii))
    end
    if inji
      mult -= (lj - round(Int, injj))
    end
  end

  ii = 1
  ij = 1
  while ii <= li && ij <= lj
    if psi_i[ii] > psi_j[ij]
      ij += 1
    elseif psi_i[ii] < psi_j[ij]
      ii += 1
    else  # equal
      if psi_i[ii] != yi && psi_j[ij] != yj && psi_i[ii] != yj && psi_j[ij] != yi
        mult += 1
      end
      ii += 1
      ij += 1
    end
  end

  d = (mult * K_ij) / (li * lj)

  return d::Float64
end

function calc_dot(key::Tuple{Integer,Vector,Integer,Vector}, K::Matrix, y::Vector)
  i = key[1]
  j = key[3]
  K_ij = K[i,j]

  d = calc_dot(key, K_ij, y)

  return d::Float64
end

function calc_dconst(key::Tuple{Integer,Vector}, X::Matrix, y::Vector, n_c::Integer, idmi::Vector)

    i = key[1]
    psi_i = key[2]
    li = length(psi_i)
    yi = y[i]

    xi = view(X, :, i)

    m = length(xi)
    dc = zeros(m * n_c)

    inii = yi in psi_i
    dc[idmi[yi]] = - ( (li - round(Int, inii)) * xi ) / li

    for ii = 1:li
      if psi_i[ii] != yi
        dc[idmi[psi_i[ii]]] = xi / li
      end
    end

    return dc::Vector{Float64}
end

# calc Lambda * Phi for kernel prediction
function calc_dotlphi(key::Tuple{Integer,Vector,Integer}, K_ij::Float64, y::Vector, n_c::Integer)
  i = key[1]
  psi_i = key[2]
  li = length(psi_i)
  yi = y[i]

  mults = zeros(n_c)
  for ii in psi_i
    if ii == yi
      mults[ii] = -(li - 1.0) / li
    else
      mults[ii] = 1.0 / li
    end
  end

  if !(yi in psi_i)
    mults[yi] = -1.0
  end

  return (mults * K_ij)::Vector{Float64}
end

function calc_dotlphi(key::Tuple{Integer,Vector,Integer}, K::Matrix, y::Vector, n_c::Integer)
  i = key[1]
  j = key[3]
  K_ij = K[i,j]
  return calc_dotlphi(key, K_ij, y, n_c)::Vector{Float64}
end
