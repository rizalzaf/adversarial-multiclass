function linear_kernel(x1::Vector, x2::Vector)
  return dot(x1, x2)::Float64
end

function polynomial_kernel(x1::Vector, x2::Vector, d::Integer=2)
  return ((1 + dot(x1, x2)) ^ d)::Float64
end

# function gaussian_kernel(x1::Vector, x2::Vector, sigma::Real=1.0)
#   # k = exp(-1/(2*sigma^2)  ||x1 - x2||^2 )
#   return exp( -norm(x1 - x2)^2 / (2*sigma^2) )::Float64
# end

function gaussian_kernel(x1::Vector, x2::Vector, gamma::Real=1.0)
  # k = exp(-gamma  ||x1 - x2||^2 )
  return exp( -gamma * norm(x1 - x2)^2 )::Float64
end
