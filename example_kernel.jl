include("types.jl")
include("shared.jl")
include("util.jl")
include("kernel.jl")
include("adv_kernel_cg.jl")

## set
solver = :mosek
# solver = :gurobi

log = 0
psdtol = 1e-6
perturb = 1e-12
obj_reltol_cv = 0.0
obj_reltol_test = 0.0
verbose = false

smaller_cv = false   # 3 instead of 5

### prepare data

dname = "glass"
D_all = readcsv("data-example/" * dname * ".csv")
id_train = readcsv("data-example/" * dname * ".train")
id_test = readcsv("data-example/" * dname * ".test")

id_train = round(Int64, id_train)
id_test = round(Int64, id_test)

println(dname)

### Cross Validation, using first split
## First stage

id_tr = vec(id_train[1,:])
id_ts = vec(id_test[1,:])
X_train = D_all[id_tr,1:end-1]
y_train = round(Int, D_all[id_tr, end])

X_test = D_all[id_ts,1:end-1]
y_test = round(Int, D_all[id_ts, end])

X_train, mean_vector, std_vector = standardize(X_train)
X_test = standardize(X_test, mean_vector, std_vector)

if smaller_cv
  Cs =  [2.0^i for i=0:4:8]
  Gs =  [2.0^(-8+i) for i=0:4:8]
else
  Cs =  [2.0^i for i=0:3:12]
  Gs =  [2.0^(-12+i) for i=0:3:12]
end

ncs = length(Cs)

Pars = [ Tuple{Float64,Float64}((Cs[i], Gs[j])) for i=1:ncs, j=1:ncs ]
Pars = vec(Pars)
npar = length(Pars)

# fold
n_train = size(X_train, 1)
n_test = size(X_test, 1)
kf = 5

# k folds
folds = k_fold(n_train, kf)

loss_list = zeros(npar)
loss01_list = zeros(npar)

# The first stage of CV
idx = randperm(n_train)
X_train = X_train[idx,:]
y_train = y_train[idx]

println("First CV")

for i = 1:npar

  println(i, " | Adversarial | C = ", Pars[i][1], ", Gamma = ", Pars[i][2])

  losses = zeros(n_train)
  losses1 = zeros(n_train)
  # k fold
  for j = 1:kf
    # prepare training and validation
    id_tr = vcat(folds[[1:j-1; j+1:end]]...)
    id_val = folds[j]

    X_tr = X_train[id_tr, :];  y_tr = y_train[id_tr]
    X_val = X_train[id_val, :];  y_val = y_train[id_val]

    C = Pars[i][1]
    gamma = Pars[i][2]

    print("    ",j, "-th fold : ")
    @time model = train_adv_kernel_cg(X_tr, y_tr, C, :gaussian, [gamma], perturb=perturb, obj_reltol=obj_reltol_cv, solver=solver, log=log, psdtol=psdtol, verbose=verbose)

    _, ls, _, ls1, _, _ = test_adv_kernel(model, X_val, y_val, X_tr, y_tr)

    losses[id_val] = ls
    losses1[id_val] = ls1

  end

  loss_list[i] = mean(losses)
  loss01_list[i] = mean(losses1)
  # println("loss : ", string(mean(losses)))
  println("  => loss01 : ", string(mean(losses1)))

end

ind_max= indmin(loss01_list)
C0 = Pars[ind_max][1]
G0 = Pars[ind_max][2]
if smaller_cv
  Cs =  [C0*2.0^(i-2) for i=1:3]
  Gs =  [G0*2.0^(i-2) for i=1:3]
else
  Cs =  [C0*2.0^(i-3) for i=1:5]
  Gs =  [G0*2.0^(i-3) for i=1:5]
end
ncs = length(Cs)

Pars = [ Tuple{Float64,Float64}((Cs[i], Gs[j])) for i=1:ncs, j=1:ncs ]
Pars = vec(Pars)
npar = length(Pars)

## Second stage
idx = randperm(n_train)
X_train = X_train[idx,:]
y_train = y_train[idx]

println("Second CV")
for i = 1:npar

  println(i, " | Adversarial | C = ", Pars[i][1], ", Gamma = ", Pars[i][2])

  losses = zeros(n_train)
  losses1 = zeros(n_train)
  # k fold
  for j = 1:kf
    # prepare training and validation
    id_tr = vcat(folds[[1:j-1; j+1:end]]...)
    id_val = folds[j]

    X_tr = X_train[id_tr, :];  y_tr = y_train[id_tr]
    X_val = X_train[id_val, :];  y_val = y_train[id_val]

    C = Pars[i][1]
    gamma = Pars[i][2]

    print("    ",j, "-th fold : ")
    @time model = train_adv_kernel_cg(X_tr, y_tr, C, :gaussian, [gamma], perturb=perturb, obj_reltol=obj_reltol_cv, solver=solver, log=log, psdtol=psdtol, verbose=verbose)

    _, ls, _, ls1, _, _ = test_adv_kernel(model, X_val, y_val, X_tr, y_tr)

    losses[id_val] = ls
    losses1[id_val] = ls1
    #println("loss : ", string(ls))
    #println("loss01 : ", string(ls))

  end

  loss_list[i] = mean(losses)
  loss01_list[i] = mean(losses1)
  #println("loss : ", string(mean(losses)))
  println("  => loss01 : ", string(mean(losses1)))

end

ind_max= indmin(loss01_list)
C_best = Pars[ind_max][1]
G_best = Pars[ind_max][2]

### Evaluation

n_split = size(id_train, 1)

v_model = Vector{ClassificationModel}()
v_result = Vector{Tuple}()
v_acc = zeros(n_split)
v_cs_result = zeros(n_split, 5)

println("Evaluation")
for i = 1:n_split
  # standardize
  id_tr = vec(id_train[i,:])
  id_ts = vec(id_test[i,:])
  X_train = D_all[id_tr,1:end-1]
  y_train = round(Int, D_all[id_tr, end])

  X_test = D_all[id_ts,1:end-1]
  y_test = round(Int, D_all[id_ts, end])

  X_train, mean_vector, std_vector = standardize(X_train)
  X_test = standardize(X_test, mean_vector, std_vector)

  #train and test
  @time model = train_adv_kernel_cg(X_train, y_train, C_best, :gaussian, [G_best], perturb=perturb, obj_reltol=obj_reltol_test, solver=solver, log=log, psdtol=psdtol, verbose=verbose)

  result = test_adv_kernel(model, X_test, y_test, X_train, y_train)
  loss01 = result[3]
  acc = 1.0 - loss01
  cs_result = count_constraints(model, y_train)

  println("accuracy : ", acc)

  push!(v_model, model)
  push!(v_result, result)
  v_acc[i] = acc
  v_cs_result[i, :] = collect(cs_result)
end

println(dname)
println("mean accuracy : ", mean(v_acc))
println("std accuracy : ", std(v_acc))
