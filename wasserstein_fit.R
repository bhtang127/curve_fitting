library(CVXR)
library(splines)
library(ggplot2)

trans.picture = function(pic){
  if(any(pic < 0)) stop("Not a Valid Picture")
  pic = pic / sum(pic)
  
  N = dim(pic)[1]; M = dim(pic)[2]
  xys = c(); Q = c()
  for (i in 1:N) {
    for (j in 1:M) {
      if(pic[i,j] > 0){
        xys = rbind(xys, c(i/N, j/M))
        Q = c(Q, pic[i,j])
      }
    }
  }
  
  list(xys = xys, Q = Q)
}

init.bs = function(num_point, df, xys, Q){
  basis = bs(seq(0,1,length.out = num_point), df=df, intercept=TRUE)
  X = xys[,1]; Y = xys[,2]
  
  EX = sum(X*Q); EY = sum(Y*Q) 
  covXY = matrix(0,2,2)
  covXY[1,1] = sum((X-EX)^2 * Q)
  covXY[2,2] = sum((Y-EY)^2 * Q)
  covXY[1,2] = sum((X-EX)*(Y-EY) * Q)
  covXY[2,1] = sum((X-EX)*(Y-EY) * Q)
  
  pc1 = eigen(covXY)$vectors[,1]
  print(pc1)
  
  if(pc1[1] == 0) {
    initX = rep(EX, df)
    initY = seq(0,1, length.out = df)
  }
  else if(pc1[2] == 0) {
    initY = rep(EY, df)
    initX = seq(0,1, length.out = df)
  }
  else {
    xmin = 0; ymin = 0
    xmax = 0; ymax = 0
    dis = max(EX, 1-EX, EY, 1-EY)
    vmin = min(abs(pc1))
    for(t in seq(0,dis/vmin, length.out=100)){
      if(EX - t*pc1[1] <= 0 | EY - t*pc1[2] <= 0 |
         EX - t*pc1[1] >= 1 | EY - t*pc1[2] >= 1){
        xmin = EX - t*pc1[1]
        ymin = EY - t*pc1[2]
        break
      }
    }
    for(t in seq(0,dis/vmin, length.out=100)){
      if(EX + t*pc1[1] <= 0 | EY + t*pc1[2] <= 0 |
         EX + t*pc1[1] >= 1 | EY + t*pc1[2] >= 1){
        xmax = EX + t*pc1[1]
        ymax = EY + t*pc1[2]
        break
      }
    }
    initX = seq(xmin,xmax, length.out = df)
    initY = seq(ymin,ymax, length.out = df)
  }
  
  init = cbind(initX, initY)
  list(basis = basis, init = init)
}

sp.fit = function(gamma, basis, xys, lambda=1e-3){
  gamma[gamma < 0] = 0
  N = dim(gamma)[1]; M = dim(gamma)[2]
  p = dim(basis)[2]
  
  theta1 = Variable(p)
  theta2 = Variable(p)
  X = xys[,1]; Y = xys[,2]
  onesN = matrix(1, N, 1)
  onesM = matrix(1, 1, M)
  basis = matrix(basis, dim(basis)[1], dim(basis)[2])
  optX = sum( (X %*% onesM - onesN %*% t(basis %*% theta1))^2 * gamma )
  optY = sum( (Y %*% onesM - onesN %*% t(basis %*% theta2))^2 * gamma )
  reg = lambda * sum(diff(theta1)^2 + diff(theta2)^2)
  prob = Problem(Minimize(optX + optY + reg))
  res = solve(prob)
  
  list(status=res$status,
       theta1 = res$getValue(theta1), 
       theta2 = res$getValue(theta2))
}

construct.eta = function(th1, th2, basis, xys){
  N = dim(xys)[1]; M = dim(basis)[1]
  X = xys[,1]; Y = xys[,2]
  onesN = matrix(1, N, 1)
  onesM = matrix(1, 1, M)
  basis = matrix(basis, dim(basis)[1], dim(basis)[2])
  
  eta = (X %*% onesM - onesN %*% t(basis %*% th1))^2 + 
        (Y %*% onesM - onesN %*% t(basis %*% th2))^2
  eta
}

lp.fit = function(eta, Q){
  N = dim(eta)[1]; M = dim(eta)[2]
  
  gamma = Variable(N, M)
  P = Variable(M)
  opt = sum(eta * gamma)
  onesN = matrix(1, 1, N)
  onesM = matrix(1, M, 1)
  constr = list(t(onesN %*% gamma) == P,
                sum(P) == 1,
                gamma %*% onesM == Q,
                gamma >= 0)
  prob = Problem(Minimize(opt), constr)
  res = solve(prob)
  
  list(status = res$status, 
       gamma = res$getValue(gamma),
       P = res$getValue(P))
}

loss = function(eta, gamma){
  sum(eta * gamma)
}

wasserstein.fit = function(init, basis, Q, xys, lambda = 1e-2,
                           maxiter=1000, tol = 1e-8){
  theta1 = init[,1]; theta2 = init[,2]
  
  Ps = c(); thetaX = c(); thetaY = c(); losses = c()
  for(it in 1:maxiter){
    cat(paste("it:", it, " "))
    eta = construct.eta(theta1, theta2, basis, xys)
    res.lp = lpfit(eta, Q)
    if(res.lp$status != "optimal") warning("Linear Programing Suboptimal")
    gamma = res.lp$gamma
    Ps = rbind(Ps, t(res.lp$P))
    
    res.sp = sp.fit(gamma, basis, xys, lambda = lambda)
    if(res.sp$status != "optimal") warning("Spline Fitting Suboptimal")
    theta1.new = res.sp$theta1
    theta2.new = res.sp$theta2
    
    obj = loss(construct.eta(theta1.new, theta2.new, basis, xys), gamma)
    losses = c(losses, obj)
    thetaX = rbind(thetaX, t(theta1.new))
    thetaY = rbind(thetaY, t(theta2.new))
    
    dif = norm(theta1.new - theta1, type = "2") +
          norm(theta2.new - theta2, type = "2")
    if(dif < tol){
      theta1 = theta1.new
      theta2 = theta2.new
      break
    }
    theta1 = theta1.new
    theta2 = theta2.new
  }
  
  converge = it < maxiter
  list(converge = converge, it = it, 
       theta1 = theta1, theta2 = theta2,
       Ps = Ps, losses = losses,
       thetaX.trace = thetaX,
       thetaY.trace = thetaY,
       basis = basis)
}


pic = matrix(0,100,100)
for(i in 1:500){
  x = floor(rnorm(1)*10+50)
  y = floor(rnorm(1)*2+50)
  pic[x,y] = pic[x,y]+1
}
res = trans.picture(pic)
res2 = init.bs(100,7, res$xys, res$Q)
basis = res2$basis
init = res2$init

fit = wasserstein.fit(init, basis, res$Q, res$xys, maxiter = 15)

i = 1
par(mfrow=c(1,2))
plot(res$xys[,1],res$xys[,2], xlim = c(0,1),ylim = c(0,1))
plot(basis%*%fit$thetaX.trace[i,], basis%*%fit$thetaY.trace[i,], xlim = c(0,1), ylim = c(0,1), type = "l")
i = i+1

