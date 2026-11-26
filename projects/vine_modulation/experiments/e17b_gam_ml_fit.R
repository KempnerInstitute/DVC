# The comparator's model fitted by penalised maximum likelihood (E17b, E17d).
#
# gamCopula 0.0.8 fits by a scoring iteration in which the working residual (score / information) is clipped at its
# 2.5% and 97.5% quantiles in every iteration (`gamCopula:::wz.update`: `qq <- quantile(u/w, c(0.025, 0.975))`,
# `z <- partrans + pmin(pmax(u/w, qq[1]), qq[2])`).  The fixed point of that iteration is not the penalised
# maximum-likelihood estimate.  `fit_ml` below is the same iteration for the Gaussian family without the clipping:
#   - same linear predictor scale as the package (eta = 2 atanh(rho)), same bound on rho (factor 1 - 1e-5),
#   - same working model: mgcv::gam of the working response on the formula with the expected information as weights,
#     smoothing parameters by mgcv's default criterion in every iteration,
#   - same stopping rule (median absolute change of the linear predictor below `tol`),
#   - added: the step is halved while the log-likelihood decreases (the package has no step control).
# Returns the gam object of the last iteration, the number of iterations, the log-likelihood and a convergence flag.
suppressPackageStartupMessages(library(mgcv))

gaussian_loglik <- function(rho, x, y) -0.5*log1p(-rho^2) - (rho^2*(x^2+y^2) - 2*rho*x*y)/(2*(1-rho^2))

bound_eta <- function(eta) 2*atanh(tanh(eta/2)*(1-1e-05))

fit_ml <- function(data, formula, tol=1e-5, n.iters=100) {
  x <- qnorm(data$u1); y <- qnorm(data$u2); n <- nrow(data)
  value_at <- function(eta) sum(gaussian_loglik(tanh(eta/2), x, y))
  rho0 <- cor(x, y)
  rho0 <- optimize(function(r) -sum(gaussian_loglik(r, x, y)), c(-0.99, 0.99))$minimum   # the package starts at the MLE
  eta <- rep(2*atanh(rho0), n)
  value <- value_at(eta)
  working <- update(formula, z ~ .)
  model <- NULL; converged <- FALSE; halvings <- 0
  for (k in seq_len(n.iters)) {
    rho <- tanh(eta/2); s <- 1-rho^2
    score <- (rho*s + (1+rho^2)*x*y - rho*(x^2+y^2))/s^2 * s/2          # d loglik / d eta
    data$wt <- (1+rho^2)/4                                               # expected information for eta
    step <- 1
    repeat {
      data$z <- eta + step*score/data$wt
      candidate <- gam(working, data=data, weights=wt)
      eta_new <- bound_eta(as.numeric(predict(candidate, data)))
      value_new <- value_at(eta_new)
      if (is.finite(value_new) && value_new >= value - 1e-9*n) break
      if (step < 1/60) break
      step <- step/2; halvings <- halvings + 1
    }
    if (!is.finite(value_new)) stop("log-likelihood not finite")
    change <- median(abs(eta_new - eta))
    eta <- eta_new; value <- value_new; model <- candidate
    if (change < tol) { converged <- TRUE; break }
  }
  list(model=model, iterations=k, loglik=value, converged=converged, halvings=halvings)
}

predict_ml <- function(fit, frame) {
  rho <- tanh(bound_eta(as.numeric(predict(fit$model, frame)))/2)
  if (any(!is.finite(rho)) || any(abs(rho) >= 1)) stop("invalid predicted parameter")
  rho
}
