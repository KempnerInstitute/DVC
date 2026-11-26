# E17 comparator: the gamCopula fitting engine (Vatter and Nagler) on one edge, on the same conditional transforms,
# family (Gaussian), link scale (copula parameter), candidate terms and validation data as the other arms.
# Forms:  vc4  time-varying coefficients of the four candidate terms (shrinkage splines, so a term can vanish)
#         vc1  time-varying coefficient of the true term only (the comparator told which term matters)
#         te   tensor smooth of time and the modulating coordinate (the comparator's own conditional form)
# Smoothing parameters: mgcv's criterion inside every scoring iteration (the package's default).  The size of the
# time basis is chosen on the validation data, separately for every form.
# Usage: Rscript e17_gam_edge.R FOLDER RLIBRARY BASES(comma separated)
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[2], .libPaths()))
suppressPackageStartupMessages(library(gamCopula))
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
folder <- args[1]
bases <- as.integer(strsplit(args[3], ",")[[1]])
roles <- c("train", "validation", "test")
data <- lapply(roles, function(role) read.csv(gzfile(file.path(folder, paste0(role, ".csv.gz")))))
names(data) <- roles
set.seed(9183)

formulas <- function(k) list(
  vc4 = as.formula(sprintf("~ s(time,k=%d,bs='cr') + s(time,by=z0,k=%d,bs='cs') + s(time,by=q0,k=%d,bs='cs') + s(time,by=z1,k=%d,bs='cs') + s(time,by=z01,k=%d,bs='cs')", k, k, k, k, k)),
  vc1 = as.formula(sprintf("~ s(time,k=%d,bs='cr') + s(time,by=z0,k=%d,bs='cr')", k, k)),
  te = as.formula(sprintf("~ te(time,z0,k=c(%d,5),bs=c('cr','cr'))", k)))

log_density <- function(rho, frame) {
  x <- qnorm(frame$u1); y <- qnorm(frame$u2)
  -0.5*log1p(-rho^2) - (rho^2*(x^2+y^2) - 2*rho*x*y)/(2*(1-rho^2))
}
predict_rho <- function(fit, frame) {
  rho <- as.numeric(gamBiCopPredict(fit, frame, target="par")$par)
  if (any(!is.finite(rho)) || any(abs(rho) >= 1)) stop("invalid predicted parameter")
  rho
}

records <- list()
best <- list()
for (form in c("vc4", "vc1", "te")) for (k in bases) {
  key <- paste(form, k, sep="_")
  started <- proc.time()[["elapsed"]]
  out <- tryCatch({
    fit <- gamBiCopFit(data$train, formula=formulas(k)[[form]], family=1, tau=FALSE, method="FS", tol.rel=1e-5, n.iters=100)
    list(fit=fit$res, validation=mean(log_density(predict_rho(fit$res, data$validation), data$validation)),
         conv=fit$conv, edf=as.numeric(sum(EDF(fit$res))), error="")
  }, error=function(e) list(fit=NULL, validation=NA, conv=NA, edf=NA, error=conditionMessage(e)))
  records[[key]] <- list(form=form, k=k, validation=out$validation, converged=isTRUE(out$conv == 0), edf=out$edf,
                         error=out$error, seconds=proc.time()[["elapsed"]] - started)
  if (!is.null(out$fit) && is.finite(out$validation) && (is.null(best[[form]]) || out$validation > best[[form]]$validation))
    best[[form]] <- list(key=key, validation=out$validation, fit=out$fit)
}
predictions <- data.frame(row=seq_len(nrow(data$test)))
selected <- list()
for (form in names(best)) {
  predictions[[form]] <- predict_rho(best[[form]]$fit, data$test)
  selected[[form]] <- best[[form]]$key
}
write.csv(predictions, gzfile(file.path(folder, "test_par.csv.gz")), row.names=FALSE)
write_json(list(records=records, selected=selected,
                versions=sapply(c("gamCopula", "VineCopula", "mgcv"), function(x) as.character(packageVersion(x))),
                r=R.version.string),
           file.path(folder, "gam_results.json"), pretty=TRUE, auto_unbox=TRUE, digits=16, na="null")
