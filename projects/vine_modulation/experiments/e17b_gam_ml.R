# E17b comparator: the three forms of e17_gam_edge.R (same formulas, bases, smoothing criterion and validation choice)
# fitted by penalised maximum likelihood, i.e. by the package's scoring iteration without its clipping of the working
# residuals (see e17b_gam_ml_fit.R).  Inputs and outputs as e17_gam_edge.R.
# Usage: Rscript e17b_gam_ml.R FOLDER RLIBRARY BASES(comma separated) FITTER(e17b_gam_ml_fit.R)
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[2], .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
source(args[4])
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

log_density <- function(rho, frame) gaussian_loglik(rho, qnorm(frame$u1), qnorm(frame$u2))

records <- list()
best <- list()
for (form in c("vc4", "vc1", "te")) for (k in bases) {
  key <- paste(form, k, sep="_")
  started <- proc.time()[["elapsed"]]
  out <- tryCatch({
    fit <- fit_ml(data$train, formulas(k)[[form]], tol=1e-5, n.iters=100)
    list(fit=fit, validation=mean(log_density(predict_ml(fit, data$validation), data$validation)), conv=fit$converged,
         iterations=fit$iterations, halvings=fit$halvings, edf=as.numeric(sum(fit$model$edf)), loglik=fit$loglik, error="")
  }, error=function(e) list(fit=NULL, validation=NA, conv=NA, iterations=NA, halvings=NA, edf=NA, loglik=NA, error=conditionMessage(e)))
  records[[key]] <- list(form=form, k=k, validation=out$validation, converged=isTRUE(out$conv), iterations=out$iterations,
                         halvings=out$halvings, edf=out$edf, loglik=out$loglik, error=out$error,
                         seconds=proc.time()[["elapsed"]] - started)
  if (!is.null(out$fit) && is.finite(out$validation) && (is.null(best[[form]]) || out$validation > best[[form]]$validation))
    best[[form]] <- list(key=key, validation=out$validation, fit=out$fit)
}
predictions <- data.frame(row=seq_len(nrow(data$test)))
selected <- list()
for (form in names(best)) {
  predictions[[form]] <- predict_ml(best[[form]]$fit, data$test)
  selected[[form]] <- best[[form]]$key
}
write.csv(predictions, gzfile(file.path(folder, "test_par.csv.gz")), row.names=FALSE)
write_json(list(records=records, selected=selected, versions=sapply(c("mgcv"), function(x) as.character(packageVersion(x))),
                r=R.version.string),
           file.path(folder, "gam_results.json"), pretty=TRUE, auto_unbox=TRUE, digits=16, na="null")
