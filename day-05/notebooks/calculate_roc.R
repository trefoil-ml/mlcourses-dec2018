calculate_roc <- function(df, cost_of_fp, cost_of_fn, n=100) {
  tpr <- function(df, threshold) {
    sum(df$pred >= threshold & df$deposit == "term.deposit") / sum(df$deposit == "term.deposit")
  }
  
  fpr <- function(df, threshold) {
    sum(df$pred >= threshold & df$deposit == "no") / sum(df$deposit == "no")
  }
  
  cost <- function(df, threshold, cost_of_fp, cost_of_fn) {
    sum(df$pred >= threshold & df$deposit == "no") * cost_of_fp + 
      sum(df$pred < threshold & df$deposit == "term.deposit") * cost_of_fn
  }
  
  roc <- data.frame(threshold = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tpr <- sapply(roc$threshold, function(th) tpr(df, th))
  roc$fpr <- sapply(roc$threshold, function(th) fpr(df, th))
  roc$cost <- sapply(roc$threshold, function(th) cost(df, th, cost_of_fp, cost_of_fn))
  
  return(roc)
}