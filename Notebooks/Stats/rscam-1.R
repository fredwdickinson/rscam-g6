library(rstan)

# Read data, check for NAs - Stan can't with them
data = read.csv("small_df.csv")
sum(is.na(data$Age))
sum(is.na(data$HVGSc.Interest))

# Transform the column names for ease.
stan_data <- list(
  N = nrow(data),
  age = data$Age,
  is_hvgsc = data$HGVSc.Interest
)

model_string = "
data {
  int<lower=0> N;     
  vector[N] age;                 
  vector[N] is_hvgsc;       
}

parameters {
  real beta0;                 
  real beta1;                   
  real<lower=0> sigma;
}

model {
  // Assumes mean 65 age, sigma=8.
  beta0 ~ normal(65, 8);        
  
  // Prior assumes that beta1 has no influence (mean zero).
  beta1 ~ normal(0, 10);         
  sigma ~ exponential(0.25);      
  age ~ normal(beta0 + beta1*is_hvgsc, sigma);
}
"

# =============

res <- stan(
  model_code = model_string,
  data = stan_data,
  pars = c("beta0", "beta1", "sigma"),
  iter = 5000,
  warmup = 500,
  thin = 1,
  chains = 4,
  seed = 1
)

# Traceplot to check mixing (it's good)
traceplot(res)

# Summary object
summ = summary(res)$summary
summ


beta1_ci_95 <- summ["beta1", c("2.5%", "97.5%")]
beta1_ci_95
