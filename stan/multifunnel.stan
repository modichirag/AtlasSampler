data {
  int<lower=0> D;
}

parameters {
  vector[D] v; 
  matrix[D, 9] theta;
}

model {
      v ~ normal(0, 3);
      for (k in 1 : D) {
            theta[k] ~ normal(0, exp(v[k]/2));
      }
}
