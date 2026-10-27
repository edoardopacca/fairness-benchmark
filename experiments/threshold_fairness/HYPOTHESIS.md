# Scores, decisions, and the controlled diagnostic

Prepared before changing the benchmark or inspecting new test results, 27 October 2026.

For a probability score P and a continuous sensitive variable S:

1. Conditional mean parity means E[P|S=s]=c (almost everywhere). It controls a first
   moment, not the entire conditional distribution.
2. Full independence P ⟂ S means that the conditional distribution of P does not depend
   on S. It implies conditional mean parity and parity of every fixed-threshold decision.
3. Decision demographic parity at t means Pr(P≥t|S=s)=c_t. At one threshold this neither
   implies full independence nor constant conditional means. Parity at every threshold
   determines the conditional distribution and implies independence (up to null sets).

For example, decisions can be constant while group-specific scores vary entirely below
the threshold. Conversely, constant score means can hide different mass on either side.
Low estimates of dependence are not assertions of population independence.

## Analytic example specified for this experiment

Let S~Uniform(-1,1), q(s)=0.6+0.2s, B|S~Bernoulli(q(S)),
P*=0.5+0.4(B−q(S)), and Y|S,B~Bernoulli(P*).

Since q∈[0.4,0.8], the B=0 branch is 0.5−0.4q∈[0.18,0.34],
and the B=1 branch is 0.9−0.4q∈[0.58,0.74]. Both are valid probabilities.
E[P*|S]=0.5+0.4(E[B|S]−q)=0.5, and by iterated expectation E[Y|S]=0.5.
At threshold 0.5, D=B exactly, so Pr(D=1|S=s)=q(s), ranging from 0.4 to 0.8
(population endpoint supremum gap 0.4). The overall decision prevalence is 0.6.
The conditional variance of P* is 0.16q(s)(1−q(s)), which varies with s;
P* and S are therefore not independent. Polynomial score-level GeDI has zero
population coefficients for every finite degree because Cov(f(S),P*)=0.

The Bayes accuracy at 0.5 is E[max(P*,1−P*)]=0.5+0.8E[q(1−q)]
=0.5+0.8(0.24−0.04/3)=0.6813333333. With three equal-width groups,
the population group decision rates are 0.4666667, 0.6, 0.7333333,
and their gap is 0.2666667; this differs from the endpoint supremum gap.

X=(S,B,Z1,Z2), with independent standard-normal nuisance Z. A neural network is
trained against sampled Y using BCE; P* is never supplied as a target or feature.
The analytic oracle is evaluated separately and always labelled as an oracle.
The existing quadratic/nonmonotonic tasks retain their original feature definitions
(S is not directly included there), so comparisons are within scenario.

`python -m experiments.threshold_fairness.oracle` verifies both conditional branches
algebraically on a dense grid and by Monte Carlo with 1,000,000 observations.
The construction is a controlled diagnostic and does not establish real-world discrimination.
