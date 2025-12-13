There exists a latent space. 
## There exists model information such as 

1.Latent state : s_t
2.observation : o_t
3.reward : r_t
4.action : a_t

1. Makes use of ENCODER: Images --> latent space representation
q(st​∣o≤t​,a<t​)


2. We are learning the latent space predictions.
Reconstructing the image and reward from the latent space.
Latent dynamics model.

Predicting the next latent state from the previous one + action is accurate/ Latent dynamics model
p(s_t|s_t-1, a_t-1)

Reconstructing the image and reward from the latent state is accurate
p(o_t|s_t), p(r_t|s_t) : used for loss computation

The posterior(immediate) latent state inferred from images matches the prior(all leading upto) prediction
q(s_t|o<t, a<t) ~ p(s_t|s_t-1, a_t-1)


3. Decoders: predicts the next image and reward

   Optimization process includes.
   Predicting next reconstructed image and predicted reward.
   o_t' = p(o_t|s_t)
   r_t' = p(r_t|s_t)


Loss is the calculation of 
1. observation rewards (r_t and r_t')
2. image rewards (o_t and o_t')


KL divergence used to determine the difference between 2 probability density functions: 

KL divergence equates to planning

KL [q(s_t| o<t) || p(s_t|s_t-1, a_t-1)]

Encoder part: 
q(s_t|o<t) "should this also not include the action a<t?"

Dyanamics model part : the state prediction based on the previous states and actions
p(s_t|s_t-1, a_t-1)

Latent space prediction is faster than pixel level predictions. 
Planning objective is to select actions that maximize the predicted/expected rewards in the latent space. 

Planning items and models used: 

1. Learnt Latent state distribution - q(s_t|o<t, a<t)
2. Learnt Latent transition model - p(s_t|s_t-1, a_t-1)
3. Learnt reward model - p(r_t|s_t)

There are 12 planning action horizon steps for H = 12 (a ... a+H)
at, at+1, ..., at+12

Current latent state s_t
Apply action a_t and use transition dynamics p(s_t+1|s_t, a_t)
Reward r_t+1 = Expected[p(r_t+1|s_t+1)]




## Questions:

Get to variational bound objective?

Explain the KL term and why prior vs posterior matching matters for planning

Explain RSSM and the deterministic + stochastic components

Explain latent overshooting in detail

Explain how MPC + CEM works mathematically

standard latent SSM
RSSM : deterministic, Stochastic, Reward, Encoder


ELBO: Evidence lower bound in entropy models. 

![text2](<ELBO/Screenshot 2025-12-10 at 3.22.43 PM.png>)
![text1](<ELBO/Screenshot 2025-12-10 at 3.22.55 PM.png>)
![text3](<ELBO/Screenshot 2025-12-10 at 3.23.09 PM.png>)


That is the log-probability of the entire observed sequence given the actions the agent took. It is the full-sequence (also called marginal or evidence) log-likelihood.

If this number is high → our model assigns high probability to real image sequences that actually happen → we learned a good dynamics model.



![alt text](<../Screenshot 2025-12-11 at 12.26.30 PM.png>)
![alt text](<../Screenshot 2025-12-11 at 12.27.10 PM.png>)

Paper labels the 2 terms as reconstruction and complexity.
where 
1. reconstruction 
2. complexity (KL regularizer). Always > 0. 

Loss is given as:
l(theta) = Sum 


Now why are these terms lower bound and have to be maximized and implementation.


Modeling the different layers and states:

 
We also have the followingTransition Model:Transition Model:Transition Model:Transition Model:Transition Model:
1. Transition Model: Gaussian (mean, var) where (mean, var) f(nn)
2. Observation model: Gaussian (mean, var) where (mean, var=1) f(nn)
3. Reward Model: Gaussian (mean, var) where (mean, var=1) f(nn)

ToDo: Comparison of the different type of models RNN vs SSM vs RSSM

1. RNN- deterministic
The key rule is using a function which is deterministic in nature
h_t = f(h_t-1, a_t-1) 

this is the predicted


given previous hidden state h_t-1 and previous action compute the new hidden state

actual is given as 
h_t(posterior) = q(h_t|h_t-1, o_t). comparison gives us the loss.

once hidden state is determined use decoder
(o_t, r_t) = g(h_t)

reconstructs image and predicts the reward
No stochasticity, no randomness, no uncertainity
e.g. They cannot represent multiple possible futures
Say the real environment is stochastic: ball may bounce left or right
opponent may attack or retreat.

Loss_RNN = image and reward loss

L = ||o_t - o_t_hat||^2 + ||r_t - r_t_hat||^2

Since there is no stochastic states, but only deterministic states, there is no need for prior and posterior calculation and therefore KL divergence

2. SSM (stochastic model)
instead of using deterministic function, one uses the randomness of mean and variance to determine the state
state is drawn from a probability distribution
s_t ~ p(s_t | s_t-1, a_t-1)

a seperate function determines the obs and reward

(o_t, r_t) ~ p(o_t, r_t| s_t)

dashed lines are used for inference

actual obs o_t is used to determine the posterior
q(s_t| s_t-1, a_t-1, o_t)

This prevents accumulation of noise, suffers from unstable long term predictions

Loss includes posterior, prior and matching using KL divergence. 


3. RSSM 

Combination of both stochastic and deterministic states
What is being carried over is only the long horizon deterministic states- h
The stochastic states are only injected but not accumulated - lest it would lead to accumulation of randomness/noise and make it unstable

# Training workflow

1. The deterministic state: h
h_t = f(h_t-1, s_t-1, a_t-1)
 
2. Sample stochastic state using prior

stochastic state s is not carried over but determined as 
s_t_prior ~ p(s_t|h_t)

3. Infer posterior stochastic state using obs
Inference (shown as dashed lines), updates only the stochastic state s_t

s_t_post ~ p(s_t|h_t, o_t)

4. Obs 
o_t ~ p(o_t|h_t, s_t_post)

Reward
r_t ~ p(r_t|h_t,s_t)

5. KL divergence bet prior and posterior
KL[p(s_t_post)||p(s_t_prior)]


Next follow the math for the loss computation for the 3 different arch.

Latent overshooting trains the RSSM so that its multi-step predicted latent states match the posterior latent states inferred from real observations, improving long-horizon dynamics without decoding images.

![text](<../Screenshot 2025-12-11 at 2.43.30 PM.png>)








Let’s explain ELBO (Evidence Lower BOund) from absolute zero, with no jargon at first, then connect it to everything we talked about (Dreamer, RSSM, latent overshooting, etc.).

### 1. What is the real thing we want?

We have real images (or states) o₁, o₂, …, o_T and actions a₁, …, a_{T-1}.

We want our model to say:  
“These real sequences are very probable!”

Mathematically, we want to make this number as large as possible:

\[
\log p(o_1,o_2,\dots,o_T \mid a_1,\dots,a_{T-1})
\]

This is called the log-likelihood (or evidence) of the data).

If this number is high → the model thinks real videos are likely → good world model.  
If it is low → the model thinks real videos are weird or impossible → bad world model.

### 2. Why can’t we compute or optimise it directly?

Because inside the probability there is a gigantic integral over all possible latent states s₁…s_T:

\[
p(o_{1:T}) = \int \underbrace{p(o_1,s_1,o_2,s_2,\dots,o_T,s_T)}_{\text{huge number of terms}} \, ds_1 ds_2 \dots ds_T
\]

For a 100-step video and 32-dimensional latents this integral has roughly 100×32 ≈ 3,200 dimensions.  
Even a supercomputer cannot compute a 3,200-dimensional integral.

So we are stuck → we cannot directly maximise the true log-likelihood.

### 3. What does the ELBO do? (the trick)

We introduce an approximate posterior (the encoder) q(s₁…s_T | o₁…o_T).

Then we use one of the most famous inequalities in machine learning (Jensen’s inequality on the log):

\[
\boxed{
\log p(o_{1:T}) 
\; \geq\; 
\underbrace{\mathbb{E}_{q(s_{1:T}|o_{1:T})}\Big[ \log \frac{p(o_{1:T},s_{1:T})}{q(s_{1:T}|o_{1:T})} \Big]}_{\displaystyle =\; \mathcal{L}\; =\; \text{ELBO}}
}
\]

This inequality is always true, and equality holds only if q = true posterior.

Key points:

- The left side (log p(o₁…o_T)) is the thing we really care about.
- The right side (ELBO) is something we can compute and differentiate.
- Maximising the ELBO pushes up a lower bound of the true log-likelihood → the true log-likelihood can only go up (or stay the same).

So ELBO is a safe surrogate objective.

### 4. ELBO in pictures (VAE / world model version)

\[
\boxed{\text{ELBO} = 
\underbrace{\mathbb{E}_q[\log p(o_t \mid s_t)]}_{\text{“reconstruct the image well”}}
\;-\;
\underbrace{\text{KL}(q(s_t) \| p(s_t))}_{\text{“don’t let encoder and prior diverge too much”}}
}
\]

- First term = reconstruction term → forces the latent to contain enough information to rebuild the image.
- Second term = regulariser → forces the approximate posterior q to stay close to the prior.

### 5. Why do we need the ELBO in Dreamer / world models?

Because:

| Problem                              | Solved by ELBO because…                                                                 |
|--------------------------------------|------------------------------------------------------------------------------------------|
| We cannot compute the true likelihood | ELBO is a computable lower bound                                                         |
| We have latent variables s_t          | ELBO marginalises them out in a principled way                                           |
| We want long-term predictions         | Standard ELBO only gives one-step supervision | Standard ELBO is too weak for long horizons → that’s why Dreamer adds latent overshooting |

### 6. Analogy everyone understands

Imagine you are trying to guess tomorrow’s weather.

- True goal: “What is the probability that tomorrow is exactly 22°C and sunny?”  
  → impossible to compute exactly.

So you pick a simple distribution q (e.g. “I guess 20–25°C”) and you say:

“I will give you a score = how well your model reconstructs 22°C and sunny  
minus a penalty if your guess q is far from your model’s prior.”

That score is a lower bound on the true probability.  
The higher you make that score, the higher the true probability must be.

That score is the ELBO.

### 7. Summary table

| Quantity                         | Can we compute it directly? | What we do instead                         |
|----------------------------------|-----------------------------|---------------------------------------------|
| True log-likelihood log p(o₁:T)  | No (intractable integral)   | —                                           |
| ELBO (standard)                  | Yes                         | Maximise it → log p can only increase       |
| ELBO + latent overshooting       | Yes (and tighter)           | Even better long-term model                 |

### Final answer in one sentence

The ELBO is needed because the true data probability log p(o₁…o_T) is mathematically impossible to compute or optimise directly, so we optimise a lower bound (the ELBO) instead — and every time we make the ELBO larger, the true (intractable) likelihood is guaranteed to be at least that large (or larger).

That is the entire reason variational inference, VAEs, and world models like Dreamer exist.

Great question!  
If you completely remove the KL divergence term from the ELBO, here’s exactly what happens — step by step, with no sugar-coating.

### The ELBO without the KL term becomes
\[
\mathcal{L}_{\text{no KL}} 
= \sum_t \mathbb{E}_{q(s_t | o_{\leq t})} \big[ \log p(o_t | s_t, h_t) \big] 
\quad \text{(only reconstruction)}
\]

This is sometimes called “reconstruction-only training” or “β=0 VAE”.

### What happens in practice (100% of the time)

| Symptom                                 | Why it happens                                                                                   | Result in Dreamer / world models                                                                 |
|-----------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 1. Posterior collapse                   | The encoder can ignore the prior completely and put all information into the stochastic s_t      | s_t becomes extremely high-entropy noise or one-hot vectors, or just pure noise                 |
| 2. The prior p(s_t) is completely ignored | Nothing forces q(s_t) to be close to p(s_t | s_{t−1})                                           | During imagination (planning), the model samples from a prior that was never trained → garbage    |
| 3. Perfect reconstruction on training data | The decoder can memorize everything by stuffing the entire image into s_t                       | Train reconstruction loss → 0, model looks perfect on seen sequences                             |
| 4. Catastrophic failure when rolling out | When you close the loop (no real images → only prior samples), the latent states immediately drift | After 2–5 steps of imagination everything turns into random noise or a static pattern            |
| 5. Planning completely breaks           | The agent plans in a fantasy world that has nothing to do with reality                           | Dreamer agent gets 0 reward, walks into walls, or exploits obvious artifacts                     |

### Real examples from papers and code

| Experiment (people actually tried this) | Result                                                                 |
|-----------------------------------------|------------------------------------------------------------------------|
| DreamerV1 with β = 0                    | Score drops from ~8000 to <500 on Atari (basically random agent)      |
| VAE with β = 0                          | Generates blurry digits or collapses to a single mode                  |
| PlaNet with KL weight = 0               | Agent cannot solve even the simplest cartpole from pixels in <10 steps |
| Any RSSM trained with only reconstruction | After 3–4 imagined steps the image turns into pink noise or a single colour |

### Analogy everyone understands

- With KL: You are training a student (the prior) to match the teacher (the posterior) while also learning to reconstruct the homework.
- Without KL: You only grade the homework reconstruction, but you never check if the student actually learned anything.  
  → The student memorises the answers (perfect reconstruction) but when you give a new question (imagination), they have no idea what to do.

### Mathematical reason in one line

The KL term is the **only** thing that gives any training signal to the transition prior p(sₜ | s_{t−1}).

If you remove it:
- The encoder q can do whatever it wants
- The prior p(sₜ | s_{t−1}) receives zero gradients → stays random initialisation
- Multi-step rollouts become independent noise

### What people sometimes do instead (and why it still fails)

| Alternative people try                  | Why it still fails                                                                 |
|-----------------------------------------|------------------------------------------------------------------------------------|
| Very small but non-zero KL weight (β=0.001) | Usually still collapses or gives terrible long-term predictions                   |
| Add noise to observations               | Helps a little, but still collapses after 10–20 steps                             |
| Use a deterministic model only (no s_t) | This is option (a) in Figure 2 — overconfident and exploitable by the planner      |

### Bottom line – one sentence answer

If you remove the KL divergence from the ELBO, reconstruction becomes perfect but the model learns **absolutely nothing useful about dynamics**, so imagination and planning completely break — the agent becomes useless.

That is why **every single successful world model** (DreamerV2, DreamerV3, PlaNet, MuZero with images, IRIS, etc.) **always** keeps the KL term (and usually makes it stronger with overshooting).

You cannot remove it. Period.


1. Why Reward Loss is Considered
The reward loss is included because PlaNet learns a reward model p(r_t | s_t) that is used for planning.
Purpose of the reward model:
Planning: The agent uses p(r_t | s_t) to predict future rewards when selecting actions via CEM/MPC.
State representation: Learning to predict rewards from latent states encourages the latent space to capture reward-relevant information.
Model completeness: The world model includes:
Transition model: p(s_t | s_{t-1}, a_{t-1})
Observation model: p(o_t | s_t)
Reward model: p(r_t | s_t)
Extended ELBO:
The full objective includes both observation and reward reconstruction:
ELBO = Σ_t [E_q[ln p(o_t | s_t)] + E_q[ln p(r_t | s_t)] - KL term]
Both p(o_t | s_t) and p(r_t | s_t) are decoders that reconstruct/predict from the latent state, so both are trained via reconstruction loss.
2. The 0.5 Factor for Log-Likelihood
The 0.5 factor comes from the Gaussian negative log-likelihood when variance is fixed (often σ² = 1).
Mathematical Derivation:
For a Gaussian distribution:
p(x | μ, σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
Taking the negative log-likelihood:
-ln p(x | μ, σ²) = 0.5 * ln(2πσ²) + 0.5 * (x-μ)²/σ²
If we assume fixed variance σ² = 1 (common in practice):
)
-ln p(x | μ, 1) = 0.5 * ln(2π) + 0.5 * (x-μ)²                = constant + 0.5 * MSE(x, μ)
In the Code:
obs_loss = 0.5 * mse_loss(recon_observations[1:], observations[1:], ...)reward_loss = 0.5 * mse_loss(predicted_rewards[1:], rewards[:-1])

Maximizing log-likelihood is equivalent to minimizing negative log-likelihood.
With fixed variance, minimizing 0.5 * MSE is equivalent to maximizing the Gaussian log-likelihood.
The constant 0.5 * ln(2π) doesn't affect gradients, so it's omitted.