There exists a latent space. 
## There exists model information such as 

1. Latent state : $s_t$
2. observation : $o_t$
3. reward : $r_t$
4. action : $a_t$

1. Makes use of ENCODER: Images --> latent space representation
$q(s_t \mid o_{\leq t}, a_{< t})$


2. We are learning the latent space predictions.
Reconstructing the image and reward from the latent space.
Latent dynamics model.

Predicting the next latent state from the previous one + action is accurate/ Latent dynamics model
$p(s_t \mid s_{t-1}, a_{t-1})$

Reconstructing the image and reward from the latent state is accurate
$p(o_t \mid s_t)$, $p(r_t \mid s_t)$ : used for loss computation

The posterior(immediate) latent state inferred from images matches the prior(all leading upto) prediction
$q(s_t \mid o_{< t}, a_{< t}) \sim p(s_t \mid s_{t-1}, a_{t-1})$


3. Decoders: predicts the next image and reward

   Optimization process includes.
   Predicting next reconstructed image and predicted reward.
   $o_t' = p(o_t \mid s_t)$
   $r_t' = p(r_t \mid s_t)$


Loss is the calculation of 
1. observation rewards ($r_t$ and $r_t'$)
2. image rewards ($o_t$ and $o_t'$)


KL divergence used to determine the difference between 2 probability density functions: 

KL divergence equates to planning

$$\text{KL}[q(s_t \mid o_{< t}) \parallel p(s_t \mid s_{t-1}, a_{t-1})]$$

Encoder part: 
$q(s_t \mid o_{< t})$ "should this also not include the action $a_{< t}$?"

Dyanamics model part : the state prediction based on the previous states and actions
$p(s_t \mid s_{t-1}, a_{t-1})$

Latent space prediction is faster than pixel level predictions. 
Planning objective is to select actions that maximize the predicted/expected rewards in the latent space. 

Planning items and models used: 

1. Learnt Latent state distribution - $q(s_t \mid o_{< t}, a_{< t})$
2. Learnt Latent transition model - $p(s_t \mid s_{t-1}, a_{t-1})$
3. Learnt reward model - $p(r_t \mid s_t)$

There are 12 planning action horizon steps for $H = 12$ ($a$ ... $a+H$)
$a_t, a_{t+1}, \ldots, a_{t+12}$

Current latent state $s_t$
Apply action $a_t$ and use transition dynamics $p(s_{t+1} \mid s_t, a_t)$
Reward $r_{t+1} = \mathbb{E}[p(r_{t+1} \mid s_{t+1})]$




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

Paper labels the 2 terms as reconstruction and complexity.
where 
1. reconstruction 
2. complexity (KL regularizer). Always $> 0$. 

Loss is given as:
$l(\theta) = \sum$ 


Now why are these terms lower bound and have to be maximized and implementation.


Modeling the different layers and states:

 
We also have the followingTransition Model:Transition Model:Transition Model:Transition Model:Transition Model:
1. Transition Model: Gaussian (mean, var) where (mean, var) f(nn)
2. Observation model: Gaussian (mean, var) where (mean, var=1) f(nn)
3. Reward Model: Gaussian (mean, var) where (mean, var=1) f(nn)

ToDo: Comparison of the different type of models RNN vs SSM vs RSSM

1. RNN- deterministic
The key rule is using a function which is deterministic in nature
$h_t = f(h_{t-1}, a_{t-1})$ 

this is the predicted


given previous hidden state $h_{t-1}$ and previous action compute the new hidden state

actual is given as 
$h_t$(posterior) $= q(h_t \mid h_{t-1}, o_t)$. comparison gives us the loss.

once hidden state is determined use decoder
$(o_t, r_t) = g(h_t)$

reconstructs image and predicts the reward
No stochasticity, no randomness, no uncertainity
e.g. They cannot represent multiple possible futures
Say the real environment is stochastic: ball may bounce left or right
opponent may attack or retreat.

Loss_RNN = image and reward loss

$$L = \|o_t - \hat{o_t}\|^2 + \|r_t - \hat{r_t}\|^2$$

Since there is no stochastic states, but only deterministic states, there is no need for prior and posterior calculation and therefore KL divergence

2. SSM (stochastic model)
instead of using deterministic function, one uses the randomness of mean and variance to determine the state
state is drawn from a probability distribution
$s_t \sim p(s_t \mid s_{t-1}, a_{t-1})$

a seperate function determines the obs and reward

$(o_t, r_t) \sim p(o_t, r_t \mid s_t)$

dashed lines are used for inference

actual obs $o_t$ is used to determine the posterior
$q(s_t \mid s_{t-1}, a_{t-1}, o_t)$

This prevents accumulation of noise, suffers from unstable long term predictions

Loss includes posterior, prior and matching using KL divergence. 


3. RSSM 

Combination of both stochastic and deterministic states
What is being carried over is only the long horizon deterministic states- h
The stochastic states are only injected but not accumulated - lest it would lead to accumulation of randomness/noise and make it unstable

# Training workflow

1. The deterministic state: $h$
$$h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$$
 
2. Sample stochastic state using prior

stochastic state $s$ is not carried over but determined as 
$$s_t^{\text{prior}} \sim p(s_t \mid h_t)$$

3. Infer posterior stochastic state using obs
Inference (shown as dashed lines), updates only the stochastic state $s_t$

$$s_t^{\text{post}} \sim p(s_t \mid h_t, o_t)$$

4. Obs 
$$o_t \sim p(o_t \mid h_t, s_t^{\text{post}})$$

Reward
$$r_t \sim p(r_t \mid h_t, s_t)$$

5. KL divergence bet prior and posterior
$$\text{KL}[p(s_t^{\text{post}}) \parallel p(s_t^{\text{prior}})]$$


## How CEM Works (Simple Steps)

### Algorithm

1. **Initialize a sampling distribution**
   - Typically a Gaussian: $x \sim \mathcal{N}(\mu, \Sigma)$

2. **Sample N candidates**

3. **Evaluate objective function** $f(x)$

4. **Select the top k samples**
   - These are the elite set

5. **Update the distribution so it matches elites**
   - New $\mu$ = mean of elite samples
   - New $\Sigma$ = covariance of elite samples

6. **Repeat until convergence**

### Pseudocode

```python
μ, Σ = initialize

for iteration in range(T):
    x_samples = sample N points from N(μ, Σ)
    scores = evaluate f(x_samples)
    
    elite = top k samples
    μ = mean(elite)
    Σ = cov(elite)
```