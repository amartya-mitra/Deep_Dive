# Accepts hyper-parameters describing the data distribution and returns
# the tuple (features, labels)

from lib import *

def get_toy_data(n_samples, feature_dict, seed, add_noise=False):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  # print(f"Using seed: {seed}")

  # Generate binary labels with equal probability
  y = torch.randint(0, 2, (n_samples,)) * 2 - 1

  # Generate a core feature from a Gaussian distribution
  # The mean of the feature is y * mu and variance is 1
  feature_c = feature_dict['core_multiplier'] * (y * feature_dict['mu'] + feature_dict['sigma'] * torch.randn(n_samples))

  # Generate a new spurious feature to the existing setup
  flip_prob = torch.rand(n_samples)
  feature_s = torch.where(flip_prob < feature_dict['p'], y, -y).float()
  if True:
    feature_s += feature_dict['spurious_multiplier'] * torch.randn(n_samples)


  # Combine the original feature and the new feature into a single dataset
  features = torch.stack((feature_c, feature_s), dim=1)
  X = features

  # Add additional noise dimensions
  if add_noise:
    noise_dim = (torch.sign(feature_s) != torch.sign(y)).sum().item()
    noise_dim = int(noise_dim * feature_dict['noise_dampener'])
    print(f"Adding {noise_dim} noise dimensions")
    noise = np.random.randn(n_samples, noise_dim)
    # noise = noise / np.linalg.norm(noise, 2, 1, keepdims=True)
    noise = feature_dict['noise_multiplier'] * noise
    X = torch.cat([X, torch.tensor(noise)], 1)

  return X, y

def init_config(n_samples):
  # Fixed value for mu
  mu, sigma = 2.0, 0.5
  # Probability of not flipping
  # 0.5: Zero correlation b/w spurious feature and target
  # 1.0: Perfect correlation b/w spurious feature and target
  p = 0.98 # Normal: 0.98, LH: 0.5
  # Core feature strength
  core_multiplier = 0.01 # 0.9
  # Spurious feature strength
  spurious_multiplier = 0.1
  # seed
  seed = 2
  # add noise?
  add_noise = False
  # Noise dampener?
  noise_dampener = 1 # 10: Memorize, 1: Generalize
  # Noise strength
  noise_multiplier = 0.1

  feature_dict = {'mu': mu,
                  'sigma': sigma,
                  'p': p,
                  'spurious_multiplier': spurious_multiplier,
                  'core_multiplier': core_multiplier,
                  'noise_multiplier': noise_multiplier,
                  'noise_dampener': noise_dampener}

  return feature_dict, seed, add_noise