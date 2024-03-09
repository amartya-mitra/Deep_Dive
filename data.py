# Accepts hyper-parameters describing the data distribution and returns
# the tuple (features, labels)

from lib import *

# Hidden manifold model
def get_HMM_data(config):
    
    n = config['n']
    dl = config['dim_latent']
    da = config['dim_ambient']
    p = config['p']
    noise = config['noise']

    feature_generator = torch.randn((dl, da))
    latent_patterns = torch.bernoulli(p * torch.ones((n, dl)))
    teacher_weights = torch.randn((dl, 1))

    # x = f(feature * latent) + noise
    x = config['nonlin'](
        torch.matmul(latent_patterns, feature_generator) /
        torch.sqrt(torch.tensor(dl)))
    x += noise * torch.randn((n, da))

    # ReLU activation on the target
    # y = latent * teacher_weights
    y = torch.mm(latent_patterns, teacher_weights).gt(0).long()
    y = 2 * y - 1

    data = {'tr': {
                'x': x[:n//2],
                'y': y[:n//2].view(-1),
                'latents': latent_patterns[:n//2]},
            'va': {
                'x': x[n//2:],
                'y': y[n//2:].view(-1),
                'latents': latent_patterns[n//2:]}}
    return data

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
  feature_s = feature_dict['sp_multiplier'] * (torch.where(flip_prob < feature_dict['p'], y, -y).float())
  if True:
    feature_s += feature_dict['sn_multiplier'] * torch.randn(n_samples)


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
  p = 0.98 # Normal: 0.98, LH: 0.5 # Changed here
  # Core feature strength
  # core_multiplier = 0.09 # 0.9 #0.01
  core_multiplier = 0.09 # 0.09 #0.01 # Changed here
  # Spurious feature strength
  sp_multiplier = 1
  # Spurious feature noise strength
  sn_multiplier = 0.1
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
                  'sp_multiplier': sp_multiplier,
                  'sn_multiplier': sn_multiplier,
                  'core_multiplier': core_multiplier,
                  'noise_multiplier': noise_multiplier,
                  'noise_dampener': noise_dampener}

  feature_dict['sigma'] = 0.2
  feature_dict['p'] = 0.9
  feature_dict['core_multiplier'] = 0.25
  feature_dict['sp_multiplier'] = 6
  feature_dict['sn_multiplier'] = 0.5

  return feature_dict, seed, add_noise

def input_rotation(angle, input, feature_1, feature_2):
  #Perform rotation

  # Define the rotation angle θ in radians (e.g., 45 degrees)
  theta = math.radians(angle)

  # Construct the rotation matrix for 2D rotation
  cos_theta, sin_theta = math.cos(theta), math.sin(theta)
  rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                  [sin_theta, cos_theta]])

  # Extract the first and third columns to apply the rotation
  rotated_X = input.clone()
  columns_to_rotate = rotated_X[:, [feature_1, feature_2]]

  # Apply the rotation to the first and third columns
  rotated_columns = columns_to_rotate @ rotation_matrix

  # Replace the original first and third columns with the rotated ones
  rotated_X[:, feature_1] = rotated_columns[:, 0]
  rotated_X[:, feature_2] = rotated_columns[:, 1]

  return rotated_X