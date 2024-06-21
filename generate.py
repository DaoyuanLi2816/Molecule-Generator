import torch
from rdkit import Chem
import torch.nn.functional as F
import warnings
from rdkit import RDLogger
from VAE import BetaTCVAE, SimpleTokenizer

def generate_nearby_smiles(model_path, smiles, tokenizer, max_len, num_samples, device, temperature=1.0, distance_multiplier=0.5):
    """
    Generate nearby SMILES strings by perturbing the latent space representation.

    Args:
        model_path (str): Path to the saved model.
        smiles (str): Input SMILES string.
        tokenizer (SimpleTokenizer): Tokenizer object to tokenize SMILES.
        max_len (int): Maximum length of the tokenized sequences.
        num_samples (int): Number of samples to generate.
        device (torch.device): Device to run the model on.
        temperature (float): Temperature for sampling.
        distance_multiplier (float): Multiplier to adjust the distance in latent space.

    Returns:
        list of str: List of generated SMILES strings. 
    """
    # Load the model
    vocab_size = 6
    embedding_dim = 16
    hidden_dim = 64
    latent_dim = 16
    nhead = 4
    num_layers = 2
    pad_idx = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BetaTCVAE(vocab_size, embedding_dim, hidden_dim, latent_dim, nhead, num_layers, pad_idx, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tokenized_smiles = tokenizer.tokenize(smiles, max_len).unsqueeze(0).to(device)
    embedded = model.embedding(tokenized_smiles)
    encoded = model.encoder(embedded)
    mu = model.mu(encoded)
    log_var = model.log_var(encoded)

    generated_smiles = []
    for i in range(num_samples):
        random_direction = torch.randn_like(mu)
        random_direction /= torch.norm(random_direction)
        z = mu + distance_multiplier * (i + 1) * random_direction

        decoded = model.decoder(z, encoded)
        out = model.fc_out(decoded)
        out = F.softmax(out / temperature, dim=-1)

        generated_smiles_idx = torch.multinomial(out.squeeze(0), 1).cpu().numpy().flatten()
        generated_smiles_str = ''.join(
            [tokenizer.idx_to_char[min(int(idx), 4)] for idx in generated_smiles_idx if idx != tokenizer.pad_idx])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(generated_smiles_str)

        if mol is not None:
            generated_smiles.append(generated_smiles_str)

    return list(set(generated_smiles))


if __name__ == "__main__":
    # Example usage
    input_smiles = 'OC(CCCCCC)C(O)C(O)C(O)CCCCCCCCC'
    num_samples = 5000
    max_len = 172  # Set to the maximum sequence length used during training
    temperature = 2.0
    distance_multiplier = 0.5
    model_path = 'beta_tc_vae_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simple_tokenizer = SimpleTokenizer()
    generated_smiles = generate_nearby_smiles(model_path, input_smiles, simple_tokenizer, max_len, num_samples, device, temperature, distance_multiplier)

    print(f"Input SMILES: {input_smiles}")
    print(f"Generated SMILES:")
    for smiles in generated_smiles:
        print(smiles)
