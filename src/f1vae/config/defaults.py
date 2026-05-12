from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefaults:
    latent_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    max_length: int
    hidden_dim: int | None = None
    embedding_dim: int | None = None


DEFAULTS = {
    "char_vae": ModelDefaults(
        latent_dim=32,
        hidden_dim=256,
        embedding_dim=32,
        batch_size=32,
        epochs=1000,
        learning_rate=1e-3,
        max_length=182,
    ),
    "grammar_vae": ModelDefaults(
        latent_dim=32,
        hidden_dim=256,
        embedding_dim=32,
        batch_size=32,
        epochs=1000,
        learning_rate=1e-3,
        max_length=500,
    ),
    "grammar_vae_masked": ModelDefaults(
        latent_dim=32,
        batch_size=32,
        epochs=2000,
        learning_rate=1e-3,
        max_length=500,
    ),
    "tree_vae": ModelDefaults(
        latent_dim=32,
        hidden_dim=256,
        embedding_dim=32,
        batch_size=32,
        epochs=1000,
        learning_rate=1e-3,
        max_length=500,
    ),
    "tree_vae_masked": ModelDefaults(
        latent_dim=128,
        hidden_dim=512,
        embedding_dim=64,
        batch_size=32,
        epochs=1000,
        learning_rate=3e-3,
        max_length=500,
    ),
    "tree_vae_masked_lhs": ModelDefaults(
        latent_dim=128,
        hidden_dim=512,
        embedding_dim=64,
        batch_size=32,
        epochs=1000,
        learning_rate=1e-3,
        max_length=500,
    ),
    "tree_vae_masked_lhs_depth": ModelDefaults(
        latent_dim=128,
        hidden_dim=512,
        embedding_dim=64,
        batch_size=32,
        epochs=1000,
        learning_rate=1e-3,
        max_length=500,
    ),
    "transformer_vae": ModelDefaults(
        latent_dim=32,
        hidden_dim=256,
        embedding_dim=64,
        batch_size=32,
        epochs=1000,
        learning_rate=1e-3,
        max_length=500,
    ),
    "vq_grammar_ae": ModelDefaults(
        latent_dim=32,
        hidden_dim=256,
        embedding_dim=32,
        batch_size=32,
        epochs=1000,
        learning_rate=1e-3,
        max_length=500,
    ),
}
