import flax.linen as nn
import jax.numpy as jnp
from flax.linen import SimpleCell

# Dimensions
# batch_dim x location_dim x time_dim x feature_dim


class MLP(nn.Module):
    hidden_dims: list[int]
    output_dim: int = 1

    @nn.compact
    def __call__(self, x, training=False):
        for i in range(len(self.hidden_dims)):
            x = nn.Dense(features=self.hidden_dims[i])(x)
            x = nn.relu(x)
        return nn.Dense(features=self.output_dim)(x)


class Preprocess(nn.Module):
    n_hidden: int = 4
    n_locations: int = 1
    embedding_dim: int = 4
    output_dim: int = 1
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, training=False):
        n_locations = x.shape[-3]
        loc = nn.Embed(num_embeddings=n_locations, features=self.embedding_dim)(jnp.arange(n_locations))
        axis = -2
        loc = jnp.repeat(loc[..., None, :], x.shape[axis], axis=axis)
        if x.ndim == 4:
            loc = jnp.repeat(loc[None, ...], x.shape[0], axis=0)
        x = jnp.concatenate([x, loc], axis=-1)  # batch x embedding_dim
        layers = [self.n_hidden]
        for i in range(len(layers)):
            x = nn.Dense(features=layers[i])(x)
            x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.output_dim)(x)
        return nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)


class RNNModel(nn.Module):
    n_hidden: int = 4
    pre_hidden: int = 4
    n_locations: int = 1
    embedding_dim: int = 4
    output_dim: int = 1

    @nn.compact
    def __call__(self, x, training=False):
        dropout_rate = 0.2
        loc = nn.Embed(num_embeddings=self.n_locations, features=self.embedding_dim)(jnp.arange(self.n_locations))
        loc = jnp.repeat(loc[..., None, :], x.shape[1], axis=1)
        x = jnp.concatenate([x, loc], axis=-1)  # batch x embedding_dim
        layers = [4]
        for i in range(len(layers)):
            x = nn.Dense(features=layers[i])(x)
            x = nn.relu(x)
        x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.pre_hidden)(x)
        x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        gru = nn.SimpleCell(features=self.n_hidden)
        x = nn.RNN(gru)(x)
        x = nn.Dense(features=6)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


class ARModel(nn.Module):
    n_locations: int = 1
    output_dim: int = 2
    dropout_rate: float = 0.2
    pre_hidden: int = 2
    n_hidden: int = 4

    @nn.compact
    def __call__(self, x, y, training=False):
        x = Preprocess(n_locations=self.n_locations, output_dim=self.pre_hidden, dropout_rate=self.dropout_rate)(
            x, training=training
        )
        n_y = y.shape[-1]
        prev_x = jnp.concatenate([y[..., None], x[..., 1 : n_y + 1, :]], axis=-1)
        states = nn.RNN(SimpleCell(features=self.n_hidden))(prev_x)
        new_states = nn.RNN(SimpleCell(features=self.n_hidden))(x[..., n_y + 1 :, :], initial_carry=states[..., -1, :])
        x = jnp.concatenate([states, new_states], axis=1)
        x = nn.Dense(features=6)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


class ARAdder(nn.Module):
    @nn.compact
    def __call__(self, x, y):
        n_y = y.shape[-1]
        return jnp.concatenate([y[..., None], x[..., 1 : n_y + 1, :]], axis=-1)


class MultiValueARAdder(nn.Module):
    @nn.compact
    def __call__(self, x, y):
        n_y = y.shape[-1]
        collected_y = jnp.moveaxis(nn.Dense(features=y.shape[0])(jnp.moveaxis(y, 0, -1)), -1, 0)
        collected_y = nn.relu(collected_y)
        return jnp.concatenate([collected_y[..., None], y[..., None], x[..., 1 : n_y + 1, :]], axis=-1)


class ARModel2(nn.Module):
    preprocess: nn.Module
    cell_pre: nn.RNNCellBase
    cell_post: nn.RNNCellBase
    ar_adder: ARAdder = ARAdder()
    output_dim: int = 2

    @nn.compact
    def __call__(self, x, y, training=False):
        x = self.preprocess(x, training=training)
        n_y = y.shape[-1]
        prev_x = self.ar_adder(x, y)
        states = nn.RNN(self.cell_pre)(prev_x)
        new_states = nn.RNN(self.cell_post)(x[..., n_y + 1 :, :], initial_carry=states[..., -1, :])
        x = jnp.concatenate([states, new_states], axis=-2)
        x = nn.Dense(features=6)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


class CompositeModel(nn.Module):
    preprocess: Preprocess


model_makers = {
    "base": lambda n_locations: ARModel2(
        Preprocess(n_locations=n_locations, output_dim=2, dropout_rate=0.2),
        SimpleCell(features=4),
        SimpleCell(features=4),
    ),
    "multi_value": lambda n_locations: ARModel2(
        Preprocess(n_locations=n_locations, output_dim=2, dropout_rate=0.2),
        nn.SimpleCell(features=4),
        nn.SimpleCell(features=4),
        ar_adder=MultiValueARAdder(),
    ),
}
