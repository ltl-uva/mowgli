import torch.nn as nn

from mowgli.models import (
    initialize_model,
    MowgliModel,
    ModularModel,
    UniversalModel,
    TransformerEncoder,
    TransformerDecoder
)
from mowgli.modules import Embeddings
from mowgli.constants import PAD_TOKEN
from mowgli.data import Vocabulary
from mowgli.helpers import ConfigurationError


def build_model(
    cfg,
    vocab,
    src_key,
    trg_key
) -> MowgliModel:
    """
    Build and initialize the model according to the configuration.
    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    model_type = cfg.get("type", "universal")

    if   model_type == "universal": builder = build_universal_model
    elif model_type == "modular":   builder = build_modular_model
    else:                           raise NotImplementedError

    model = builder(cfg, vocab, src_key, trg_key)

    # custom initialization of model parameters
    src_padding_idx = vocab[src_key[0]].s2i[PAD_TOKEN]
    trg_padding_idx = vocab[trg_key[0]].s2i[PAD_TOKEN]
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx, model_type)

    return model

def build_universal_model(
    cfg: dict,
    vocab: dict,
    src_key: str,
    trg_key: str,
) -> UniversalModel:
    assert ((src_key == ["shared"] and trg_key == ["shared"]) or len(src_key) == 1 and len(trg_key) == 1)
    src_padding_idx = vocab[src_key[0]].s2i[PAD_TOKEN]
    trg_padding_idx = vocab[trg_key[0]].s2i[PAD_TOKEN]

    src_embed = Embeddings(**cfg["encoder"]["embeddings"], vocab_size=len(vocab[src_key[0]]), padding_idx=src_padding_idx)
    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if (
            src_key == ["shared"] and trg_key == ["shared"]
            and
            vocab[src_key[0]].i2s == vocab[trg_key[0]].i2s
        ):
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError("Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(**cfg["decoder"]["embeddings"], vocab_size=len(vocab[trg_key[0]]), padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "transformer") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=src_embed.embedding_dim,
            emb_dropout=enc_emb_dropout
        )

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "transformer") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"],
            encoder=encoder,
            vocab_size=len(vocab[trg_key[0]]),
            emb_size=trg_embed.embedding_dim,
            emb_dropout=dec_emb_dropout
        )

    model = UniversalModel(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        trg_embed=trg_embed,
        vocab=vocab,
        src_key=src_key,
        trg_key=trg_key
    )

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    return model


def build_modular_model(cfg: dict, vocab: dict, src_key: list, trg_key: list) -> ModularModel:
    """
    Builds modular model according to the configuration. A modular model consists of separate encoders and decoders.
    """
    pad_idxs=[vocab[k].s2i[PAD_TOKEN] for k in src_key+trg_key]
    assert isinstance(src_key, list) and isinstance(trg_key, list)
    assert len(src_key) > 1 or len(trg_key) > 1
    assert "shared" not in src_key and "shared" not in trg_key
    assert all(pad_idx == pad_idxs[0] for pad_idx in pad_idxs)
    assert "encoder" in cfg or all([src in cfg for src in [f"encoder_{src}" for src in src_key]])
    assert "decoder" in cfg or all([trg in cfg for trg in [f"decoder_{trg}" for trg in trg_key]])

    encoder_keys = {l: "encoder{}".format(f"_{l}" if f"encoder_{l}" in cfg else "") for l in src_key}
    decoder_keys = {l: "decoder{}".format(f"_{l}" if f"decoder_{l}" in cfg else "") for l in trg_key}

    # Build separate embeddings for all source languages
    src_embeds = nn.ModuleDict(
        {
            l: Embeddings(
                **cfg[encoder_keys[l]]["embeddings"],
                vocab_size=len(vocab[l]),
                padding_idx=pad_idxs[0]
            )
            for l in src_key
        }
    )

    # Build separate embeddings for all target languages
    trg_embeds = nn.ModuleDict(
        {
            l: Embeddings(
                **cfg[decoder_keys[l]]["embeddings"],
                vocab_size=len(vocab[l]),
                padding_idx=pad_idxs[0]
            )
            for l in trg_key
        }
    )

    # Build separate encoders for all source languages
    encoders = nn.ModuleDict(
        {
            l: TransformerEncoder(
                **cfg[encoder_keys[l]],
                emb_size=src_embeds[l].embedding_dim,
                emb_dropout=cfg[encoder_keys[l]]["embeddings"].get("dropout", cfg[encoder_keys[l]].get("dropout", 0.)),
            )
            for l in src_key
        }
    )

    # Build separate decoders for all source languages
    decoders = nn.ModuleDict(
        {
            l: TransformerDecoder(
                **cfg[decoder_keys[l]],
                vocab_size=len(vocab[l]),
                emb_size=trg_embeds[l].embedding_dim,
                emb_dropout=cfg[decoder_keys[l]]["embeddings"].get("dropout", cfg[decoder_keys[l]].get("dropout", 0.)),
            )
            for l in trg_key
        }
    )

    # Build complete modular model with separate encoders and separate decoders
    model = ModularModel(
        encoder=encoders,
        decoder=decoders,
        src_embed=src_embeds,
        trg_embed=trg_embeds,
        vocab=vocab,
        src_key=src_key,
        trg_key=trg_key
    )

    # Share cross-attention for all encoders and decoders (not necessary now since there is only 1 decoder)
    assert len(trg_key) == 1, "Only many-to-one is supported for now."

    # Tie embeddings not possible for modular models
    if cfg.get("tied_embeddings", False):
        raise ConfigurationError("Can not tie embeddings for modular models.")

    # Tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        for l in trg_key:
            assert trg_embeds[l].lut.weight.shape == model.decoder[l].output_layer.weight.shape, (
                "For `tied_softmax`, decoder `embedding_dim` and `hidden_size` must be equal."
            )
            model.decoder[l].output_layer.weight = trg_embeds[l].lut.weight

    return model
