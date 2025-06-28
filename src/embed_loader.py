import os
import torch
import safetensors.torch as st

def inject_sdxl_embedding(pipe, safetensor_path: str, token: str = "sdxl_cyberrealistic_simpleneg"):
    """
    Injects a custom SDXL embedding (A1111 format) into the given pipeline.

    - pipe: the StableDiffusionXLPipeline (or similar) instance
    - safetensor_path: path to .safetensors file containing 'clip_g' and 'clip_l'
    - token: the placeholder token name to add to tokenizer
    """
    # 1. Load tensor weights from safetensors file
    weights = st.load_file(safetensor_path)
    emb_g = weights.get("clip_g")  # candidate for encoder1 or encoder2
    emb_l = weights.get("clip_l")
    if emb_g is None or emb_l is None:
        raise ValueError("Embedding file must contain 'clip_g' and 'clip_l' tensors")
    # Приведение к float32 для совместимости
    emb_g = emb_g.to(dtype=torch.float32)
    emb_l = emb_l.to(dtype=torch.float32)

    # 2. Add new token to both tokenizers
    tokenizers = [pipe.tokenizer]
    if hasattr(pipe, "tokenizer_2"):
        tokenizers.append(pipe.tokenizer_2)
    for tok in tokenizers:
        if token not in tok.get_vocab():
            tok.add_tokens(token)
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    print(f"[embedding] Token '{token}' added with ID={token_id}")

    # 3. Determine which embedding matches which encoder by hidden size
    encoders = [(pipe.text_encoder, emb_g, "clip_g"),]
    if hasattr(pipe, "text_encoder_2"):
        encoders.append((pipe.text_encoder_2, emb_l, "clip_l"))

    for encoder, emb, name in encoders:
        hidden_dim = encoder.text_model.embeddings.token_embedding.weight.size(1)
        # ensure emb 2D
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        if emb.size(1) != hidden_dim:
            # try swap roles if the other emb fits
            other = emb_l if emb is emb_g else emb_g
            if other.ndim == 1:
                other = other.unsqueeze(0)
            if other.size(1) == hidden_dim:
                print(f"[embedding] Swapping embeddings for {name}")
                emb = other
            else:
                raise RuntimeError(
                    f"Embedding dimension mismatch for {name}: file {emb.size(1)} vs model {hidden_dim}"
                )
        # extend
        old_w = encoder.text_model.embeddings.token_embedding.weight.data
        new_w = torch.cat([old_w, emb], dim=0)
        encoder.text_model.embeddings.token_embedding.weight = torch.nn.Parameter(new_w)
        print(f"[embedding] Embedded '{name}' applied to encoder hidden {hidden_dim}")

    print(f"[embedding] Weights from '{os.path.basename(safetensor_path)}' successfully injected")
