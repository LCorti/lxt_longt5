import numpy as np
import torch
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

from pathlib import Path
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens


def print_tokens(tokens: torch.Tensor, logits: torch.Tensor = None):
    if logits is not None:
        # print the tokens and their logits
        for token, logit in zip(tokens, logits):
            print(f"Token: {token}, Logit: {logit.item()}")
    else:
        for token in tokens:
            print(f"Token: {token}")


def save_heatmap(pos, ref_tokens: list, relevance: list, model_name: str):
    tokens = clean_tokens(ref_tokens)
    # Removing up </s> tokens from the model
    if "</s>" in tokens:
        tokens = tokens[:-1]
        relevance = relevance[:-1]

    pdf_heatmap(
        tokens,
        relevance,
        path=Path("output", model_name, f"map_token_{pos}.pdf"),
        backend="xelatex",
    )
    print("Heatmap saved.")


if __name__ == "__main__":
    print("=" * 40)
    model_name = "r1-qwen"
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # modify the Qwen2 module to compute LRP in the backward pass
    print(f"Patching {model_name}")
    monkey_patch(modeling_qwen2, verbose=True)
    print(f"Loading and setting up model: {model_path}")
    model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    max_input_length = config.max_position_embeddings  # 131072
    max_output_length = tokenizer.model_max_length  # 16384

    # model.train()
    # model.gradient_checkpointing_enable()
    # deactive gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False
    # model.eval()

    # Input data
    paper = "For developing a data-driven text rewriting algorithm for paraphrasing, it is essential to have a monolingual corpus of aligned paraphrased sentences.\nNews article headlines are a rich source of paraphrases; they tend to describe the same event in various different ways, and can easily be obtained from the web.\nWe compare two methods of aligning headlines to construct such an aligned corpus of paraphrases, one based on clustering, and the other on pairwise similarity-based matching.\nWe show that the latter performs best on the task of aligning paraphrastic headlines."
    prompt = f"Rewrite the following text in simpler language.\nText: {paper}"

    # get input embeddings so that we can compute gradients w.r.t. input embeddings
    input_ids = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(model.device)

    print("-" * 40)
    print(f"Input size: {input_ids.shape}")

    max_new_tokens = 256
    init_ids_len = input_ids.shape[-1]
    min_length = init_ids_len + 2
    max_length = max_new_tokens + init_ids_len
    max_length = max_length if max_length < max_output_length else max_output_length
    eos_token_id = tokenizer.eos_token_id
    end_cot_token_id = 151649  # </think> (to detect when the model stops thinking)
    decoding_strategy = "sample"  # greedy

    print("Output settings")
    print(f" - min_length: {min_length}")
    print(f" - max_length: {max_length}")
    print("-" * 40)

    logits_processor = LogitsProcessorList(
        [
            NoRepeatNGramLogitsProcessor(ngram_size=2),
            RepetitionPenaltyLogitsProcessor(penalty=1.2),
            MinLengthLogitsProcessor(min_length=min_length, eos_token_id=eos_token_id),
            ForcedEOSTokenLogitsProcessor(
                max_length=max_length, eos_token_id=eos_token_id
            ),
        ]
    )
    if decoding_strategy == "sample":
        logits_processor.append(TopPLogitsWarper(top_p=0.95))

    print("=" * 40)
    print("Running generation...")
    curr_len = 0
    relevance_scores = []
    next_token = None
    end_of_cot = False

    """
    - model is unstable
    - CoT can take quite long
    - attribution with respect to what? only input? include CoT?
    """

    while init_ids_len + curr_len <= max_length and next_token != eos_token_id:
        input_embeds = model.get_input_embeddings()(input_ids)
        output_logits = model(
            inputs_embeds=input_embeds.requires_grad_(),
            use_cache=False,
        ).logits
        raw_next_logits = output_logits[0, -1, :].cpu()
        input_ids_copy = input_ids.cpu()
        next_token_scores = logits_processor(
            input_ids_copy, raw_next_logits.unsqueeze(0)
        ).to(model.device)
        k = 1

        if decoding_strategy == "greedy":
            max_logits, max_indices = torch.max(next_token_scores, dim=-1)
            topk_logits, topk_indices = torch.topk(next_token_scores, k=k, dim=-1)
            max_raw_logits, _ = torch.max(output_logits[0, -1, :], dim=-1)
        elif decoding_strategy == "sample":
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            topk_indices = torch.multinomial(probs, num_samples=k)
            max_raw_logits = output_logits[0, -1, topk_indices[0].item()]
        else:
            print("error")

        # print(topk_indices)
        # topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices)
        # print("=" * 40)
        # print("## Generated tokens (top-{k}):")
        # print_tokens(topk_tokens, topk_logits)
        topk_token = tokenizer.convert_ids_to_tokens(topk_indices)
        print(">" * 50)
        print(
            f"{topk_indices}: {topk_token} -> {topk_indices[0].item() == end_cot_token_id}"
        )
        print(">" * 50)

        # If we have just found the end_cot_token_id, track it
        if not end_of_cot and topk_indices[0].item() == end_cot_token_id:
            end_of_cot = True
            print(">" * 50)
            print("END OF THINKING")
            print(">" * 50)

        # If we are already generating text, compute the relevance scores.
        if end_of_cot:
            # print("=" * 40)
            # print("Running backward pass...")
            max_raw_logits.backward()
            relevance = (
                (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
            )
            # Normalize relevance between [-1, 1] for plotting
            relevance = relevance / relevance.abs().max()
            print(f"Before: {relevance.shape}")
            relevance = relevance[:init_ids_len]  # trim to input tokens
            print(f"After: {relevance.shape}")
            relevance_scores.append(relevance)

            # Clear out the gradients after computing current relevance score
            model.zero_grad()
            input_embeds.grad = None

            # Do attribution before preparing for next step
            print(input_ids.shape)
            print(f"Trimmed input: {input_ids[:, :init_ids_len].shape}")
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :init_ids_len])
            # print(tokens)
            save_heatmap(len(relevance_scores) - 1, tokens, relevance, model_name)

        # Update for next step
        curr_len += 1
        # concat for next generation step
        input_ids = torch.cat([input_ids, topk_indices], dim=-1)
        next_token = topk_indices[0].item()
        # print(f" > {next_token}: {tokenizer.convert_ids_to_tokens(topk_indices[0])}")

    # print(dec_input_ids)
    resp_tokens = tokenizer.convert_ids_to_tokens(input_ids[0, init_ids_len:])
    # print(resp_tokens)
    print("=" * 40)
    print("Original text:")
    print("=" * 40)
    print(tokenizer.decode(input_ids[0, :init_ids_len], skip_special_tokens=True))
    print("=" * 40)
    print("Full output:")
    print("=" * 40)
    print(tokenizer.decode(input_ids[0, init_ids_len:], skip_special_tokens=True))
    print("=" * 40)
    print("Simplified text:")
    print("=" * 40)
    print(
        tokenizer.decode(
            input_ids[0, init_ids_len + curr_len :], skip_special_tokens=True
        )
    )

    # Remove special characters that are not compatible wiht LaTeX
    # tokens = tokenizer.convert_ids_to_tokens(enc_input_ids[0])
    # tokens = clean_tokens(tokens)

    # # Average out relevance_scores
    # relevance = np.array(relevance_scores).mean(axis=0)

    # # Removing up </s> tokens from the model
    # if "</s>" in tokens:
    #     tokens = tokens[:-1]
    #     relevance = relevance[:-1]

    # print("=" * 40)
    # print("Creating heatmap...")
    # pdf_heatmap(
    #     tokens,
    #     relevance,
    #     path=f"output/{model_name}_heatmap_gen.pdf",
    #     backend="xelatex",
    # )
