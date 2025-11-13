import numpy as np
import torch
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

from pathlib import Path
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens


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
    model_name = "qwen2"
    model_path = "Qwen/Qwen2-1.5B-Instruct"
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
    prompt = f"Rewrite the following text in simpler English language. Only provide the simplified text without repeating the original one.\nText: {paper}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, agg_generation_prompt=True
    )

    # get input embeddings so that we can compute gradients w.r.t. input embeddings
    input_ids = tokenizer(
        [text],
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(model.device)

    # generated_ids = model.generate(input_ids, max_new_tokens=128, do_sample=False)
    # generated_ids = [
    #     output_ids[len(input_ids) :]
    #     for input_ids, output_ids in zip(input_ids, generated_ids)
    # ]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("#" * 40)
    # print(response)
    # print("#" * 40)

    print("-" * 40)
    print(f"Input size: {input_ids.shape}")

    max_new_tokens = 256  # Change as needed
    init_ids_len = input_ids.shape[-1]
    min_length = init_ids_len + 2
    max_length = max_new_tokens + init_ids_len
    max_length = max_length if max_length < max_output_length else max_output_length
    eos_token_id = tokenizer.eos_token_id
    decoding_strategy = "greedy"  # or sample

    print("Output settings")
    print(f" - min_length: {min_length}")
    print(f" - max_length: {max_length}")
    print("-" * 40)

    logits_processor = LogitsProcessorList(
        [
            RepetitionPenaltyLogitsProcessor(penalty=1.0),
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
    while init_ids_len + curr_len <= max_length and next_token != eos_token_id:
        input_embeds = model.get_input_embeddings()(input_ids)
        output_logits = model(
            inputs_embeds=input_embeds.requires_grad_(),
            use_cache=False,
        ).logits

        if torch.backends.mps.is_available():
            raw_next_logits = output_logits[0, -1, :].cpu()
            input_ids_copy = input_ids.cpu()
            next_token_scores = logits_processor(
                input_ids_copy, raw_next_logits.unsqueeze(0)
            ).to(model.device)
        else:
            next_token_scores = logits_processor(
                input_ids, output_logits[0, -1, :].unsqueeze(0)
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

        topk_token = tokenizer.convert_ids_to_tokens(topk_indices)
        # Print tokens that are part of the response
        # print(">" * 50)
        # print(f"{topk_indices}: {topk_token}")

        # print("=" * 40)
        # print("Running backward pass...")
        max_raw_logits.backward()
        relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0]
        # Normalize relevance between [-1, 1] for plotting
        relevance = relevance / relevance.abs().max()
        print(f"Before: {relevance.shape}")
        relevance = relevance[:init_ids_len]  # trim to input tokens
        print(f"After: {relevance.shape}")
        relevance_scores.append(relevance)

        # Clear out the gradients after computing current relevance score
        if input_embeds.grad is not None:
            input_embeds.grad = None
        model.zero_grad()

        # Do attribution before preparing for next step
        print(input_ids.shape)
        print(f"Trimmed input: {input_ids[:, :init_ids_len].shape}")
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :init_ids_len])
        save_heatmap(len(relevance_scores) - 1, tokens, relevance, model_name)

        # Update for next step
        curr_len += 1
        # concat for next generation step
        input_ids = torch.cat([input_ids, topk_indices], dim=-1)
        next_token = topk_indices[0].item()
        # print(f" > {next_token}: {tokenizer.convert_ids_to_tokens(topk_indices[0])}")

    simpl = tokenizer.decode(input_ids[0, init_ids_len:], skip_special_tokens=True)
    if simpl.startswith("user"):
        simpl = simpl.replace("user", "", 1).strip()
    print("=" * 40)
    print("Simplified output:")
    print("=" * 40)
    print(simpl)
