import torch

from transformers import T5Tokenizer
from lxt.efficient.models.flan_t5 import T5ForConditionalGeneration
from lxt.utils import pdf_heatmap, clean_tokens


def print_tokens(tokens: torch.Tensor, logits: torch.Tensor = None):
    if logits is not None:
        # print the tokens and their logits
        for token, logit in zip(tokens, logits):
            print(f"Token: {token}, Logit: {logit.item()}")
    else:
        for token in tokens:
            print(f"Token: {token}")


if __name__ == "__main__":
    print("=" * 40)
    model_path = "google/flan-t5-small"
    print(f"Loading and setting up model: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # model.train()
    # model.gradient_checkpointing_enable()
    # deactive gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    print("=" * 40)
    enc_prompt = """Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
    Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""
    # Decoder input: the beginning of the text to complete
    dec_prompt = """According to the text, the 1922 expedition reached 8,"""
    print(f"## Encoder input:\n{enc_prompt}")
    print(f"## Decoder input:\n{dec_prompt}")

    # get input embeddings so that we can compute gradients w.r.t. input embeddings
    enc_input_ids = tokenizer(
        enc_prompt, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(model.device)
    enc_input_embeds = model.get_input_embeddings()(enc_input_ids)

    # decoder_input_ids = model._shift_right(input_ids).to(model.device)
    dec_input_ids = tokenizer(dec_prompt, return_tensors="pt").input_ids.to(
        model.device
    )
    dec_input_ids = model._shift_right(dec_input_ids)
    dec_input_embeds = model.get_input_embeddings()(dec_input_ids)
    # print("Decoder input token:")
    # print_tokens(tokenizer.convert_ids_to_tokens(dec_input_ids[0]))

    print("=" * 40)
    print("Running inference...")
    output_logits = model(
        inputs_embeds=enc_input_embeds.requires_grad_(),
        decoder_inputs_embeds=dec_input_embeds.requires_grad_(),
        use_cache=False,
    ).logits
    # Get the top k tokens and their logits
    k = 5
    max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)
    topk_logits, topk_indices = torch.topk(output_logits[0, -1, :], k=k, dim=-1)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices)
    print("=" * 40)
    print("## Generated tokens (top-{k}):")
    print_tokens(topk_tokens, topk_logits)

    print("=" * 40)
    print("Running backward pass...")
    max_logits.backward()
    relevance = (
        (enc_input_embeds.grad * enc_input_embeds).float().sum(-1).detach().cpu()[0]
    )
    # Normalize relevance between [-1, 1] for plotting
    relevance = relevance / relevance.abs().max()

    # Remove special characters that are not compatible wiht LaTeX
    tokens = tokenizer.convert_ids_to_tokens(enc_input_ids[0])
    # tokens = clean_tokens(tokens)

    # Removing up </s> tokens from the model
    if "</s>" in tokens:
        tokens = clean_tokens(tokens)[:-1]
        relevance = relevance[:-1]

    print("=" * 40)
    print("Creating heatmap...")
    pdf_heatmap(tokens, relevance, path="output/flan_t5_heatmap.pdf", backend="xelatex")
