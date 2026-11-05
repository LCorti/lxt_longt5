import torch
from transformers import AutoTokenizer

from lxt.efficient.models.longt5 import LongT5ForConditionalGeneration
from lxt.utils import clean_tokens, pdf_heatmap


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
    model_name = "long-t5-summ"
    model_path = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    print(f"Loading and setting up model: {model_path}")
    model = LongT5ForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # deactive gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    paper = "\nABSTRACT\nFor developing a data-driven text rewriting algorithm for paraphrasing, it is essential to have a monolingual corpus of aligned paraphrased sentences.\nNews article headlines are a rich source of paraphrases; they tend to describe the same event in various different ways, and can easily be obtained from the web.\nWe compare two methods of aligning headlines to construct such an aligned corpus of paraphrases, one based on clustering, and the other on pairwise similarity-based matching.\nWe show that the latter performs best on the task of aligning paraphrastic headlines.\n\nSECTION 1: Introduction\nIn recent years, text-to-text generation has received increasing attention in the field of Natural Language Generation (NLG).\nIn contrast to traditional concept-to-text systems, text-to-text generation systems convert source text to target text, where typically the source and target text share the same meaning to some extent.\nApplications of text-to-text generation include sum- marization (Knight and Marcu, 2002), question- answering (Lin and Pantel, 2001), and machine translation.\nFor text-to-text generation it is important to know which words and phrases are semantically close or exchangable in which contexts.\nWhile there are various resources available that capture such knowledge at the word level (e.g., synset knowledge in WordNet), this kind of information is much harder to get by at the phrase level.\nTherefore, paraphrase acquisition can be considered an important technology for producing resources for text-to-text generation.\nParaphrase generation has already proven to be valuable for Question Answering (Lin and Pantel, 2001; Riezler et al., 2007), Machine Translation (CallisonBurch et al., 2006) and the evaluation thereof (RussoLassner et al., 2006; Kauchak and Barzilay, 2006; Zhou et al., 2006), but also for text simplification and explanation.\nIn the study described in this paper, we make an effort to collect Dutch paraphrases from news article headlines in an unsupervised way to be used in future paraphrase generation.\nNews article headlines are abundant on the web, and are already grouped by news aggregators such as Google News.\nThese services collect multiple articles covering the same event.\nCrawling such news aggregators is an effective way of collecting related articles which can straightforwardly be used for the acquisition of paraphrases (Dolan et al., 2004; Nelken and Shieber, 2006).\nWe use this method to collect a large amount of aligned paraphrases in an automatic fashion.\n\nSECTION 2: Method. \nWe aim to build a high-quality paraphrase corpus.\nConsidering the fact that this corpus will be the basic resource of a paraphrase generation system, we need it to be as free of errors as possible, because errors will propagate throughout the system.\nThis implies that we focus on obtaining a high precision in the paraphrases collection process.\nWhere previous work has focused on aligning news-items at the paragraph and sentence level (Barzilay and Elhadad, 2003), we choose to focus on aligning the headlines of news articles.\nWe think this approach will enable us to harvest reliable training material for paraphrase generation quickly and efficiently, without having to worry too much about the problems that arise when trying to align complete news articles.\nFor the development of our system we use data which was obtained in the DAESO-project.\nThis project is an ongoing effort to build a Parallel Monolingual Treebank for Dutch (Marsi Proceedings of the 12th European Workshop on Natural Language Generation, pages 122\u00e2\u0080\u0093125, Athens, Greece, 30 \u00e2\u0080\u0093 31 March 2009.\nQc 2009 Association for Computational Linguistics document, and each original cluster as a collection of documents.\nFor each stemmed word i in sentence j, T Fi,j is a binary variable indicating if the word occurs in the sentence or not.\nThe T F \u00e2\u0088\u0097I DF score is then: TF.IDFi = T Fi,j \u00c2\u00b7 log | Table 1: Part of a sample headline cluster, with sub-clusters and Krahmer, 2007) and will be made available through the Dutch HLT Agency.\nPart of the data in the DAESO-corpus consists of headline clusters crawled from Google News Netherlands in the period April\u00e2\u0080\u0093August 2006.\nFor each news article, the headline and the first 150 characters of the article were stored.\nRoughly 13,000 clusters were retrieved.\nTable 1 shows part of a (translated) cluster.\nIt is clear that although clusters deal roughly with one subject, the headlines can represent quite a different perspective on the content of the article.\nTo obtain only paraphrase pairs, the clusters need to be more coherent.\nTo that end 865 clusters were manually subdivided into sub-clusters of headlines that show clear semantic overlap.\nSub- clustering is no trivial task, however.\nSome sentences are very clearly paraphrases, but consider for instance the last two sentences in the example.\nThey do paraphrase each other to some extent, but their relation can only be understood properly with |{dj : ti \u00e2\u0088\u0088 dj }| |D| is the total number of sentences in the cluster and |{dj : ti \u00e2\u0088\u0088 dj }| is the number of sen tences that contain the term ti.\nThese scores are used in a vector space representation.\nThe similarity between headlines can be calculated by using a similarity function on the headline vectors, such as cosine similarity.\n2.1 Clustering.\nOur first approach is to use a clustering algorithm to cluster similar headlines.\nThe original Google News headline clusters are reclustered into finer grained sub-clusters.\nWe use the k-means implementation in the CLUTO1 software package.\nThe k-means algorithm is an algorithm that assigns k centers to represent the clustering of n points (k < n) in a vector space.\nThe total intra-cluster variances is minimized by the function k V = (xj \u00e2\u0088\u0092 \u00c2\u00b5i)2 i=1 xj \u00e2\u0088\u0088Si where \u00c2\u00b5i is the centroid of all the points xj \u00e2\u0088\u0088 Si.The PK1 cluster-stopping algorithm as pro posed by Pedersen and Kulkarni (2006) is used to find the optimal k for each sub-cluster: C r(k) \u00e2\u0088\u0092 mean(C r[1...\u00e2\u0088\u0086K ]) world knowledge.\nAlso, there are numerous headlines that can not be sub-clustered, such as the first P K 1(k) = std(C r[1...\u00e2\u0088\u0086K ]) three headlines shown in the example.\nWe use these annotated clusters as development and test data in developing a method to automatically obtain paraphrase pairs from headline clusters.\nWe divide the annotated headline clusters in a development set of 40 clusters, while the remainder is used as test data.\nThe headlines are stemmed using the porter stemmer for Dutch (Kraaij and Pohlmann, 1994).\nInstead of a word overlap measure as used byHere, C r is a criterion function, which mea sures the ratio of withincluster similarity to betweencluster similarity.\nAs soon as P K 1(k) ex ceeds a threshold, k \u00e2\u0088\u0092 1 is selected as the optimum number of clusters.\nTo find the optimal threshold value for cluster- stopping, optimization is performed on the development data.\nOur optimization function is an F - score: (1 + \u00ce\u00b22) \u00c2\u00b7 (precision \u00c2\u00b7 recall) Barzilay and Elhadad (2003), we use a modified F\u00ce\u00b2 = (\u00ce\u00b22 precision + recall) T F \u00e2\u0088\u0097I DF word score as was suggested by Nelken \u00c2\u00b7 and Shieber (2006).\nEach sentence is viewed as a 1 http://glaros.dtc.umn.edu/gkhome/views/cluto/ We evaluate the number of aligments between possible paraphrases.\nFor instance, in a cluster of four sentences, 4) = 6 alignments can be made.\nIn our case, precision is the number of alignments retrieved from the clusters which are relevant, divided by the total number of retrieved alignments.\nRecall is the number of relevant retrieved aligments divided by the total number of relevant alignments.\nWe use an F\u00ce\u00b2 -score with a \u00ce\u00b2 of 0.25 as we favour precision over recall.\nWe do not want to optimize on precision alone, because we still want to retrieve a fair amount of paraphrases and not only the ones that are very similar.\nThrough optimization on our development set, we find an optimal threshold for the PK1 algorithm thpk1 = 1.\nFor each original cluster, k-means clustering is then performed using the k found by the cluster stopping function.\nIn each newly obtained cluster all headlines can be aligned to each other.\n2.2 Pairwise similarity.\nOur second approach is to calculate the similarity between pairs of headlines directly.\nIf the similarity exceeds a certain threshold, the pair is accepted as a paraphrase pair.\nIf it is below the threshold, it is rejected.\nHowever, as Barzilay and Elhadad (2003) have pointed out, sentence mapping in this way is only effective to a certain extent.\nBeyond that point, context is needed.\nWith this in mind, we adopt two thresholds and the Cosine similarity function to calculate the similarity between two sentences: cos(\u00ce\u00b8) = V 1 \u00c2\u00b7 V 2 V 1 V 2 where V 1 and V 2 are the vectors of the two sentences being compared.\nIf the similarity is higher than the upper threshold, it is accepted.\nIf it is lower than the lower theshold, it is rejected.\nIn the remaining case of a similarity between the two thresholds, similarity is calculated over the contexts of the two headlines, namely the text snippet that was retrieved with the headline.\nIf this similarity exceeds the upper threshold, it is accepted.\nThreshold values as found by optimizing on the development data using again an F0.25-score, are T hlower = 0.2 and T hupper = 0.5.\nAn optional final step is to add alignments that are implied by previous alignments.\nFor instance, if headline A is paired with headline B, and headline B is aligned to headline C , headline A can be aligned to C as Ty pe Precision Recallk m ea ns cl us ter in g 0.91 0.43 clu ste rs on lyk m ea ns cl us ter in g 0.66 0.44 all he ad lin es pa irw ise si mi lar ity 0.93 0.39 clu ste rs on ly pa irw ise si mi lar ity 0.76 0.41 all he ad lin es Table 2: Precision and Recall for both methods Pl ay st ati on 3 m or e ex pe nsi ve th an co m pe tit or P l a y s t a t i o n 3 w i l l b e c o m e m o r e e x p e n s i v e t h a n X b o x 3 6 0 So ny po stp on es Blu Ra y m ov ie s So ny po stp on es co mi ng of blu ra y dv ds Pri ce s Pl ay st ati on 3 kn ow n: fro m 49 9 eu ro s E3 20 06 : Pl ay st ati on 3 fro m 49 9 eu ro s So ny PS 3 wi th Blu R ay for sal e fro m No ve m be r 11 th PS 3 av ail abl e in Eu ro pe fro m No ve m be r 17 th Table 3: Examples of correct (above) and incorrect (below) alignments well.\nWe do not add these alignments, because in particular in large clusters when one wrong alignment is made, this process chains together a large amount of incorrect alignments.\n\nSECTION 3: Results. \nThe 825 clusters in the test set contain 1,751 sub- clusters in total.\nIn these sub-clusters, there are 6,685 clustered headlines.\nAnother 3,123 headlines remain unclustered.\nTable 2 displays the paraphrase detection precision and recall of our two approaches.\nIt is clear that k-means clustering performs well when all unclustered headlines are artificially ignored.\nIn the more realistic case when there are also items that cannot be clustered, the pairwise calculation of similarity with a back off strategy of using context performs better when we aim for higher precision.\nSome examples of correct and incorrect alignments are given in Table 3.\n\nSECTION 4: Discussion. \nUsing headlines of news articles clustered by Google News, and finding good paraphrases within these clusters is an effective route for obtaining pairs of paraphrased sentences with reasonable precision.\nWe have shown that a cosine similarity function comparing headlines and using a back off strategy to compare context can be used to extract paraphrase pairs at a precision of 0.76.\nAlthough we could aim for a higher precision by assigning higher values to the thresholds, we still want some recall and variation in our paraphrases.\nOf course the coverage of our method is still somewhat limited: only paraphrases that have some words in common will be extracted.\nThis is not a bad thing: we are particularly interested in extracting paraphrase patterns at the constituent level.\nThese alignments can be made with existing alignment tools such as the GIZA++ toolkit.\nWe measure the performance of our approaches by comparing to human annotation of sub- clusterings.\nThe human task in itself is hard.\nFor instance, is we look at the incorrect examples in Table 3, the difficulty of distinguishing between paraphrases and non-paraphrases is apparent.\nIn future research we would like to investigate the task of judging paraphrases.\nThe next step we would like to take towards automatic paraphrase generation, is to identify the differences between paraphrases at the constituent level.\nThis task has in fact been performed by human annotators in the DAESO-project.\nA logical next step would be to learn to align the different constituents on our extracted paraphrases in an unsupervised way.\n\nSECTION: Acknowledgements\nThanks are due to the Netherlands Organization for Scientific Research (NWO) and to the Dutch HLT Stevin programme.\nThanks also to Wauter Bosma for originally mining the headlines from Google News.\nFor more information on DAESO, please visit daeso.uvt.nl.\n"

    print("=" * 40)
    enc_prompt = f"Summarise the following paper.\nPaper: {paper}\nSummary:"
    # Decoder input: the beginning of the text to complete
    dec_prompt = """Summary:"""
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
    k = 3
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
    tokens = clean_tokens(tokens)
    # Removing up </s> tokens from the model
    if "</s>" in tokens:
        tokens = tokens[:-1]
        relevance = relevance[:-1]

    print("=" * 40)
    print("Creating heatmap...")
    pdf_heatmap(
        tokens, relevance, path=f"output/{model_name}_heatmap.pdf", backend="xelatex"
    )
