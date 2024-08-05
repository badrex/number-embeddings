# %%
# basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Generator
import json


# utilities
from tqdm import tqdm, trange

# data processing
from scipy import stats
import faiss 

from sentence_transformers import SentenceTransformer

import argparse

# read variable model_id from command line
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="The model ID")
args = parser.parse_args()

model_id = args.model

# %%
# a module for lexicalizing numbers (e.g. 8 -> eight)")
import inflect
number_lexicalizer = inflect.engine()


# try it out 
number_lexicalizer.number_to_words(8)

# %%
id_to_model = {
    "miniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "miniLM-L12": "sentence-transformers/all-MiniLM-L12-v2",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    "jina-base": "jinaai/jina-embeddings-v2-base-en",
    "jina-small": "jinaai/jina-embeddings-v2-small-en",
    "jina-code": "jinaai/jina-embeddings-v2-base-code",
    "LaBSE": "sentence-transformers/LaBSE",
    #"textCLIP": "sentence-transformers/clip-ViT-B-32"
}

# %%
encoder_models = defaultdict()

for m in tqdm(id_to_model, desc="Loading models"):
    if m == model_id:
        encoder_models[m] = SentenceTransformer(
            id_to_model[m], 
            trust_remote_code=True
    )

# %%
# open the file and load the JSON data
file_path = './restaurants.json'
with open(file_path, 'r') as file:
    restaurants_data = json.load(file)

restaurant_documents = [
    restaurant['description'] for restaurant in restaurants_data
]

#len(restaurants_data), len(restaurant_documents)

# %%
restaurant_queries = [
    ("Show me restaurants rated {} stars or above", ">="),
    ("I'm looking for restaurants with at least {} stars", ">="),
    ("Suggest restaurants rated {} stars or higher", ">="),
    ("Find restaurants with at least {} stars rating", ">="),
    ("List restaurants rated {} stars and above", ">="),
    ("Recommend restaurants rated with {} stars or more", ">="),
    ("Restaurants with a minimum of {} stars", ">="),
    ("Show me restaurants rated with {} stars rating and above", ">="),
    ("Restaurants rated no less than {} stars", ">="),
    ("Find Restaurants with star ratings of or above {} stars", ">=")
]

# %%
def generate_test_sample(
    query: str,
    operator: str, 
    candidates: List[str], 
    max_items_to_retrieve:int, 
    target_number: int, 
    random_seed: int=42) -> Dict[str, Any]:
    """
    Generate a test sample for the item recommendation task.
    :param query: a query sentence template
    :param operator: a comparison operator (either '>' or '>=')
    :param candidates: a list of candidate sentences
    :param max_items_to_retrieve: the maximum number of items to retrieve
    :param target_number: the target number for the query
    :param random_seed: a random seed for reproducibility
    :return: a dictionary containing the query, candidates, and relevance scores
    """
    # handle edge cases
    if operator not in [">", ">="]:
        raise ValueError("Operator must be either '>' or '>='")
    
    if max_items_to_retrieve > len(candidates):
        err_msg = "max_items_to_retrieve can't be greater than num of candidates"
        raise ValueError(err_msg)

    if target_number == 10 and operator == ">":
        raise ValueError("Cannot find items with target attribute above 10.")
    
    # set random seed for reproducibility
    np.random.seed(random_seed)
    
    # create the query sentence
    # for example, query = "Show me restaurants rated above {} stars" and 
    # target_number = 5 --> query = "Show me restaurants rated above 5 stars"
    query_sent = query.format(target_number)

    # lexicalize the target number
    target_number_lex = number_lexicalizer.number_to_words(target_number)

    # create the query sentence with lexicalized number
    # above example, query_lex = "Show me restaurants rated above five stars"
    query_sent_lex = query.format(target_number_lex)
    
    # associate a value for each search item
    # where only N of those are equal to or higher than the target number
    items_to_retrieve = np.random.randint(1, max_items_to_retrieve)

    # adjust target number based on operator
    target_number = target_number + 1 if operator == ">" else target_number

    hit_items = np.random.randint(target_number, 11, items_to_retrieve)
    miss_items = np.random.randint(
        1, target_number, len(candidates) - items_to_retrieve
    )

    # list of number for each item
    items_numbers: List[int] = np.concatenate([hit_items, miss_items])

    # define a boolean list to check if the rating is hit (should be returned)
    relevance_scores: List[int] = [
        0 if rating < target_number else 1 for rating in items_numbers
    ]

    # ensure correct number of relevant items
    if np.sum(relevance_scores) != items_to_retrieve:
        print(relevance_scores)
        raise ValueError("Relevance score != equal to max_items_to_retrieve!")    


    candidates_sent = [
        candidate.format(num)
        for candidate, num in zip(candidates, items_numbers)
    ]

    candidates_sent_lex = [
        candidate.format(number_lexicalizer.number_to_words(num))
        for candidate, num in zip(candidates, items_numbers)
    ]
    
    # construct the result
    return {
        "query": {
            "numeral": query_sent,
            "lexical": query_sent_lex
        },
        "candidates": {
            "numeral": candidates_sent,
            "lexical": candidates_sent_lex
        },
        "relevance_scores": relevance_scores
    }

# %%
# try out test sample generation
t_sample = generate_test_sample(
    query="Show me restaurants rated above {} stars", 
    operator=">",
    candidates=[
        "Tickets, known for creative tapas, has been given {} stars.",
        "The Restaurant at Meadowood, showcasing California cuisine, boasts a {} star rating.",
        "Toyo Eatery, celebrating modern Filipino flavors, has earned {} stars.",
        "Septime, a neo-bistro in Paris, holds {} stars.",
    ], 
    max_items_to_retrieve=2, 
    target_number=7
)

print(t_sample)

# %%
# try out test sample generation
t_sample_2 = generate_test_sample(
    query="Show me restaurants rated above {} stars", 
    operator=">",
    candidates=[
        "Tickets, known for creative tapas, has been given {} stars.",
        "The Restaurant at Meadowood, showcasing California cuisine, boasts a {} star rating.",
        "Toyo Eatery, celebrating modern Filipino flavors, has earned {} stars.",
        "Septime, a neo-bistro in Paris, holds {} stars.",
    ], 
    max_items_to_retrieve=3, 
    target_number=7,
    random_seed=1234
)

print(t_sample_2)

# %%
def generate_test_samples_for_query(
    query_item: Tuple[str, str],
    candidates: List[str],
    max_items_to_retrieve: int, 
    random_seeds: List[int]) -> Generator:
    """
    Generate a test dataset for the item retrieval task.
    """
    # unpack the query item
    query, operator = query_item

    # generate test samples for each target number
    # sample a target number 
    for rand_seed in random_seeds:
        np.random.seed(rand_seed)

        target_number = np.random.randint(6, 10)

        yield generate_test_sample(
            query=query, 
            operator=operator,
            candidates=candidates,
            max_items_to_retrieve=max_items_to_retrieve,
            target_number=target_number,
            random_seed=rand_seed
        )  

# %%
test_cases = []

for q in restaurant_queries:
    test_cases.extend(
        [
            t_case for t_case in generate_test_samples_for_query(
                query_item=q,
                candidates=restaurant_documents,
                max_items_to_retrieve=10,
                random_seeds=[
                    42, 1234, 5678, 9101, 321, 765, 13, 1212, 42, 8, 46648
                ]
            )
        ]
    )

# %%
hit_to_emoji = {0: "", 1: "✅"}

# %%
# evaluate the performance of the model
# code to evaluate a single test case and return precision and recall at 10
def evaluate_test_case(
        query: str, 
        candidates: List[str], 
        relevance_scores: List[int],
        top_k: int, 
        model: SentenceTransformer,
        debug=False) -> Tuple[float, float]:
    """
    Evaluate a test case using a given model.
    :param query: a query sentence
    :param candidates: a list of candidate sentences
    :param relevance_score: a list of relevance scores
    :param top_k: the number of items to retrieve
    :param model: a sentence transformer model to use
    :return: a tuple of precision and recall at 10
    """
    # encode the query and candidates
    query_embedding = model.encode(query) #.reshape(1, -1)
    candidate_embeddings = model.encode(candidates)

    # compute the cosine similarity between the query and candidates
    similarity = np.dot(candidate_embeddings, query_embedding.T)

    # rank the candidates based on the similarity
    ranked_indices = np.argsort(similarity, axis=0)[::-1]

    ranked_similarity = similarity[ranked_indices]

    # retrieve the relevance scores based on the ranking
    ranked_relevance = np.array(relevance_scores)[ranked_indices]

    # print query and top 10 results
    if debug:
        print(f"Query: {query}")
        print()
        print("\n".join(
            [
                f"{i + 1:>3}: {candidates[j]:<90}{ranked_similarity[i]:>5.3f} {hit_to_emoji[ranked_relevance[i]]:>3}" \
                    for i, j in enumerate(ranked_indices[:top_k])
            ]), end='\n'
        ) 
        print(f"Number of relevant items: {np.sum(relevance_scores)}", end="\n\n") 

    # compute precision and recall at 10
    precision_at_k = np.sum(ranked_relevance[:top_k]) / top_k
    recall_at_k = np.sum(ranked_relevance[:top_k]) / np.sum(relevance_scores)

    return precision_at_k, recall_at_k


# %%
# evaluate the performance of the model
# code to evaluate a single test case and return precision and recall at 10
def evaluate_test_case_faiss(
        query: str, 
        candidates: List[str], 
        relevance_scores: List[int],
        top_k: int, 
        model: SentenceTransformer,
        debug=False) -> Tuple[float, float]:
    """
    Evaluate a test case using a given model.
    :param query: a query sentence
    :param candidates: a list of candidate sentences
    :param relevance_score: a list of relevance scores
    :param top_k: the number of items to retrieve
    :param model: a sentence transformer model to use
    :return: a tuple of precision and recall at 10
    """
    # encode the query and candidates
    query_embedding = model.encode(query).reshape(1, -1)
    candidate_embeddings = model.encode(candidates)

    # use faiss to normalize the vectors
    faiss.normalize_L2(query_embedding)
    faiss.normalize_L2(candidate_embeddings)

    # compute the cosine similarity between the query and candidates using faiss
    index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
    index.add(candidate_embeddings)
    ranked_similarity, ranked_indices = index.search(query_embedding, top_k)

    # retrieve the relevance scores based on the ranking
    relevance_scores = np.array(relevance_scores)
    ranked_relevance = relevance_scores[ranked_indices[0]]

    # print query and top 10 results
    if debug:
        print(f"Query: {query}")
        print()
        print("\n".join(
            [
                f"{i + 1:>3}: {candidates[j]:<90}{ranked_similarity[0][i]:>5.3f} {hit_to_emoji[ranked_relevance[i]]:>3}" \
                    for i, j in enumerate(ranked_indices[0][:top_k])
           ]), end='\n'
        ) 
        print(f"Number of relevant items: {np.sum(relevance_scores)}", end="\n\n") 

    # compute precision and recall at 10
    precision_at_k = np.sum(ranked_relevance[:top_k]) / top_k
    recall_at_k = np.sum(ranked_relevance[:top_k]) / np.sum(relevance_scores)

    return precision_at_k, recall_at_k


# %%
evaluate_test_case(
    query="Show me restaurants rated above 8 stars",
    candidates=[
        "Tickets, known for creative tapas, has been given 9 stars.",
        "The Restaurant at Meadowood, showcasing California cuisine, boasts a 7 star rating.",
        "Toyo Eatery, celebrating modern Filipino flavors, has earned 8 stars.",
        "Septime, a neo-bistro in Paris, holds 8 stars.",
    ],
    relevance_scores=[1, 0, 0, 0],
    top_k=4,
    model=encoder_models[model_id],
    debug=True
)

# %%
evaluate_test_case_faiss(
    query="Show me restaurants rated above 8 stars",
    candidates=[
        "Tickets, known for creative tapas, has been given 9 stars.",
        "The Restaurant at Meadowood, showcasing California cuisine, boasts a 7 star rating.",
        "Toyo Eatery, celebrating modern Filipino flavors, has earned 8 stars.",
        "Septime, a neo-bistro in Paris, holds 8 stars.",
    ],
    relevance_scores=[1, 0, 0, 0],
    top_k=4,
    model=encoder_models[model_id],
    debug=True
)

# %%

recall_values = []

for i in tqdm(range(10)):
    p, r = evaluate_test_case_faiss(
        test_cases[i]["query"]["numeral"],
        test_cases[i]["candidates"]["numeral"],
        test_cases[i]["relevance_scores"],
        top_k=10,
        model=encoder_models[model_id],
        debug=True
    )
    #print(f"Case {i:>3}:    Recall@10: {r:.3f}")
    print()

    recall_values.append(r)


print(f"Mean Recall@10: {np.mean(recall_values):.3f}")
print(f"Std Recall@10: {np.std(recall_values):.3f}")

# # %%
recall_values = []

for encoder in encoder_models:

    print(f"Model: {encoder}    ", end="")
    for i in tqdm(range(len(test_cases)), total=len(test_cases)):
        p, r = evaluate_test_case(
            test_cases[i]["query"]["numeral"],
            test_cases[i]["candidates"]["numeral"],
            test_cases[i]["relevance_scores"],
            top_k=10,
            model=encoder_models[encoder],
            debug=True
        )
        #print(f"Case {i:>3}:    Recall@10: {r:.3f}")
        #print()

        recall_values.append(r)

    print()
    print(f"Model: {encoder}: ")
    print(f"mean R@10: {np.mean(recall_values):.3f}, ", end="")
    print(f"std R@10: {np.std(recall_values):.3f}")