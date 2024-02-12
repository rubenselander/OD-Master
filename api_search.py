import json
import cohere
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
import json
import os

# load the local .env file
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.environ["COHERE_API_KEY"])
VECTOR_STORE_NAME = "eurostat"  # Name of vector store on activeloop hub
# TOKEN = os.environ["ACTIVELOOP_TOKEN"]


vector_store = VectorStore(path=VECTOR_STORE_NAME)


# # Potential filter function for vector_store.search


def cohere_embedding_function(texts, model="embed-multilingual-v3.0"):
    if isinstance(texts, str):
        texts = [texts]

    response = co.embed(texts, model=model, input_type="search_query")
    return response.embeddings


def search_tables(search_string: str, k: int = 10):
    """Performs a search in tables based on the given search string."""
    results = vector_store.search(
        embedding_data=search_string,
        embedding_function=cohere_embedding_function,
        # exec_option="tensor_db",
        return_tensors=["text", "code", "start_date", "end_date"],
        k=k,
    )
    return results


def format_search_results(search_results: dict, include_score: bool = False) -> dict:
    """Formats the search results to the format expected by the frontend."""
    if not include_score:
        search_results.pop("score", None)
    formatted_results = []
    # number of results is equal to the length of any of the lists in the dict
    # set the number of results to the length of the FIRST list in the dict
    nbr_of_results = len(list(search_results.values())[0])
    # each dict in the formatted_results list should have all the same keys
    # as the search_results dict

    for i in range(nbr_of_results):
        result = {}
        for key, value in search_results.items():
            if isinstance(value[i], list) and len(value[i]) == 1:
                result[key] = value[i][0]
            else:
                result[key] = value[i]
        formatted_results.append(result)

    return formatted_results


def search_eurostat(search_string: str, year: int = None, k=10) -> dict:
    """Performs a search in Eurostat based on the given search string."""
    search_results = search_tables(search_string, k=k)
    # possible reranking is done here
    # RERANK
    formatted_results = format_search_results(search_results)
    return formatted_results


def od_search(search_string: str, k: int = 10):
    """Performs a search in tables based on the given search string."""

    results = search_eurostat(search_string, k=k)
    base_url = "https://ec.europa.eu/eurostat/databrowser/view/{CODE}/default/table"
    results_with_urls = []
    for res in results:
        url = base_url.format(CODE=res["code"].lower())
        res["url"] = url
        results_with_urls.append(res)
    return results_with_urls
