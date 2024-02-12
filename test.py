from deeplake import deepcopy

import json
import os

# load the local .env file
from dotenv import load_dotenv

print("copying")
load_dotenv()


ORG_NAME = "rubenselander"  # Organization name on activeloop hub
VECTOR_STORE_NAME = "eurostat_cohere"  # Name of vector store on activeloop hub
TOKEN = os.environ["ACTIVELOOP_TOKEN"]
path = f"hub://{ORG_NAME}/{VECTOR_STORE_NAME}"
dest = "eurostat"

deepcopy(src=path, dest="eurostat")
