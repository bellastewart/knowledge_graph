import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client
from helpers.df_helpers import GraphJSON, graphjson_to_nx # adjust if it's defined in the same script
import json
import networkx as nx



def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


#no chunking metadata passed in 
def docsgraphPrompt(input: str, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        'You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. '
        'You are provided with a context chunk (delimited by ```) Your task is to extract the ontology of terms mentioned in the given context, representing the key concepts as per the context with well-defined and widely used names of materials, systems, methods.'
        'You always report a technical term or abbreviation and keep it as it is.'
        'If you receive a location to an image, you must use it as a node which <id> will be the location and the <type> will be "image" and relate the information in the context to make the nodes and edges relation.'
        '<relation> in an edge must truly reveal important information that can provide scientific insight from the <source> to the <target>'
        'Return a JSON with two fields: <nodes> and <edges>.\n'
        'Each node must have <id> and <type>.\n'
        'Each edge must have <source>, <target>, and <relation>.'
    )

    USER_PROMPT = f"context: ```{input}``` \n\n Extract the knowledge graph in structured JSON: "
    print ('Generating triples...')
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        cleaned_response = response.strip().strip("```")
        raw_result = json.loads(cleaned_response)

        # Validate the JSON
        validated_result = GraphJSON.model_validate(raw_result)

        # Create the graph (no metadata needed)
        G = graphjson_to_nx(validated_result)

        # Save the graph
        nx.write_graphml(G, "temp/full_text.graphml")

        print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    except Exception as e:
        print("\n\nERROR ### Could not parse or validate graph JSON. Here is the buggy response:\n", response)
        print("Exception:", e, "\n\n")
        G = None

    return G