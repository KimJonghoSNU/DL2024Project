from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("/workspace/data03/pretrained_models/Meta-Llama-3.1-8B-Instruct/")


def html_to_txt():
    urls = ["https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html"]
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    print(docs_transformed[0].page_content)
    # save it to text file
    with open("data/daylight.txt", "w") as f:
        f.write(docs_transformed[0].page_content)
    print(len(docs_transformed[0].page_content))
    print(len(tok.tokenize(docs_transformed[0].page_content)))
    
with open("data/daylight_filtered.txt", "r") as f:
    txt = f.read()
    print(len(tok.tokenize(txt)))
    print(tok.model_max_length)
    model = AutoModel.from_pretrained("/workspace/data03/pretrained_models/Meta-Llama-3.1-8B-Instruct/")
    # max length of the model:
    # print(model.config.max_position_embeddings)
# print(docs_transformed[1].page_content[1000:2000])