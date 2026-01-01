from langchain_unstructured import UnstructuredLoader

file_paths = [
    "./Duke.pdf",
]


loader = UnstructuredLoader(file_path= file_paths,chunking_strategy ="by_title", combine_text_under_n_chars=100)
for doc in loader.load():
    print(doc)