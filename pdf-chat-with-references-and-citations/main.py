import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, StuffDocumentsChain, ConversationalRetrievalChain, LLMChain, create_citation_fuzzy_match_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.vectorstores import FAISS

# InstructorEmbedding 
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pickle

# OpenAI Embedding
from langchain.embeddings import OpenAIEmbeddings

from pprint import pprint

# loader = TextLoader('single_text_file.txt')
# loader = DirectoryLoader('documents/', glob="./*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()
# print("Document has been loaded.")
# print("documents:", documents)

# Split it up
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)
# print("Document has been split.", len(texts))
# print(f"texts[0]\n{texts[0]}\n\ntexts[1]\n{texts[1]}\n\ntexts[-1]\n{texts[-1]}\n\nlen(texts):{len(texts)}")

# Create embeddings
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
# print("\nEmbeddings received from model.")
# print(f"embeddings: {embeddings}")

# faiss_obj = FAISS.from_documents(texts, embeddings)
# print(f"\\faiss_obj: {faiss_obj}")

# Store embeddings locally so we don't run the model again
# with open("embeddings/langchain-tos.pkl", "wb") as f:
#     pickle.dump(faiss_obj, f)
# print("Embeddings have been stored.")

# Load embeddings
import time
start_time = time.time()
with open(f"embeddings/langchain-tos.pkl", "rb") as f:
    loaded_embeddings = pickle.load(f)
end_time = time.time()
print(f"Embeddings have been loaded in {end_time - start_time} seconds.", loaded_embeddings)
faiss_obj = loaded_embeddings

retriever = faiss_obj.as_retriever(search_kwargs={"k": 3})

query = 'What is langchain OSS code?'
# query = 'Do I have to register an account for using the langsmith platform?'

docs = retriever.get_relevant_documents(query)
# docs = [Document(page_content='on-line help files, technical documentation and user manuals made available by LangChain for the LangSmith Platform. “LangChain Distributed Code” means any software code provided by LangChain to Customer for use in connection with the LangSmith Platform, other than LangChain OSS Code. “LangChain OSS Code” means any software code made available by LangChain under an open source license, including at https://github.com/langchain-ai/langchain. “LangSmith Platform” means LangChain’s application development platform for monitoring, testing, and debugging large language models applications, which may be provided both as a cloud offering or on-premise (including local) deployment.', metadata={'source': 'documents\\langchain-terms-of-service.pdf', 'page': 0}),
#  Document(page_content='an authorized license key from LangChain.  The license key may impose limits on the use of the LangSmith Platform or LangChain Distributed Code, such as the license term or number of authorized users.  Customer is solely responsible for maintaining active license key(s) to ensure continued access and use of the corresponding features of the LangSmith Platform or LangChain Distributed Code, and LangChain reserves the right to suspend access in the event Customer’s license key(s) are invalid or expired.   2.5 Customer Limitations.  Customer will not directly or indirectly: (a) reverse engineer, decompile, disassemble, modify, create derivative works of or otherwise create, attempt to create or derive, or permit or assist any third party to create or derive, the source code underlying the LangSmith Platform; (b) attempt to probe, scan or test the vulnerability of the LangSmith Platform, breach the security or authentication measures of the LangSmith Platform without proper authorization', metadata={'source': 'documents\\langchain-terms-of-service.pdf', 'page': 1}),
#  Document(page_content='Code, System Data and LangChain’s Confidential Information (“LangChain Materials”).  Customer exclusively owns all right, title and interest in and to the applications it develops using the LangSmith Platform (excluding any LangChain Materials).  “System Data” means data collected by LangChain regarding the LangSmith Platform that may be used to generate logs, statistics or reports regarding the performance, availability, usage, integrity or security of the LangSmith Platform. 4.4 Feedback.  Customer may from time to time provide LangChain suggestions or comments for enhancements or improvements, new features or functionality or other feedback (“Feedback”) with respect to the LangSmith Platform.  LangChain will have full discretion to determine whether or not to proceed with the development of any requested enhancements, new features or functionality.  LangChain will have the full, unencumbered right, without any obligation to compensate or reimburse Customer, to use, incorporate and', metadata={'source': 'documents\\langchain-terms-of-service.pdf', 'page': 3})]
print("\nretriever.get_relevant_documents:")
pprint(docs)
print()

# create the chain to answer questions
repo_id = "tiiuae/falcon-7b-instruct"
# repo_id = "bigscience/bloom"
llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.1}
    )

# Trying RetrievalQA chain
# chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     verbose=True,
#     retriever=retriever,
#     return_source_documents=True
#     )
# llm_response = chain(query)
# pprint(llm_response)

# Trying RetrievalQAWithSources chain
# chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm,
#     verbose=True,
#     retriever=retriever,
#     return_source_documents=True
#     )
# llm_response = chain(query)
# pprint(llm_response)

# Trying qa_citation.ipynb
# def format_docs(docs):
#     return "\n\n".join([d.page_content for d in docs])

# chain = create_citation_fuzzy_match_chain(llm)
# result = chain.run(question=query, context=format_docs(docs))
# pprint(result)

# Trying wiscojo's solution
# Function to format each document with an index, source, and content.
def format_document(doc, index, prompt):
    """Format a document into a string based on a prompt template."""
    print("doc from inside format_document():\n", doc)
    print("index from inside format_document():\n", index)
    print("prompt from inside format_document():\n", prompt)
    # Create a dictionary with document content and metadata.
    base_info = {"page_content": doc.page_content, "index": index, "source": doc.metadata['source'] + ", " + str(doc.metadata['page'])}
    
    # Check if any metadata is missing.
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        raise ValueError(f"Missing metadata: {list(missing_metadata)}.")
    
    # Filter only necessary variables for the prompt.
    document_info = {k: base_info[k] for k in prompt.input_variables}
    return prompt.format(**document_info)

# Custom chain class to handle document combination with source indices.
class StuffDocumentsWithIndexChain(StuffDocumentsChain):
    def _get_inputs(self, docs, **kwargs):
        # Format each document and combine them.
        doc_strings = [
            format_document(doc, i, self.document_prompt)
            for i, doc in enumerate(docs, 1)
        ]
        
        # Filter only relevant input variables for the LLM chain prompt.
        inputs = {k: v for k, v in kwargs.items() if k in self.llm_chain.prompt.input_variables}
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs

# Define a chat prompt with instructions for citing documents.
combine_doc_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""Your role is to provide information based on the following sources.
When referencing the documents, add a citation right after. Use "[SOURCE_NUMBER]" for the citation (e.g. "The Space Needle is in Seattle [1][2].").

Sources:
{context}

Chat History:
{chat_history}

Question:
{question}"""
)

# Initialize the custom chain with a specific document format.
combine_docs_chain = StuffDocumentsWithIndexChain(
    llm_chain=LLMChain(
        llm=llm,
        prompt=combine_doc_prompt,
    ),
    document_prompt=PromptTemplate(
        input_variables=["index", "source", "page_content"],
        template="[{index}] {source}:\n{page_content}",
    ),
    document_variable_name="context",
)

convR_qa = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=LLMChain(llm=llm,prompt=combine_doc_prompt),
    combine_docs_chain=combine_docs_chain,
    return_source_documents=True,
    return_generated_question=True,
    verbose=True
    )

llm_response = convR_qa(inputs={"question": query, "chat_history": []})
pprint(llm_response)
