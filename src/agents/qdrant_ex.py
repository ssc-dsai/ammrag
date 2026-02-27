import os
import uuid
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import QdrantVectorSearchTool
from qdrant_client import QdrantClient
from crewai_tools.tools.qdrant_vector_search_tool.qdrant_search_tool import QdrantConfig

from qdrant_client.models import PointStruct, Distance, VectorParams
from pydantic import BaseModel, Field

# class QdrantConfig(BaseModel):
#     qdrant_url: str
#     qdrant_api_key: str | None = None
#     collection_name: str
#     limit: int = 3
#     score_threshold: float = 0.35
#     filter: Any | None = None

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text.strip())
    return text

# Generate OpenAI embeddings
def get_openai_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Store text and embeddings in Qdrant
def load_pdf_to_qdrant(pdf_path, qdrant, collection_name):
    # Extract text from PDF
    text_chunks = extract_text_from_pdf(pdf_path)

    # Create Qdrant collection
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )

    # Store embeddings
    points = []
    for chunk in text_chunks:
        embedding = get_openai_embedding(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk}
        ))
    qdrant.upsert(collection_name=collection_name, points=points)

# Initialize Qdrant client and load data
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection_name = "nss_test"
pdf_path = "/workspaces/canchat-v2-nss/nss/crewai/data/cache/General Documentation/WBA FAQ Jul 2023.pdf"
load_pdf_to_qdrant(pdf_path, qdrant, collection_name)

# Initialize Qdrant search tool

qdrant_tool = QdrantVectorSearchTool(
    qdrant_config=QdrantConfig(
        qdrant_url=os.getenv("QDRANT_URL","http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name,
        limit=3,
        score_threshold=0.35
    )
)

# qdrant_tool = QdrantVectorSearchTool(
#     collection_name="my_collection",
#     content_field="content", # The field in Qdrant holding the text content
#     url="http://localhost:6333", # Or your Qdrant Cloud URL
#     api_key="your_api_key" # Required for cloud
# )

# Create CrewAI agents
search_agent = Agent(
    role="Senior Semantic Search Agent",
    goal="Find and analyze documents based on semantic search",
    backstory="""You are an expert research assistant who can find relevant
    information using semantic search in a Qdrant database.""",
    tools=[qdrant_tool],
    verbose=True
)

answer_agent = Agent(
    role="Senior Answer Assistant",
    goal="Generate answers to questions based on the context provided",
    backstory="""You are an expert answer assistant who can generate
    answers to questions based on the context provided.""",
    tools=[qdrant_tool],
    verbose=True
)

# Define tasks
search_task = Task(
    description="""Search for relevant documents about the {query}.
    Your final answer should include:
    - The relevant information found
    - The similarity scores of the results
    - The metadata of the relevant documents""",
    agent=search_agent,
    expected_output="search results in point form"
)

answer_task = Task( # type: ignore[index]    
    description="""Given the context and metadata of relevant documents,
    generate a final answer based on the context.""",
    agent=answer_agent,
    context=[search_task],
    expected_output="A short, concise answer based on the context provided"
)

# Run CrewAI workflow
crew = Crew(
    agents=[search_agent, answer_agent],
    tasks=[search_task, answer_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(
    inputs={"query": "What is WBA?"}
)
print(result)