
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

loader = PyPDFLoader("/workspaces/codespaces-blank/SMTP, POP3, IMAP.pdf")
data = loader.load()
# print(len(data)) #its actually pages how many pages are there


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
# print("Total no of documents : " , len(docs))
# basically previously we have 18 pages as data each page have differect length of characters
# but now we split that into equal no of chuncks that is 1000
# so each docs contains 1000 characters


# Now initialize your embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# embedding converts the data into numberic values and chroma is a vector database to store the numeric values

# so now we have to embed all the documents
vectorStore = Chroma.from_documents(documents = docs, embedding = embeddings)

#  retrievel will retrive the related docs which user asks from the database
retriever = vectorStore.as_retriever(search_type = "similarity", search_kwargs = {"k" : 10})

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temprature = 0.2, max_tokens = None)

#  now creating a chain

system_prompt = (
    "You are an assitant to generate multiple choice question on the given topic. "
    "four options must be given and also give the correct answer"
    "give the response in the form of json object. "
    "like one attribute could question next is options these must be array the correct_ans"
    "Use the following pieces of retrived context to generate the questions "
    "give only one question at a time"
    "do not repeat the question"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, prompt
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
reg_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

how_many_questions = int(input("How many questions do you want to generate: "))
chat_history = []
while(how_many_questions > 0):
    context = history_aware_retriever.invoke({"input": "give me the question on the topic of SMTP"})
    response = reg_chain.invoke({
        "input" : "give me the question on the topic of SMTP",
        "chat_history" : chat_history,
        "context" : context
    })

    mcq = response["answer"].strip()[7:-3].strip()
    mcq = json.loads(mcq)

    print(mcq["question"])

    print("Options:")
    for i, option in enumerate(mcq["options"]):
        print(f"{i+1}. {option}")

    user_ans = int(input("Enter your answer: "))

    if mcq["options"][user_ans-1] == mcq["correct_ans"]:
        print("your Answer is Correct")
    else:
        print("your Answer is Wrong")
        print(f"Correct Answer: {mcq['correct_ans']}")
    chat_history.append(HumanMessage(content=response["answer"]))
    how_many_questions -= 1




