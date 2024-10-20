# Multiple-Choice Question Generator

This project is designed to generate multiple-choice questions based on the content of a PDF document using LangChain and Google Generative AI. The user can specify the number of questions to be generated, and the program ensures that questions are not repeated during the session.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- Generates multiple-choice questions from a PDF document.
- Provides four options for each question, along with the correct answer.
- Keeps track of asked questions to prevent repetition.
- Utilizes embeddings for effective document retrieval.

## Requirements

- Python 3.7 or higher
- LangChain
- langchain-google-genai
- langchain-chroma
- langchain-community
- Other dependencies specified in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/mcq-generator.git
   cd mcq-generator
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure to have the necessary credentials and configurations set up for Google Generative AI.

## Usage

1. Place the PDF document from which you want to generate questions in the project directory. Ensure the path in the code matches your file.

2. Run the script:

   ```bash
   python mcq_generator.py
   ```

3. Follow the prompts to specify the number of questions you wish to generate.

## How It Works

1. **Load PDF Document**: The script loads the PDF document using `PyPDFLoader`.
2. **Text Splitting**: The content is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embeddings**: The text is converted into numerical embeddings using `GoogleGenerativeAIEmbeddings`.
4. **Vector Store**: The embeddings are stored in a vector database using `Chroma`.
5. **Question Generation**: A retrieval chain is created using the history-aware retriever, ensuring that previously asked questions are not repeated.
6. **User Interaction**: The user is prompted to answer the questions, and feedback is provided based on their answers.

## Example Code Snippet

Here's a brief look at the core functionality of the script:

```python
loader = PyPDFLoader("path/to/your/document.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorStore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Create retrieval chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temprature=0.2, max_tokens=None)
reg_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
