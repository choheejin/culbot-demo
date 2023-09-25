from flask import Flask, render_template, request, jsonify
import tiktoken as tiktoken
import os
import re
import time
import openai
import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = float(0.2)
openai.api_key = OPENAI_API_KEY


app = Flask(__name__)

def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])

def openai_call(
    system_prompt: str,
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
                # TODO: different limits for different models.

                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

                # Use chat completion API
                messages = [{"role": "system" , "content": system_prompt},{"role": "system", "content": trimmed_prompt}]
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
    df = pd.DataFrame({
                'id':results['ids'][0],
                'score':results['distances'][0],
                'question': dataframe[dataframe.vector_id.isin(list(map(int, results['ids'][0])))]['question'],
                'answer': dataframe[dataframe.vector_id.isin(list(map(int, results['ids'][0])))]['answer'],
                })
    return df

def question_variations(x):
    system_prompt1 = "You are an agent that generically modifies incoming questions as input."
    prompt1 = f"""
            Please modify the following question into a generic one.
            Here is an example.

            ```
            ```

            Question: {x}
            After:

            """
    response1 = openai_call(system_prompt1, prompt1, max_tokens=2000)
    print("변형 질문:: ", response1)

    return response1

def vector_QA(question):
    title_query_result1 = query_collection(
        collection=medical_question_collection,
        query=question,
        max_results=10,
        dataframe=df,
    )

    qa_set1 = ''
    question_list1 = title_query_result1.question
    answer_list1 = title_query_result1.answer
    for idx, (q, a)  in enumerate(zip(question_list1, answer_list1)):
        qa_set1 += "QA {}.\n question: {} answer: {} \n ".format(idx, q, a)
        # qa_list.append(" question: {} answer: {}".format(q, a))

    print(qa_set1)
    return qa_set1


def select_QA(question, qa_set):
    system_prompt2 = "you are an AI Docter"
    prompt2 = f"""
            I will provide ten 'Question-Answer Sets' that appear to be related to 'Question'.
            Do not return the result until you calculate the similarity score yourself.
            The content of 'Question' and the ten 'Question-Answer Sets' should be similar down to specific details.
            Please select the 'Question-Answer Set' that appears most similar to 'Question'.
            But, If you believe that 'Question' and the ten 'Question-Answer Sets' seem unrelated, return 'I don't know.'.

            Question: ```{question}```
            Qusetion-Answer Set: ```{qa_set}```
            Result:

            """
    response2 = openai_call(system_prompt2, prompt2, max_tokens=2000)

    print("선택 응답:: ", response2)
    return response2

def generate_answer(question, qa_set):
    system_prompt3 = "you are an AI Docter"
    prompt3 = f"""
            Provide an appropriate answer to the question based on the given Question-Answer set \n

            Question:
                `
                    {question}
                `

            Qusetion-Answer Set
                `
                    {qa_set}
                `
            """

    response3 = openai_call(system_prompt3, prompt3, max_tokens=2000)
    return response3

@app.route('/post', methods=['POST'])
def post():
    if request.method == "POST":
        user = request.form["user"]
        question = question_variations(user)
        time.sleep(1)
        qa_set = vector_QA(question)
        time.sleep(1)
        response = select_QA(question, qa_set)
        time.sleep(1)

        ## 무조건 교수님 응답
        match_q = re.search(r'question:\s*(.*?)\s*answer:', response, re.DOTALL | re.IGNORECASE)
        match_a = re.search(r'answer:\s*(.*?)\s*$', response, re.DOTALL | re.IGNORECASE)

        if match_a and match_q:
            professor_question = match_q.group(1)
            professor_ans = match_a.group(1)
        else:
            professor_question = ''
            professor_ans = ''

        ## GPT 응답
        bot_message = generate_answer(question, qa_set)
        time.sleep(1)

        return jsonify({'result': {'user': user, 'professor_ans': professor_ans, 'professor_question': professor_question, 'bot': bot_message}})



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    chat_history=[]
    chroma_client = chromadb.Client()

    if OPENAI_API_KEY is not None:
        openai.api_key = OPENAI_API_KEY
        print ("OPENAI_API_KEY is ready")
    else:
        print ("OPENAI_API_KEY environment variable not found")


    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name='text-embedding-ada-002')

    medical_question_collection = chroma_client.create_collection(name='medical_question', embedding_function=embedding_function)
    medical_answer_collection = chroma_client.create_collection(name='medical_answer', embedding_function=embedding_function)

    df = pd.read_json("./after_embedding2.json")
    df.head()
    batch_size = 166
    chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    # print(chunks)
    # Add the content vectors in batches
    for chunk in chunks:
        # print(chunk['vector_id'])
        medical_question_collection.add(
            ids=list(map(str, chunk['vector_id'])),
            embeddings=chunk['question_vector'].tolist(),  # Assuming you have the 'question_vector' column
        )

    # Assuming you have your medical_answer_collection and answer_vector ready

    # Add the title vectors in batches
    for chunk in chunks:
        medical_answer_collection.add(
            ids=list(map(str, chunk['vector_id'])),
            embeddings=chunk['answer_vector'].tolist(),  # Assuming you have the 'answer_vector' column
        )
    app.run(host='localhost', port=8800, debug=True)
