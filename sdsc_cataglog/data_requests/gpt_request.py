import openai
from utils.utils import clip_text_according_to_token_number

openai.api_key = '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'



def get_response_from_chatgpt(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.9,
        top_p=1,
    )

    return response.choices[0].text.strip()

def get_response_from_chatgpt_with_context(prompt, context):
    try:
        context.append({'role': 'user', 'content': prompt})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=context
        )
        context.append({'role': 'assistant', 'content': response.choices[0].message.content.strip()})
        return response.choices[0].message.content.strip(), context
    except openai.error.APIError as e:
        print("OpenAI API error:", e)
    except Exception as e:
        print("Exception:", e)

    return None, None

def get_text_embedding_with_openai(text, model="text-embedding-ada-002"):
    """
    针对字符串过长的情况只是简单的进行了截断。
    """
    if text is None:
        return None
    text = text.replace("\n", " ")
    clip_text_according_to_token_number(text, num = 6000)
        
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']