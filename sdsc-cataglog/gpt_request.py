import openai

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