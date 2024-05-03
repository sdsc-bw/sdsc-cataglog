from data_requests.localgpt_request import get_response_from_localgpt

def get_main_theme(use_case):
    prompt = f"What is the theme studied in the following use case, please answer only with a keyword less then 20 letter: {use_case}"

    response = get_response_from_localgpt(prompt)
    return response

def get_keywords(use_case):
    prompt = f"What is the theme studied in the following use case, what keywords describe the use case, please give me 3 suggestions and seperate them with comma, without explanation: {use_case}"

    response = get_response_from_localgpt(prompt)

    keywords = response.split(',', 2)
    
    return keywords