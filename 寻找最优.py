import os
import pandas as pd
import qianfan
from sklearn.metrics import accuracy_score

# Set environment variables
os.environ["QIANFAN_ACCESS_KEY"] = "xx"
os.environ["QIANFAN_SECRET_KEY"] = "xx"

# Load Excel file
file_path = '/Desktop/sample.xlsx'
df = pd.read_excel(file_path)

# Initialize the model
chat_comp = qianfan.ChatCompletion()

# Define the function to calculate accuracy
def calculate_accuracy(predictions, targets):
    return accuracy_score(targets, predictions)

# Define the function to generate responses
def generate_response(text, temperature, top_p, penalty_score):
    prompt_text = f"忘掉所有之前的指令。你现在是一位金融、管理与会计专家。我会给出一个投资者对上市公司comment的文本，你需要回答这个comment文本是暗示投资者trust还是不trust这家公司。请在“trust”、“不trust”、“unknown”中只选择一个选项，并且不要提供任何额外的回答：{text}"
    try:
        response = chat_comp.do(
            endpoint="ernie-speed-128k",
            user_id="user",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            temperature=temperature,
            top_p=top_p,
            penalty_score=penalty_score
        )
        response_text = response['body']['result']  # Extract the actual response text
        print(f"Temperature: {temperature}, Top_p: {top_p}, Penalty_score: {penalty_score}, Text: {text}, Response: {response_text}")
        return response_text
    except Exception as e:
        error_message = f"Error: {e}"
        print(error_message)
        return error_message

# Define the function to extract keywords
def extract_keyword(response):
    if response.startswith("trust"):  
        return "trust" 
    elif response.startswith("不trust"):  
        return "不trust" 
    elif response.startswith("unknown"):  
        return "unknown" 
    else:
        return "unknown"  

# Try different combinations of temperature, top_p, and penalty_score
best_combination = None
best_accuracy = 0
temperature_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # Specific temperature values
top_p_values = [0.5, 0.7, 0.9]  # Specific top_p values
penalty_score_values = [0.5, 1, 1.5]  # Specific penalty_score values

for temp in temperature_values:
    for top_p in top_p_values:
        for penalty_score in penalty_score_values:
            predictions = []
            for text in df['comment']:  # "comment" column from the dataframe
                response = generate_response(text, temp, top_p, penalty_score)
                keyword = extract_keyword(response)
                predictions.append(keyword)
                print(f"Extracted keyword: {keyword}")
            
            # Calculate accuracy for the current combination
            accuracy = calculate_accuracy(predictions, df['text classification'])  # "text classification" column
            print(f"Temperature: {temp}, Top_p: {top_p}, Penalty_score: {penalty_score}, Accuracy: {accuracy}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = (temp, top_p, penalty_score)

# Output the best combination and accuracy
print(f"Best combination: Temperature: {best_combination[0]}, Top_p: {best_combination[1]}, Penalty_score: {best_combination[2]}, Accuracy: {best_accuracy}")
