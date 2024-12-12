import openai
import time

# Your OpenAI API key
OPENAI_API_KEY = ''
openai.api_key = OPENAI_API_KEY

def call_chatgpt(conversation):
    """
    Call OpenAI API to generate a response based on the prompt.
    """
    # Add a system message for the conversation
    conversation.insert(0, {
        "role": "system",
        "content": "You are a helpful and empathic assistant. Your responses should be concise, around 63 characters long."
    })

    for _ in range(15):  # Retry up to 5 times
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=conversation,
                temperature=0.7,
                max_tokens=300
            )
            return response['choices'][0]['message']['content']
        except openai.error.RateLimitError:
            print("Rate limit reached. Retrying in 15 seconds...")
            time.sleep(15)  # Wait for 10 seconds before retrying
        except Exception as e:
            return f"Error: {e}"
    return "Error: Max retries reached."

def generate_human_responses(bot_file, generated_file):
    """
    Generate <human> lines for each <bot> line without maintaining prior context.
    Save the generated data to a file.
    """
    try:
        print(f"Reading bot lines from: {bot_file}")
        with open(bot_file, "r", encoding="utf-8") as infile, open(generated_file, "w", encoding="utf-8") as outfile:
            for line_number, line in enumerate(infile, 1):
                line = line.strip()

                if line.startswith("<bot>"):
                    # Extract the <bot> message content
                    bot_message_content = line.replace("<bot>", "").replace("<endOfText>", "").strip()

                    # Create a single pair context
                    conversation = [{"role": "assistant", "content": bot_message_content}]
                    print(f"Processing <bot> line {line_number}: {bot_message_content}")

                    # Generate <human> response
                    human_response = call_chatgpt(conversation)
                    human_response += " <endOfText>"  # Append <endOfText> to the response

                    # Write the pair to the output file
                    outfile.write(f"<bot> {bot_message_content}\n")
                    outfile.write(f"<human> {human_response}\n")
                    print(f"Generated <human> response: {human_response}")
                else:
                    print(f"Skipped line {line_number}: {line}")

        print(f"Generated data saved to {generated_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define file paths
    bot_file = "/Users/sofiagermer/Desktop/SOFIA/IAS/WinterSemester_24_25/ml_proj/Project-ML/data/data_generation/bot_data_temp.txt"
    #generated_file = "/Users/sofiagermer/Desktop/SOFIA/IAS/WinterSemester_24_25/ml_proj/Project-ML/data/data_generation/generated_data.txt"
    generated_file_2 = "/Users/sofiagermer/Desktop/SOFIA/IAS/WinterSemester_24_25/ml_proj/Project-ML/data/data_generation/generated_data_4.txt"
    
    print("Starting the bot response generation process...")
    generate_human_responses(bot_file, generated_file_2)
    print("Process completed!")
