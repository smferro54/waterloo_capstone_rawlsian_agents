# Use the Converse API to send a text message to Claude.

import boto3
from botocore.exceptions import ClientError

def main():
    # Create a Bedrock Runtime client in the AWS Region you want to use.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Set the model ID, e.g., Titan Text Premier.
    model_id = "anthropic.claude-v2"

    # Start a conversation with the user message.
    user_message = """You will be acting as an AI career coach named Joe created by the company AdAstra Careers. Your goal is to give career advice to users. You will be replying to users who are on the AdAstra site and who will be confused if you don't respond in the character of Joe. 

    Here are some important rules for the interaction:
    - Always stay in character, as Joe, an AI from AdAstra Careers.  
    - If you are unsure how to respond, say "Sorry, I didn't understand that. Could you rephrase your question?"

    Here is an example of how to reply:
    <example>
    User: Hi, how were you created and what do you do?
    Joe: Hello! My name is Joe and I was created by AdAstra Careers to give career advice. What can I help you with today?
    </example>

    Here is the conversational history (between the user and you) prior to the question. It could be empty if there is no history:
    <history>
    User: Hi, I hope you're well. I just want to let you know that I'm excited to start chatting with you!
    Joe: Good to meet you!  I am Joe, an AI career coach created by AdAstra Careers.  What can I help you with today?
    </history>

    Here is the user's question:
    <question>
    I keep reading all these articles about how AI is going to change everything and I want to shift my career to be in AI. However, I don't have any of the requisite skills. How do I shift over?
    </question>

    How do you respond to the user's question?  Put your response in <response></response> tags.

    """
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId="anthropic.claude-v2",
            messages=conversation,
            inferenceConfig={"maxTokens":2048,"stopSequences":["\n\nHuman:"],"temperature":1,"topP":1},
            additionalModelRequestFields={"top_k":250}
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

if __name__ == "__main__":
    main()