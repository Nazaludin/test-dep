import os
# import nltk
import ssl
import streamlit as st
import random
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline, Conversation

# ssl._create_default_https_context = ssl._create_unverified_context
# nltk.data.path.append(os.path.abspath("nltk_data"))
# try:
#     # Coba download 'punkt'
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     # Jika 'punkt' belum didownload, download sekarang
#     try:
#         _create_unverified_https_context = ssl._create_unverified_context
#     except AttributeError:
#         pass
#     else:
#         ssl._create_default_https_context = _create_unverified_https_context

#     nltk.download('all')

# st.set_page_config(
#     page_title="Chatbot",
#     page_icon="💬",
#     layout="centered",
#     initial_sidebar_state="auto",
#     menu_items=['None']
# )


conversation_history = []
chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")
intents = [
    {
        "tag": "greeting1",
        "patterns": ["Hi", "Hello", "Hey"],
        "responses": ["Hi there", "Hello", "Hey"]
    },
    {
        "tag": "greeting2",
        "patterns": ["How are you?"],
        "responses": ["I'm fine, Thank you"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


def chatbot_one(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]

    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

    # Jika tidak ada respons yang sesuai dengan tag yang diprediksi
    return "Maaf, saya tidak mengerti pertanyaan Anda saat ini."


counter = 0


def chatbot_two(prompt):
    conversation = Conversation(prompt)
    chatbot_response = chatbot(conversation)
    return chatbot_response.generated_responses[-1]


def is_common_question(user_input):
    for intent in intents:
        for pattern in intent['patterns']:
            if pattern.lower() in user_input.lower():
                return True
    return False


def main():
    menu_options = ['Home', 'About', 'Profile']
    selected_option = st.sidebar.selectbox('Menu', menu_options)

    if selected_option == 'About':
        about()
    if selected_option == 'Home':
        chatbot_page()
    if selected_option == 'Profile':
        profile()

  # Add a placeholder to ensure space at the bottom of sidebar
    st.sidebar.markdown('<div style="height: 200px;"></div>',
                        unsafe_allow_html=True)

    # Add a short description about the developer at the bottom of the sidebar
    st.sidebar.markdown(
        """
        <div style="position: fixed; bottom: 0; width: 200px;">
            <hr style="margin: 0;">
            <h2 style="font-style: italic; font-size: 12px;">👨‍💻 Developer</h2>
            <h4 style="font-size: 12px;">About the Chatbot</h4>
            <p style="font-size: 12px;">This chatbot was developed by Nazaludin using Streamlit. It utilizes a model trained on 400 million Facebook public domain conversations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def chatbot_page():
    st.title("Naza Chatbot")
    st.write("Please type a message and press Enter to start the conversation.")
    st.divider()

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    user_input = st.chat_input("Say something")

    response_chat = ""  # Initialize response_chat outside the conditional blocks

    if user_input:
        st.session_state['history'].append(("User", user_input))

        if is_common_question(user_input):
            response_chat = chatbot_one(user_input)
            st.session_state['history'].append(("Chatbot 1", response_chat))
        else:
            response_chat = chatbot_two(user_input)
            st.session_state['history'].append(("Chatbot 2", response_chat))

    # Display chat history
    for speaker, message in st.session_state['history']:
        if speaker == "User":
            with st.chat_message("user"):
                st.write(f"You: {message}")
        else:
            with st.chat_message("assistant"):
                st.write(f"Assistant: {message}")


def about():
    st.title('About')

    st.title("Welcome to this Chatbot!")
    st.write("We are here to provide you with a responsive and informative service.")
    st.write("This chatbot has been designed with advanced technology that enables quick and relevant responses to your questions or needs. With the use of innovative Natural Language Processing (NLP) technology, this chatbot is able to understand the context of the conversation better, thus providing more appropriate and precise answers.")
    st.write("We are committed to continuously improving the capabilities of this chatbot to make it more responsive and adaptive to your needs. Every interaction you have with this chatbot helps us to continue to enrich and develop its capabilities.")
    st.write("Feel free to ask questions or share what you need from this chatbot. We are here to help you!")


def profile():
    st.title('Profil')

    # Path menuju gambar di folder src
    # Ganti dengan path menuju gambar yang Anda miliki
    image_path = "src/nazaludin.png"

    # Tampilkan gambar dengan PIL dan Streamlit
    image = Image.open(image_path)
    st.image(image, width=200)
    st.write("Nazaludin Nur Rahmat")
    st.write("21537141030")
    st.write("S1 - Teknologi Informasi")


if __name__ == '__main__':
    main()
