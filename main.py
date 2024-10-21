import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
model_name = "your-username/english-to-roman-urdu"  # Update with the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Streamlit app
st.title("tehreemmalikkkkk/your-username-english-to-roman-urdu")

# Input prompt
input_text = st.text_area("Enter English text:")

if st.button("Translate"):
    if input_text:
        # Prepare the input for the model
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate translation
        outputs = model.generate(**inputs)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.success(f"Translation: {translation}")
    else:
        st.warning("Please enter some text to translate.")
