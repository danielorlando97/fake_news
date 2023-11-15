import streamlit as st

st.set_page_config(
    page_title="VeritasCorp",
    page_icon="✨", 
    layout="wide"
)

st.title("✨ VeritasCorp")

st.info("""
Prototype created for VeritasCorp to distinguish Fabulas 
from real news. This prototype is multipolar and 
flexible, supports text inputs, images and 
other news details such as number of comments, 
score and others. 
""")

left, right = st.columns(2)

with right:
    image_upload = st.file_uploader(
        "Upload the image of the news", 
        type=['jpg', 'png', 'jpeg']
    )

    if image_upload:
        with st.expander(
            'See uploaded image', 
            expanded=False
        ):
            st.image(image_upload)

with left:
    text = st.text_input('Write the text of the news', '')
    comments = st.number_input('How many comments does the news have?')
    radio = st.number_input('What is the news up-radio?')
    score = st.number_input('What is the news score?')

 
if st.button('Analyze', disabled=not (
    image_upload and text and comments and radio and score
)):
    pass