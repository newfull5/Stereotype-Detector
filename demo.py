import pandas as pd
import streamlit as st
import plotly.express as px


def write_header():
    st.title('Korean Stereotype Detector')
    st.markdown('''
        - Write any sentence containing stereotypes 
        - This application is made with TUNiB-Electra and K-StereoSet.
    ''')

cols = ['stereotype','anti-stereotype','unrelated','profession','race','gender','religion']

def write_textbox():
    input_text = st.text_area(label='Write your sentence', key=1, height=40)
    button = st.button(label='Run')


    df = pd.DataFrame(
        [["Product A", 5.6], ["Product B", 5.8]],
        columns=["Product", "Comfort"]
    )

    fig = px.bar(df, x="Product", y=["Comfort"], barmode='group', height=400)

    if button:
        with st.spinner(text='This may take a moment...'):
            output = ('input_text')
        st.plotly_chart(fig)

    else:
        st.plotly_chart(fig)




if __name__ == '__main__':
    st.set_page_config(page_title='Korean stereotype detector', page_icon='☮️', layout='wide')
    write_header()
    write_textbox()