import streamlit as st
import pandas as pd
# import numpy as np
# import altair as alt
# import pydeck as pdk
import json


# @st.cache(persist=True)
def load_data():
    with open('final_corpus.json', 'r') as fp:
        data = json.load(fp)

    return data


data = load_data()

"""
# Streamlit

Streamlit is an open-source app framework for Machine Learning and 
Data Science teams. Create beautiful data apps in hours, not weeks. 
All in pure Python..
"""

"""
## Use Case
Text Analytics on BBC News Articles
"""

# st.markdown("# Documents")

documents = data['docs']

lookup = {label: i for i, label in enumerate(list(documents.keys()))}

# The user can pick which type of object to search for.
object_type = st.selectbox("Select Article", list(documents.keys()),
                           5)

f"""
{documents[object_type].encode("ascii", "ignore")}
"""


def named_entites():

    ne_type = list()
    ne_text = list()

    for item in data["named_entity"][lookup[object_type]]:
        ne_type.append(item["Type"])
        ne_text.append(item["Text"])

    ne_df = pd.DataFrame({"Word": ne_text, "Type": ne_type})

    st.table(ne_df)


def key_phrases():

    kp_text = list()

    for item in data["key_phrases"][lookup[object_type]]:
        kp_text.append(item["Text"])

    kp_df = pd.DataFrame({"Type": kp_text})

    st.table(kp_df)
    # st.write(kp_df)


def sentiment():
    doc_sentiment = data["sentiment_list"][lookup[object_type]]

    doc_senti_type = ''

    if doc_sentiment["Sentiment"] == "POSITIVE":
        doc_senti_type = '<span style="color:#228B22"> *Positive*' \
                         '</span>'
    elif doc_sentiment["Sentiment"] == "NEGATIVE":
        doc_senti_type = '<span style="color:#FF4500"> *Negative*' \
                         '</span>'
    elif doc_sentiment["Sentiment"] == "NEUTRAL":
        doc_senti_type = '<span style="color:#4169E1"> *Neutral*' \
                         '</span>'
    elif doc_sentiment["Sentiment"] == "MIXED":
        doc_senti_type = '<span style="color:#BC8F8F"> *Mixed*' \
                         '</span>'

    st.markdown(f'### Sentiment is {doc_senti_type}', unsafe_allow_html=True)

    f"""
    
    * **Positive**: {doc_sentiment["Score"]["Positive"]}
    * **Negative**: {doc_sentiment["Score"]["Negative"]}
    * **Neutral**: {doc_sentiment["Score"]["Neutral"]}
    * **Mixed**: {doc_sentiment["Score"]["Mixed"]}
    
    """


def topics():
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    # import matplotlib.colors as mcolors

    topics = data["topics"][lookup[object_type]]

    # cols = [color for name, color in
    #         mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',

                      prefer_horizontal=1.0)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        if i == 5:
            continue
        topic_words = topics[i]['terms']
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud, interpolation='bilinear')
        plt.gca().set_title(f"Topic-{i + 1} ({topics[i]['title'].upper()})",
                            fontdict=dict(size=20), pad=20)
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=10, y=10)
    plt.tight_layout()
    # plt.show()
    st.pyplot()


option = st.radio("Select Action",
                  ["Named Entities", "Key Words", "Sentiment", "Topics"], 0)

if option == "Named Entities":
    """
    ## Named Entities
    """
    named_entites()
elif option == "Key Words":
    """
    ## Key Phrases
    """
    key_phrases()
elif option == "Sentiment":
    """
    ## Sentiment Detection
    """
    sentiment()
elif option == "Topics":
    """
    ## Topics
    """
    topics()
