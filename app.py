import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

def init_pinecone():
    # find API key at app.pinecone.io
    pc = Pinecone(api_key=st.secrets["api_key"])
    cloud = 'aws'
    region =  'us-east-1'
    index_name='gif-search'
    if index_name not in pc.list_indexes().names():
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=384,
            metric='cosine',
            spec=pinecone.ServerlessSpec(cloud=cloud, region=region)
        )
    
    return pc.Index('gif-search')
    
def init_retriever():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

index = init_pinecone()
retriever = init_retriever()


def card(urls):
    figures = [f"""
        <figure style="margin-top: 5px; margin-bottom: 5px; !important;">
            <img src="{url}" style="width: 130px; height: 100px; padding-left: 5px; padding-right: 5px" >
        </figure>
    """ for url in urls]
    return st.markdown(f"""
        <div style="display: flex; flex-flow: row wrap; text-align: center; justify-content: center;">
        {''.join(figures)}
        </div>
    """, unsafe_allow_html=True)

 
st.write("""
## ⚡️ AI-Powered GIF Search ⚡️
""")

query = st.text_input("What are you looking for?", "")

if query != "":
    with st.spinner(text="Similarity Searching..."):
        xq = retriever.encode([query]).tolist()
        xc = index.query(vector=xq, top_k=30, include_metadata=True)
        
        urls = []
        for context in xc['matches']:
            urls.append(context['metadata']['url'])

    with st.spinner(text="Fetching GIFs 🚀🚀🚀"):
        card(urls)


st.write("""
## ⚡️ For the official dataset and paper refer below:⚡️
<h3><a href="https://arxiv.org/abs/1604.02748">Paper</a></h3>
""")
