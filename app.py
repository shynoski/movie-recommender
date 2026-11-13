import streamlit as st

st.set_page_config(
    page_title="Flick Finder",
    page_icon="🎬",
    layout="wide"
)

# App branding
st.title("🔎 FLICK FINDER")

# Introduction
st.header("Find your next favorite movie!")
st.write("Choose five movies you love to discover what to watch next.")

# Search section
st.subheader("Search for a movie")

search_col, empty_col = st.columns([2, 1])

with search_col:
    search = st.text_input(
        "Movie title",
        placeholder="Search by movie title...",
        label_visibility="collapsed"
    )

# Popular movies section
st.subheader("Popular Movies")
st.caption("Suggested starting points")