import streamlit as st

st.set_page_config(
    page_title="Flick Finder",
    page_icon="🎬",
    layout="wide"
)
def add_movie(title):
    if "picks" not in st.session_state:
        st.session_state.picks = []
    if title not in st.session_state.picks and len(st.session_state.picks) < 5:
        st.session_state.picks.append(title)

# App branding
st.title("🔎 FLICK FINDER")

# Introduction
st.header("Find your next favorite movie!")
st.write("Choose five movies you love to discover what to watch next.")

# Search section
st.subheader("Search for your favorite movie")

search_col, empty_col = st.columns([2, 1])

with search_col:
    search = st.text_input(
        "Movie title",
        placeholder="Search by movie title...",
        label_visibility="collapsed"
    )

# Popular movies section
st.subheader("Popular Movies others like")
st.caption("Suggested starting points")


popular = [
    ("Forrest Gump (1994)", "Comedy, Drama, Romance"),
    ("Shawshank Redemption, The (1994)", "Crime, Drama"),
    ("Pulp Fiction (1994)", "Comedy, Crime, Drama"),
    ("Matrix, The (1999)", "Action, Sci-Fi, Thriller"),
    ("Toy Story (1995)", "Adventure, Animation, Children"),
    ("Jurassic Park (1993)", "Action, Adventure, Sci-Fi"),
]

per_row = 3
for start in range(0, len(popular), per_row):
    cols = st.columns(per_row)
    for col, (title, genres) in zip(cols, popular[start:start + per_row]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(genres)
                st.button(
                    "➕ Add",
                    key=f"add_{title}",
                    on_click=add_movie,
                    args=(title,),
                    disabled=title in st.session_state.get("picks", [])
                             or len(st.session_state.get("picks", [])) >= 5,
                )

# Selected favorites progress
if "picks" not in st.session_state:
    st.session_state.picks = []

picks = st.session_state.picks
st.subheader("Your selected favorite movies")
st.progress(len(picks) / 5, text=f"{len(picks)} of 5 selected")
for title in picks:
    st.write(f"• {title}")


if "recs" not in st.session_state:
    st.session_state.recs = []

def generate_recs():
    #TEMPORARY placeholder results
    st.session_state.recs = [
        "Placeholder Movie A (2000)",
        "Placeholder Movie B (2001)",
        "Placeholder Movie C (2002)",
        "Placeholder Movie D (2003)",
        "Placeholder Movie E (2004)",
    ]
st.markdown(
    """
    <style>
    .st-key-recs_btn button {
        height: 5em;
    }
    .st-key-recs_btn button p {
        font-size: 2em;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    st.button(
        "🎬 Get your recommendations",
        key="recs_btn",
        type="primary",
        on_click=generate_recs,
        disabled=len(st.session_state.picks) != 5,
        use_container_width=True,
    )

if st.session_state.recs:
    st.subheader("Recommended for you")
    for title in st.session_state.recs:
        st.write(f"• {title}")