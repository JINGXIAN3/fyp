import streamlit as st

# Title
st.title("🎬 Movie Recommender System")

# Sidebar selection
option = st.sidebar.radio(
    "Choose Recommendation Method:",
    (
        "Collaborative Filtering",
        "Content-Based Filtering (Category Selection)",
        "Content-Based Filtering (Textbox Input)",
        "Hybrid Filtering"
    )
)

# Description box in sidebar
st.sidebar.markdown("""
### How to Use:
- Select a method on the left.
- Follow the instructions that appear in the main area.
""")

# Main content based on selection
if option == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering")
    st.write("Recommend movies based on user-user or item-item similarities.")
    # TODO: Add collaborative filtering logic here

elif option == "Content-Based Filtering (Category Selection)":
    st.subheader("🎯 Content-Based Filtering (Category Selection)")
    st.write("Select preferences like genre, year, or director for recommendations.")
    # TODO: Add category-based content filtering logic here

elif option == "Content-Based Filtering (Textbox Input)":
    st.subheader("📝 Content-Based Filtering (Textbox Input)")
    st.write("Type a movie description or review, and we'll recommend similar movies.")
    user_input = st.text_area("Enter a movie description or review:")
    if user_input and st.button("Get Recommendations"):
        # TODO: Replace with your NLP-based content filter
        st.write("Recommended movies based on your input:")
        # for movie in recommend_from_text(user_input): st.write(movie)

elif option == "Hybrid Filtering":
    st.subheader("🔀 Hybrid Filtering")
    st.write("Combine collaborative and content-based filtering for improved recommendations.")
    # TODO: Add hybrid filtering logic here
