import streamlit as st

# Streamlit UI
st.title("Movie Recommender System")

# Add two buttons: Based on Category and Based on Description/Review
button1 = st.button("Based on Category")
button2 = st.button("Based on Description/Review")

# Placeholder text for each button's functionality
if button1:
    st.subheader("Movie Recommendations Based on Category")
    st.write("Select your preferences based on categories like genre, target audience, and more.")
    # Here, you'll add functionality later to recommend based on categories

elif button2:
    st.subheader("Movie Recommendations Based on Description/Review")
    st.write("Provide a movie description or review, and we'll recommend movies based on that.")
    # Here, you'll add functionality later to recommend based on description or review

# Optional: You can display information for each option
st.sidebar.markdown("""
### How to Use:
- **Based on Category**: Select movie categories and preferences.
- **Based on Description/Review**: Enter a movie description or review, and get recommendations.
""")
