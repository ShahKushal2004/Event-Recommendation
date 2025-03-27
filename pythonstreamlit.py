import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process

# Set up the app
st.set_page_config(
    page_title="Mumbai Social Events Recommender",
    page_icon="🌿",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main { background: #f0f2f6; padding: 2rem; }
    .recommendation-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .recommendation-card:hover { transform: translateY(-3px); }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stButton>button:hover { background-color: #45a049; }
    .location-chip {
        display: inline-block;
        background: #e3f2fd;
        padding: 0.25rem 0.75rem;
        border-radius: 16px;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Kushal\Desktop\locations.csv")
        
        # Clean data
        df.columns = df.columns.str.strip()
        df.fillna("", inplace=True)
        
        # Create combined text for recommendations
        df["Combined_Text"] = (df["Category"] + " " + df["Description"]).str.lower()
        
        # Process coordinates
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Initialize models
@st.cache_resource
def initialize_models(df):
    try:
        # Event recommendation model
        event_vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_matrix = event_vectorizer.fit_transform(df["Combined_Text"])
        event_model = NearestNeighbors(n_neighbors=5, metric="cosine")
        event_model.fit(tfidf_matrix)
        
        # Location model
        location_model = NearestNeighbors(n_neighbors=5, metric="haversine")
        location_model.fit(np.radians(df[["Latitude", "Longitude"]]))
        
        return event_vectorizer, event_model, location_model, tfidf_matrix
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        return None, None, None, None

# Recommendation functions
def recommend_events(query, df, vectorizer, model, tfidf_matrix):
    matches = process.extract(query, df['Event Name'].astype(str), limit=5)
    best_match = matches[0][0] if matches and matches[0][1] > 50 else None
    if not best_match:
        return None
    
    idx = df[df['Event Name'] == best_match].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=5)
    return df.iloc[indices[0]]

def recommend_locations(location, df, model):
    # Try exact match first
    location_events = df[df['Location'].str.lower() == location.lower()]
    if not location_events.empty:
        return location_events.head(5)
    
    # Find nearest locations
    try:
        loc_data = df[df['Location'].str.lower() == location.lower()].iloc[0]
        distances, indices = model.kneighbors(
            np.radians([[loc_data["Latitude"], loc_data["Longitude"]]]), 
            n_neighbors=5
        )
        return df.iloc[indices[0]]
    except:
        return df[df['Location'].str.contains(location, case=False)].head(5)

# Main app
def main():
    st.title("🌿 Mumbai Social Events Recommender")
    st.markdown("Discover meaningful social events across Mumbai")
    
    # Load data
    df = load_data()
    if df.empty:
        return
    
    # Initialize models
    event_vectorizer, event_model, location_model, tfidf_matrix = initialize_models(df)
    
    # Recommendation tabs
    tab1, tab2 = st.tabs(["🔍 Find Similar Events", "📍 Find Events by Location"])
    
    with tab1:
        st.subheader("Find Events Similar To...")
        query = st.text_input("Enter an event name:", key="event_query")
        
        if st.button("Find Similar Events", key="event_btn"):
            if query.strip():
                with st.spinner("Finding similar events..."):
                    recs = recommend_events(query, df, event_vectorizer, event_model, tfidf_matrix)
                
                if recs is not None:
                    st.success("Here are similar events you might like:")
                    for _, row in recs.iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>{row['Event Name']}</h4>
                                <p><strong>Category:</strong> {row['Category']}</p>
                                <p><strong>Location:</strong> <span class="location-chip">{row['Location']}</span></p>
                                <p><strong>Date:</strong> {row['Date']}</p>
                                <p><strong>Rating:</strong> {row['Rating']}/5</p>
                                <p><strong>Description:</strong> {row['Description'][:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No similar events found. Try a different name.")
            else:
                st.warning("Please enter an event name")
    
    with tab2:
        st.subheader("Find Events Near...")
        location = st.selectbox(
            "Select a location:", 
            options=sorted(df['Location'].unique()),
            index=0,
            key="location_select"
        )
        
        if st.button("Find Events", key="location_btn"):
            with st.spinner(f"Searching events in {location}..."):
                recs = recommend_locations(location, df, location_model)
            
            if not recs.empty:
                st.success(f"Events in/near {location}:")
                for _, row in recs.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{row['Event Name']}</h4>
                            <p><strong>Category:</strong> {row['Category']}</p>
                            <p><strong>Date:</strong> {row['Date']}</p>
                            <p><strong>Rating:</strong> {row['Rating']}/5</p>
                            <p><strong>Coordinates:</strong> {row['Latitude']:.4f}, {row['Longitude']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"No events found in {location}. Try a nearby location.")

if __name__ == "__main__":
    main()