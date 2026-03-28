import streamlit as st
import pandas as pd
import google.generativeai as genai
from sodapy import Socrata
from thefuzz import fuzz
from geopy.geocoders import Nominatim
import json

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="NYC Budget Agent", page_icon="🗽", layout="wide")

# Use st.secrets for production or .env for local
# For local testing, ensure your .streamlit/secrets.toml has these keys
GEMINI_KEY = st.secrets.get("GEMINI_KEY")
NYC_TOKEN = st.secrets.get("NYC_TOKEN")

if not GEMINI_KEY:
    st.error("Missing Gemini API Key. Please add it to your secrets.")
    st.stop()

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-2.0-flash') # Updated to latest stable

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

@st.cache_data
def get_search_intent(user_query):
    prompt = f"""
    Analyze this query: '{user_query}'
    1. Identify a single broad search keyword for a city database (e.g., 'safety', 'lighting', 'parks', 'traffic').
    2. Identify the target NYC Borough (Manhattan, Bronx, Brooklyn, Queens, Staten Island).
    Return ONLY a valid JSON object: {{"keyword": "word", "borough": "Borough Name"}}
    """
    try:
        response = model.generate_content(prompt).text.strip().replace('```json', '').replace('```', '')
        return json.loads(response)
    except:
        return {"keyword": "safety", "borough": ""}

@st.cache_data
def get_coordinates(address, boro):
    if not address or str(address).strip().lower() in ['nan', 'none', '']:
        return None, None
    geolocator = Nominatim(user_agent="nyc_budget_app_v1")
    try:
        full_query = f"{address}, {boro}, New York, NY"
        location = geolocator.geocode(full_query, timeout=10)
        if location:
            return location.latitude, location.longitude
    except:
        return None, None
    return None, None

@st.cache_data
def fetch_nyc_data(keyword, target_boro):
    client = Socrata("data.cityofnewyork.us", app_token=NYC_TOKEN, timeout=120)
    dataset_id = "vn4m-mk4t"
    
    where_clause = """
    (response LIKE '%Unable to prioritize%' OR response LIKE '%Further study%' OR response LIKE '%not funded%')
    AND (street IS NOT NULL OR site_street IS NOT NULL OR cross_street_1 IS NOT NULL)
    """
    
    boro_reverse_map = {"Manhattan": "1", "Bronx": "2", "Brooklyn": "3", "Queens": "4", "Staten Island": "5"}
    if target_boro in boro_reverse_map:
        where_clause += f" AND boro = '{boro_reverse_map[target_boro]}'"

    results = client.get(dataset_id, q=keyword, where=where_clause, limit=300)
    client.close()
    return pd.DataFrame(results)

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.title("🗽 NYC Budget Request Finder")
st.markdown("Find unfunded community requests near you using AI and Open Data.")

# Sidebar for inputs
with st.sidebar:
    st.header("Search Parameters")
    user_query = st.text_area("What issue are you looking for?", 
                              placeholder="e.g. I live in Bay Ridge, what are the traffic safety issues?")
    run_button = st.button("Search Budget Records", type="primary")

if run_button and user_query:
    with st.spinner("Analyzing intent and searching NYC database..."):
        # 3a. Intent Analysis
        intent = get_search_intent(user_query)
        st.info(f"🔎 **AI Intent:** Keyword: `{intent['keyword']}` | Borough: `{intent['borough']}`")
        
        # 3b. Data Fetching
        df = fetch_nyc_data(intent['keyword'], intent['borough'])
        
        if df.empty:
            st.warning("No records found for that specific search. Try a broader keyword.")
        else:
            # 3c. Scoring
            def score_row(row):
                content = f"{row.get('request','')} {row.get('explanation','')} {row.get('street','')}".lower()
                return fuzz.token_set_ratio(user_query.lower(), content)

            df['match_score'] = df.apply(score_row, axis=1)
            top_record = df.sort_values(by='match_score', ascending=False).iloc[0]

            # 3d. Formatting Results
            boro_map = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}
            boro_name = boro_map.get(str(top_record.get('boro')), "New York")
            
            # Location String Logic
            site = top_record.get('site_street', '')
            cross = top_record.get('cross_street_1', '')
            loc_str = f"{site} & {cross}" if site and cross else (top_record.get('street') or site or cross)

            # 3e. Display Results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🎯 Best Match Found")
                st.metric("Match Confidence", f"{top_record['match_score']}%")
                
                st.markdown(f"**Agency:** {top_record.get('responsible_agency')}")
                st.markdown(f"**Location:** {loc_str}, {boro_name}")
                
                with st.expander("View Full Request Details", expanded=True):
                    st.write(f"**Request:** {top_record.get('request')}")
                    st.write(f"**Official Response:** {top_record.get('response')}")

            with col2:
                # Geocoding & Map
                lat, lon = get_coordinates(loc_str, boro_name)
                if lat and lon:
                    st.subheader("📍 Location Map")
                    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
                    st.map(map_data)
                    st.caption(f"Coordinates: {lat}, {lon}")
                else:
                    st.warning("Could not pin exact coordinates on the map.")

            # 4. Handoff Payload
            st.divider()
            st.subheader("📦 Agent Handoff Payload")
            payload = {
                "query_intent": user_query,
                "matched_issue": top_record.get('request'),
                "agency": top_record.get('responsible_agency'),
                "location": loc_str,
                "lat": lat,
                "lon": lon
            }
            st.json(payload)

else:
    st.light() # Show instructions if no search has run
    st.info("Enter a query in the sidebar to begin.")
