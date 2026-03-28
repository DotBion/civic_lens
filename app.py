import streamlit as st
import pandas as pd
from sodapy import Socrata
import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
load_dotenv()
# Must use v1beta for the Gemini 3 Image Preview model
client = genai.Client(http_options={'api_version': 'v1beta'})
socrata_client = Socrata("data.cityofnewyork.us", None)

st.set_page_config(page_title="Civic Advocacy Engine", layout="wide")

# ==========================================
# 2. THE DATA AGENTS (Socrata APIs)
# ==========================================
def get_crash_evidence(lat, lon, radius_meters=500):
    query = f"SELECT number_of_persons_injured, number_of_persons_killed WHERE within_circle(location, {lat}, {lon}, {radius_meters}) AND crash_date > '2024-01-01T00:00:00.000' LIMIT 2000"
    try:
        df = pd.DataFrame.from_records(socrata_client.get("h9gi-nx95", query=query))
        if df.empty: return {"total_injured": 0, "total_killed": 0}
        return {
            "total_injured": int(pd.to_numeric(df['number_of_persons_injured'], errors='coerce').fillna(0).sum()),
            "total_killed": int(pd.to_numeric(df['number_of_persons_killed'], errors='coerce').fillna(0).sum())
        }
    except: return {"total_injured": 0, "total_killed": 0}

def get_311_evidence(lat, lon, radius_meters=500):
    query = f"SELECT complaint_type, status WHERE within_circle(location, {lat}, {lon}, {radius_meters}) AND created_date > '2024-01-01T00:00:00.000' LIMIT 5000"
    try:
        df = pd.DataFrame.from_records(socrata_client.get("erm2-nwe9", query=query))
        if df.empty: return {"total_calls": 0, "top_category": "None"}
        return {
            "total_calls": len(df),
            "top_category": df['complaint_type'].value_counts().idxmax() if not df.empty else "Unknown"
        }
    except: return {"total_calls": 0, "top_category": "None"}

# ==========================================
# 3. THE IMAGE TOOL
# ==========================================
def generate_campaign_image(prompt_text):
    """Calls Gemini 3 Pro Image Preview and returns raw bytes for Streamlit."""
    try:
        prompt_parts = [types.Part.from_text(text=prompt_text)]
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"], 
            image_config=types.ImageConfig(aspect_ratio="16:9")
        )
        result = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[types.Content(role="user", parts=prompt_parts)],
            config=config
        )
        
        if result.candidates and result.candidates[0].content.parts:
            for part in result.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
        return None
    except Exception as e:
        st.error(f"Image API Error: {e}")
        return None

# ==========================================
# 4. THE AI ORCHESTRATOR & UI
# ==========================================
st.title("🏙️ AI Civic Advocacy Engine")
st.write("Select a community board request to instantly fetch live city data and generate a campaign.")

# The Pre-loaded "Flawless Demo" Database
campaign_database = {
    "🛑 4th Ave & 86th St, Brooklyn (Pedestrian Safety)": {
        "lat": 40.6225, "lon": -74.0284,
        "request": "Reconstruct intersection to add a concrete pedestrian island.",
        "excuse": "Agency supports this request but capital funding is currently unavailable."
    },
    "🚴 Queens Blvd & Yellowstone (Bike Lane)": {
        "lat": 40.7251, "lon": -73.8452,
        "request": "Install concrete jersey barriers for the protected bike lane.",
        "excuse": "Intersection is under study by the DOT."
    }
}

selected_issue = st.selectbox("Choose a Neighborhood Hotspot:", list(campaign_database.keys()))

if st.button("Generate Campaign", type="primary"):
    issue_data = campaign_database[selected_issue]
    
    # STEP 1: Fetch Live Socrata Data
    with st.spinner("📊 Fetching live NYPD and 311 data..."):
        crash_data = get_crash_evidence(issue_data['lat'], issue_data['lon'])
        civic_311_data = get_311_evidence(issue_data['lat'], issue_data['lon'])
        
        combined_payload = {
            "location": selected_issue,
            "community_request": issue_data['request'],
            "city_excuse": issue_data['excuse'],
            "injuries_since_2024": crash_data['total_injured'],
            "total_311_complaints": civic_311_data['total_calls'],
            "top_complaint_type": civic_311_data['top_category']
        }

    # STEP 2: Write the Story via Gemini 2.5
    with st.spinner("✍️ AI Orchestrator writing advocacy narrative..."):
        prompt = f"""
        You are an expert political strategist. Here is the live data: {combined_payload}
        
        Write an aggressive, emotional 3-paragraph advocacy campaign demanding the city fulfill the community request.
        Call out the city's excuse and use the injury and 311 data as proof of danger.
        
        Exactly after the first paragraph, you MUST include this exact tag to trigger the image tool:
        [IMAGE: A photorealistic, gritty street view of an NYC intersection heavily congested with traffic, highly detailed]
        """
        
        text_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        raw_story = text_response.text

    # STEP 3: Parse the Story and Generate the Image
    with st.spinner("🎨 Generating multimodal visual evidence..."):
        st.divider()
        
        # Split the text by the [IMAGE: ...] tag
        parts = re.split(r'(\[IMAGE:.*?\])', raw_story)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if part.startswith('[IMAGE:'):
                image_prompt = part.replace('[IMAGE:', '').replace(']', '').strip()
                image_bytes = generate_campaign_image(image_prompt)
                
                if image_bytes:
                    # Render the beautiful 16:9 image natively!
                    st.image(image_bytes, caption="AI-Generated Scenario Visual", use_container_width=True)
            else:
                # Render the emotional text
                st.markdown(part)
                
    st.success("Campaign Generated Successfully! Ready for presentation.")