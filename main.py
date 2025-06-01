from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import List, Optional
import uuid
from datetime import datetime
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import json
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
from fastapi.responses import FileResponse
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = FastAPI(title="CreatorFlow AI Backend", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key= OPENAI_API_KEY)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CampaignCreate(BaseModel):
    title: str
    brief: str
    platforms: List[str]
    audience: str
    budget: str

class Campaign(BaseModel):
    id: str
    title: str
    brief: str
    platforms: List[str]
    audience: str
    budget: str
    enhanced_brief: Optional[str] = None
    created_at: datetime

class Creator(BaseModel):
    id: str
    name: str
    handle: str
    platform: str
    followers: str
    engagement: str
    category: str
    location: str
    description: str

class CreatorSearchRequest(BaseModel):
    query: str
    campaign_id: str


class OutreachRequest(BaseModel):
    campaign_id: str
    creator_id: str

class OutreachResponse(BaseModel):
    email_content: str
    audio_url: str

class NegotiationMessage(BaseModel):
    campaign_id: str
    creator_id: str
    message: str
    sender: str 

class DealRequest(BaseModel):
    campaign_id: str
    creator_id: str
    final_rate: str
    deliverables: str
    platform: str
    timeline: str

class ContractRequest(BaseModel):
    deal_id: str


class SimpleOutreachRequest(BaseModel):
    campaign_id: str
    creator_id: str

class SimpleOutreachResponse(BaseModel):
    email_content: str
    audio_url: str

# Helper functions for Supabase operations
async def create_campaign_in_db(campaign_data: dict):
    """Create campaign in Supabase"""
    try:
        result = supabase.table("campaigns").insert(campaign_data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def get_campaign_from_db(campaign_id: str):
    """Get campaign from Supabase"""
    try:
        result = supabase.table("campaigns").select("*").eq("id", campaign_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def update_campaign_in_db(campaign_id: str, update_data: dict):
    """Update campaign in Supabase"""
    try:
        result = supabase.table("campaigns").update(update_data).eq("id", campaign_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def delete_campaign_from_db(campaign_id: str):
    """Delete campaign from Supabase"""
    try:
        result = supabase.table("campaigns").delete().eq("id", campaign_id).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def create_creator_in_db(creator_data: dict):
    """Create creator in Supabase"""
    try:
        result = supabase.table("creators").insert(creator_data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def get_creators_from_db(category: Optional[str] = None, platform: Optional[str] = None):
    """Get creators from Supabase with filters"""
    try:
        query = supabase.table("creators").select("*")
        
        if category:
            query = query.ilike("category", f"%{category}%")
        if platform:
            query = query.ilike("platform", f"%{platform}%")
            
        result = query.execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def get_creator_from_db(creator_id: str):
    """Get creator from Supabase"""
    try:
        result = supabase.table("creators").select("*").eq("id", creator_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def delete_creator_from_db(creator_id: str):
    """Delete creator from Supabase"""
    try:
        result = supabase.table("creators").delete().eq("id", creator_id).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def create_outreach_in_db(outreach_data: dict):
    """Create outreach in Supabase"""
    try:
        result = supabase.table("outreach").insert(outreach_data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def get_outreach_from_db(campaign_id: str, creator_id: str):
    """Get outreach from Supabase"""
    try:
        result = supabase.table("outreach").select("*").eq("campaign_id", campaign_id).eq("creator_id", creator_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def create_deal_in_db(deal_data: dict):
    """Create deal in Supabase"""
    try:
        result = supabase.table("deals").insert(deal_data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def get_deal_from_db(deal_id: str):
    """Get deal from Supabase"""
    try:
        result = supabase.table("deals").select("*").eq("id", deal_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def delete_deal_from_db(deal_id: str):
    """Delete deal from Supabase"""
    try:
        result = supabase.table("deals").delete().eq("id", deal_id).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def generate_simple_voice_message(text: str, campaign_id: str, creator_id: str) -> str:
    """Generate voice message using ElevenLabs API"""
    try:
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        # Voice settings
        voice_settings = {
            "stability": 0.75,
            "similarity_boost": 0.8,
            "style": 0.2,
            "use_speaker_boost": True
        }
        
        # Use a professional voice
        voice_id = "21m00Tcm4TlvDq8ikWAM" 
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": voice_settings
        }
        
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            # Save audio file
            audio_filename = f"outreach_{campaign_id}_{creator_id}_{int(datetime.now().timestamp())}.mp3"
            audio_dir = "static/audio"
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, audio_filename)
            
            with open(audio_path, "wb") as f:
                f.write(response.content)
            
            return f"/api/audio/{audio_filename}"
        else:
            print(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return f"/api/audio/fallback_{campaign_id}_{creator_id}.mp3"
            
    except Exception as e:
        print(f"Voice generation error: {str(e)}")
        return f"/api/audio/fallback_{campaign_id}_{creator_id}.mp3"

async def generate_simple_outreach_content(campaign_data: dict, creator_data: dict) -> tuple:
    """Generate simple outreach content using GPT-4"""
    
    platform_text = ", ".join(campaign_data["platforms"]) if len(campaign_data["platforms"]) > 1 else campaign_data["platforms"][0]
    prompt = f"""
    Create a personalized outreach email for an influencer collaboration.

    
    CAMPAIGN:
    - Title: {campaign_data['title']}
    - Brief: {campaign_data.get('enhanced_brief', campaign_data['brief'])}
    - Target Audience: {campaign_data['audience']}
    - Platforms: {platform_text}
    - Budget: {campaign_data['budget']} INR make it 50% of the budget
    
    CREATOR:
    - Name: {creator_data['name']}
    - Handle: {creator_data['handle']}
    - Platform: {creator_data['platform']}
    - Followers: {creator_data['followers']}
    - Category: {creator_data['category']}
    - Location: {creator_data['location']}
    
    Create:
    1. A professional email with subject line do not include any signature or closing in the email just greet with a thank you
    2. A shorter voice message script for about 40 seconds when spoken
    
    Make it personal, professional, and engaging.
    
    Return as JSON:
    {{
        "email_content": "full email with subject line",
        "voice_script": "shorter version for voice message"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at writing personalized outreach emails for influencer marketing. Always return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        return result.get("email_content", ""), result.get("voice_script", "")
        
    except Exception as e:
        print(f"GPT-4 outreach generation error: {str(e)}")
        
        # Fallback email
        fallback_email = f"""Subject: Collaboration Opportunity - {campaign_data['title']}

        Hi {creator_data['name']},

        I've been following your {creator_data['category']} content on {creator_data['platform']} and I'm impressed by your engagement with your audience.

        We're launching {campaign_data['title']} and think you'd be a perfect fit for our campaign targeting {campaign_data['audience']}.

        Campaign Details:
        {campaign_data.get('enhanced_brief', campaign_data['brief'])}

        Budget: {campaign_data['budget']} INR
        Platform: {platform_text}

        Would you be interested in discussing this collaboration opportunity?

        Best regards,
        CreatorFlow AI Team"""

        fallback_voice = f"Hi {creator_data['name']}, I've been following your {creator_data['category']} content and think you'd be perfect for our {campaign_data['title']} campaign. Would you be interested in discussing a collaboration?"
        
        return fallback_email, fallback_voice
    
async def generate_conversation_summary(messages: list) -> str:
    """Generate AI summary of negotiation conversation using GPT-4"""
    conversation_text = "\n".join([f"{msg['message_type']}: {msg['content']}" for msg in messages])
    
    prompt = f"""
    Summarize this influencer negotiation conversation:

    {conversation_text}

    Include:
    - Key negotiation points discussed
    - Agreed rates and deliverables
    - Timeline and next steps
    - Overall outcome (deal closed, ongoing, declined)
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert conversation summarizer. Always provide a concise and structured summary."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        print(f"GPT-4 conversation summary generation error: {str(e)}")
        return "Unable to generate summary at the moment. Please try again later."


from reportlab.pdfgen import canvas
import os

def create_contract_pdf(content: str, deal_id: str) -> str:
    os.makedirs("static/contracts", exist_ok=True)
    pdf_path = f"static/contracts/{deal_id}.pdf"

    c = canvas.Canvas(pdf_path)
    text_object = c.beginText(50, 800)  # Starting position

    # Set font
    text_object.setFont("Helvetica", 12)

    # Add lines of text
    for line in content.strip().split("\n"):
        text_object.textLine(line.strip())

    c.drawText(text_object)
    c.save()

    return pdf_path



# 1. CAMPAIGN CREATION ROUTES
@app.post("/api/campaigns", response_model=Campaign)
async def create_campaign(campaign: CampaignCreate):
    """Create a new campaign"""
    campaign_id = str(uuid.uuid4())
    
    campaign_data = {
        "id": campaign_id,
        "title": campaign.title,
        "brief": campaign.brief,
        "platforms": campaign.platforms,
        "audience": campaign.audience,
        "budget": campaign.budget,
        "created_at": datetime.now().isoformat()
    }
    
    result = await create_campaign_in_db(campaign_data)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create campaign")
    
    return Campaign(**result)

@app.post("/api/campaigns/{campaign_id}/enhance-brief")
async def enhance_campaign_brief(campaign_id: str):
    """Use AI to enhance campaign brief"""
    campaign_data = await get_campaign_from_db(campaign_id)
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Create platform text for multiple platforms
    platform_text = ", ".join(campaign_data["platforms"]) if len(campaign_data["platforms"]) > 1 else campaign_data["platforms"][0]

    prompt = f"""
    Here is a campaign that needs an enhanced brief.

    Title: {campaign_data['title']}
    Target Audience: {campaign_data['audience']}
    Platforms: {platform_text}
    Original Brief: {campaign_data['brief']}

    Please rewrite it as an engaging influencer brief and do not include the brand name if it is not mentioned.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a seasoned brand strategist who rewrites campaign briefs to make them clear, exciting, and inspiring for modern creators to collaborate."},
            {"role": "user", "content": prompt}
        ]
    )

    enhanced_brief = response.choices[0].message.content.strip()

    await update_campaign_in_db(campaign_id, {"enhanced_brief": enhanced_brief})
    
    return {"enhanced_brief": enhanced_brief}

@app.get("/api/campaigns/{campaign_id}", response_model=Campaign)
async def get_campaign(campaign_id: str):
    """Get campaign details"""
    campaign_data = await get_campaign_from_db(campaign_id)
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return Campaign(**campaign_data)

@app.get("/api/campaigns")
async def get_all_campaigns():
    """Get all campaigns"""
    try:
        result = supabase.table("campaigns").select("*").execute()
        return {"campaigns": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str):
    """Delete campaign"""
    campaign_data = await get_campaign_from_db(campaign_id)
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    await delete_campaign_from_db(campaign_id)
    return {"message": "Campaign deleted successfully"}

# 2. CREATOR DISCOVERY ROUTES
@app.get("/api/creators", response_model=List[Creator])
async def get_creators(
    category: Optional[str] = None,
    platform: Optional[str] = None
):
    """Get list of creators with optional filters"""
    creators_data = await get_creators_from_db(category, platform)
    
    return [Creator(**creator) for creator in creators_data]

@app.post("/api/creators")
async def create_creator(creator: Creator):
    """Create a new creator"""
    creator_data = creator.to_dict()
    if not creator_data.get("id"):
        creator_data["id"] = str(uuid.uuid4())
    
    result = await create_creator_in_db(creator_data)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create creator")
    
    return Creator(**result)

@app.delete("/api/creators/{creator_id}")
async def delete_creator(creator_id: str):
    """Delete creator"""
    creator_data = await get_creator_from_db(creator_id)
    if not creator_data:
        raise HTTPException(status_code=404, detail="Creator not found")
    
    await delete_creator_from_db(creator_id)
    return {"message": "Creator deleted successfully"}


@app.post("/api/creators/search")
async def ai_search_creators(request: CreatorSearchRequest):
    """Advanced AI-powered semantic search for optimal creators"""
    campaign_id = request.campaign_id
    query = request.query
    # Get campaign details for context
    campaign_data = await get_campaign_from_db(campaign_id)
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Get all available creators
    all_creators = await get_creators_from_db()
    if not all_creators:
        return {
            "results": [],
            "query_processed": query,
            "semantic_matches": []
        }
    
    try:
        creators_text = ""
        for idx, creator in enumerate(all_creators):
            creators_text += f"""
            Creator {idx + 1}:
            - Name: {creator['name']}
            - Handle: {creator['handle']}
            - Platform: {creator['platform']}
            - Followers: {creator['followers']}
            - Engagement: {creator['engagement']}
            - Category: {creator['category']}
            - Location: {creator['location']}
            - Description: {creator['description']}
            """
        
        comprehensive_prompt = f"""
        You are an elite influencer marketing AI analyst. Analyze this campaign and score ALL creators in a single response.
        
        CAMPAIGN DETAILS:
        - Title: {campaign_data['title']}
        - Brief: {campaign_data.get('enhanced_brief', campaign_data['brief'])}
        - Target Audience: {campaign_data['audience']}
        - Platforms: {', '.join(campaign_data['platforms'])}
        - Budget: {campaign_data['budget']}
        - Search Query: {query}
        
        CREATORS TO ANALYZE:
        {creators_text}
        
        SCORING CRITERIA (Total 100 points):
        1. Audience Alignment (0-25): How well their audience matches target demographics
        2. Content Relevance (0-25): Alignment with content themes and industry vertical
        3. Platform Optimization (0-20): Platform expertise and content format mastery
        4. Engagement Quality (0-15): Authentic engagement vs follower count ratio
        5. Brand Safety (0-10): Professional reputation and content appropriateness
        6. Geographic Relevance (0-5): Location alignment with campaign needs
        
        ADDITIONAL SCORING FACTORS:
        - Growth Potential Bonus: High (+10), Medium (+5), Low (0)
        - Collaboration Fit Bonus: Excellent (+15), Good (+10), Fair (+5), Poor (0)
        - Performance Bonus: Above Average (+8), Average (+4), Below Average (0)
        - Risk Penalty: -3 per risk factor
        
        Return a JSON object with this EXACT structure:
        {{
            "campaign_requirements": {{
                "target_demographics": ["demographic1", "demographic2"],
                "content_style": ["style1", "style2"],
                "industry_vertical": "primary industry",
                "platform_priorities": ["platform1", "platform2"],
                "content_themes": ["theme1", "theme2", "theme3"]
            }},
            "creator_scores": [
                {{
                    "creator_index": 0,
                    "match_score": 85,
                    "detailed_scores": {{
                        "audience_alignment": 22,
                        "content_relevance": 20,
                        "platform_optimization": 18,
                        "engagement_quality": 14,
                        "brand_safety": 8,
                        "geographic_relevance": 3
                    }},
                    "bonuses": {{
                        "growth_potential": 10,
                        "collaboration_fit": 15,
                        "performance": 8
                    }},
                    "penalties": {{
                        "risk_factors": 0
                    }},
                    "strengths": ["strength1", "strength2"],
                    "collaboration_fit": "excellent",
                    "growth_potential": "high",
                    "estimated_performance": "above_average",
                    "risk_factors": [],
                    "optimal_content_types": ["content_type1", "content_type2"]
                }}
            ],
            "semantic_matches": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
        }}
        
        IMPORTANT: 
        - Score ALL creators provided (creator_index 0 to {len(all_creators)-1})
        - Calculate match_score as: base_score + bonuses - penalties (max 100)
        - Include 5-7 semantic keywords that represent the search intent
        - Ensure valid JSON format
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise influencer analytics AI. Always return valid JSON only. No explanations or additional text."},
                {"role": "user", "content": comprehensive_prompt}
            ],
            temperature=0.1,
            max_tokens=4000 
        )
        
        import json
        analysis_result = json.loads(response.choices[0].message.content.strip())
        
        # Process results and combine with creator data
        scored_creators = []
        creator_scores = analysis_result.get("creator_scores", [])
        
        for score_data in creator_scores:
            creator_idx = score_data.get("creator_index")
            if creator_idx < len(all_creators):
                creator_with_score = all_creators[creator_idx].copy()
                creator_with_score["match_score"] = score_data.get("match_score", 50)
                
                creator_with_score["ai_insights"] = {
                    "strengths": score_data.get("strengths", []),
                    "collaboration_fit": score_data.get("collaboration_fit", "fair"),
                    "growth_potential": score_data.get("growth_potential", "medium"),
                    "optimal_content_types": score_data.get("optimal_content_types", [])
                }
                
                scored_creators.append(creator_with_score)
        
        # Sort by match_score (highest first)
        scored_creators.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        semantic_matches = analysis_result.get("semantic_matches", [])
        
        return {
            "results": scored_creators,
            "query_processed": query,
            "semantic_matches": semantic_matches
        }
        
    except Exception as e:
        print(f"LLM call failed for AI Search: {str(e)}")
        
        # Enhanced fallback with multi-criteria matching
        query_words = set(query.lower().split())
        campaign_words = set((
            campaign_data['title'] + " " + 
            campaign_data.get('enhanced_brief', campaign_data['brief']) + " " +
            campaign_data['audience'] + " " +
            " ".join(campaign_data['platforms'])
        ).lower().split())
        
        scored_creators = []
        for creator in all_creators:
            creator_text = (
                creator['name'] + " " + 
                creator['description'] + " " + 
                creator['category'] + " " + 
                creator['platform'] + " " +
                creator['location']
            ).lower()
            
            creator_words = set(creator_text.split())
            
            # Calculate multiple match scores
            exact_matches = len(query_words.intersection(creator_words))
            campaign_matches = len(campaign_words.intersection(creator_words))
            platform_match = 1 if creator['platform'].lower() in [p.lower() for p in campaign_data['platforms']] else 0
            
            total_match_score = (exact_matches * 3) + (campaign_matches * 2) + (platform_match * 5)
            
            if total_match_score > 0:
                creator_with_score = creator.copy()
                creator_with_score["match_score"] = min(100, total_match_score * 5)
                scored_creators.append(creator_with_score)
        
        # Sort by match score
        scored_creators.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        fallback_semantic = list(set([
            *query.lower().split()[:2],
            *[c['category'] for c in scored_creators[:3]]
        ]))[:5]
        
        return {
            "results": scored_creators[:15],
            "query_processed": query,
            "semantic_matches": fallback_semantic
        }

# 3. OUTREACH ROUTES
@app.post("/api/outreach", response_model=SimpleOutreachResponse)
async def generate_outreach(request: SimpleOutreachRequest):
    """Generate AI-powered outreach email and voice message"""
    
    # Get campaign data
    campaign_data = await get_campaign_from_db(request.campaign_id)
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Get creator data
    creator_data = await get_creator_from_db(request.creator_id)
    if not creator_data:
        raise HTTPException(status_code=404, detail="Creator not found")
    
    # Generate content
    email_content, voice_script = await generate_simple_outreach_content(campaign_data, creator_data)
    
    # Generate voice message
    audio_url = await generate_simple_voice_message(voice_script, request.campaign_id, request.creator_id)
    
    # Store in database using your existing schema
    outreach_data = {
        "campaign_id": request.campaign_id,
        "creator_id": request.creator_id,
        "outreach_text": email_content,  # This maps to your 'outreach_text' column
        "audio_url": audio_url,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        result = supabase.table("outreach").insert(outreach_data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create outreach")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return SimpleOutreachResponse(
        email_content=email_content,
        audio_url=audio_url
    )

@app.get("/api/outreach/{campaign_id}/{creator_id}")
async def get_outreach(campaign_id: str, creator_id: str):
    """Get outreach details"""
    try:
        result = supabase.table("outreach").select("*").eq("campaign_id", campaign_id).eq("creator_id", creator_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Outreach not found")
        
        outreach_data = result.data[0]
        return {
            "id": outreach_data["id"],
            "campaign_id": outreach_data["campaign_id"],
            "creator_id": outreach_data["creator_id"],
            "email_content": outreach_data["outreach_text"],
            "audio_url": outreach_data["audio_url"],
            "created_at": outreach_data["created_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/outreach/campaign/{campaign_id}")
async def get_campaign_outreach(campaign_id: str):
    """Get all outreach for a campaign"""
    try:
        result = supabase.table("outreach").select("*").eq("campaign_id", campaign_id).execute()
        return {"outreach": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/api/outreach/batch")
async def generate_batch_outreach(campaign_id: str, creator_ids: List[str]):
    """Generate outreach for multiple creators"""
    
    campaign_data = await get_campaign_from_db(campaign_id)
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    results = []
    
    for creator_id in creator_ids:
        try:
            creator_data = await get_creator_from_db(creator_id)
            if not creator_data:
                results.append({
                    "creator_id": creator_id,
                    "status": "error",
                    "message": "Creator not found"
                })
                continue
            
            # Generate content
            email_content, voice_script = await generate_simple_outreach_content(campaign_data, creator_data)
            
            # Generate voice (you can skip this for batch to save API costs)
            audio_url = f"/api/audio/batch_outreach_{campaign_id}_{creator_id}.mp3"
            
            # Store data
            outreach_data = {
                "campaign_id": campaign_id,
                "creator_id": creator_id,
                "outreach_text": email_content,
                "audio_url": audio_url,
                "created_at": datetime.now().isoformat()
            }
            
            supabase.table("outreach").insert(outreach_data).execute()
            
            results.append({
                "creator_id": creator_id,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "creator_id": creator_id,
                "status": "error",
                "message": str(e)
            })
    
    return {
        "campaign_id": campaign_id,
        "total_creators": len(creator_ids),
        "results": results,
        "success_count": len([r for r in results if r["status"] == "success"])
    }


@app.post("/api/negotiations/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe audio using Whisper API"""
    transcription = "I think your rate is fair, but I typically charge a bit more for this type of content. Can we discuss the deliverables in more detail?"
    
    return {"transcription": transcription}

@app.post("/api/negotiations/respond")
async def generate_negotiation_response(message: NegotiationMessage):
    """Generate AI negotiation response"""
    # Simulate GPT-4 negotiation agent
    if message.sender == "creator":
        ai_response = f"I understand your position. We value quality creators and are willing to work within your rate expectations. Let's discuss what deliverables would work best for both parties. We're flexible on timeline and can offer additional exposure through our other channels."
    else:
        ai_response = f"That sounds reasonable. I can deliver high-quality content that aligns with your brand values. My standard package includes 1 main post, 3 stories, and a reel with 2 rounds of revisions. How does that sound?"
    
    # Simulate ElevenLabs voice generation
    audio_url = f"/api/audio/negotiation_{message.campaign_id}_{message.creator_id}_{datetime.now().timestamp()}.mp3"
    
    # Store negotiation message in database
    negotiation_data = {
        "campaign_id": message.campaign_id,
        "creator_id": message.creator_id,
        "message": message.message,
        "sender": message.sender,
        "ai_response": ai_response,
        "audio_url": audio_url,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        supabase.table("negotiations").insert(negotiation_data).execute()
    except Exception as e:
        print(f"Failed to store negotiation: {str(e)}")
    
    return {
        "response": ai_response,
        "audio_url": audio_url,
        "sender": "ai_agent"
    }

@app.get("/api/negotiations/{campaign_id}/{creator_id}")
async def get_negotiation_history(campaign_id: str, creator_id: str):
    """Get negotiation conversation history"""
    try:
        result = supabase.table("negotiations").select("*").eq("campaign_id", campaign_id).eq("creator_id", creator_id).execute()
        return {"messages": result.data}
    except Exception as e:
        return {"messages": []}

# 5. DEAL FINALIZATION ROUTES
@app.post("/api/deals")
async def create_deal(deal: DealRequest):
    """Finalize deal terms"""
    deal_id = str(uuid.uuid4())
    
    deal_data = {
        "id": deal_id,
        "campaign_id": deal.campaign_id,
        "creator_id": deal.creator_id,
        "rate": deal.rate,
        "deliverables": deal.deliverables,
        "platform": deal.platform,
        "timeline": deal.timeline,
        "status": "finalized",
        "created_at": datetime.now().isoformat()
    }
    
    result = await create_deal_in_db(deal_data)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create deal")
    
    return result

@app.get("/api/deals/{deal_id}")
async def get_deal(deal_id: str):
    """Get deal details"""
    deal_data = await get_deal_from_db(deal_id)
    if not deal_data:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    return deal_data

@app.get("/api/deals")
async def get_all_deals():
    """Get all deals"""
    try:
        result = supabase.table("deals").select("*").execute()
        return {"deals": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/deals/{deal_id}")
async def delete_deal(deal_id: str):
    """Delete deal"""
    deal_data = await get_deal_from_db(deal_id)
    if not deal_data:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    await delete_deal_from_db(deal_id)
    return {"message": "Deal deleted successfully"}


# 6. CONTRACT GENERATION ROUTES
@app.post("/api/contracts/generate")
async def generate_contract(request: ContractRequest):
    """Generate AI-powered contract PDF"""
    deal_data = await get_deal_from_db(request.deal_id)
    if not deal_data:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    campaign_data = await get_campaign_from_db(deal_data["campaign_id"])
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Simulate GPT-4 contract generation
    contract_content = f"""
    INFLUENCER MARKETING AGREEMENT

    Campaign: {campaign_data['title']}
    Platform: {deal_data['platform']}
    Rate: {deal_data['rate']}
    Deliverables: {deal_data['deliverables']}
    Timeline: {deal_data['timeline']}

    Terms and Conditions:
    1. Content creation and posting requirements
    2. Usage rights and licensing
    3. Payment terms and conditions
    4. Performance metrics and reporting
    5. Cancellation and modification clauses

    """
    create_contract_pdf(contract_content, request.deal_id)
    # Simulate PDF generation
    pdf_url = f"/api/contracts/download/{request.deal_id}.pdf"
    
    # Store contract in database
    contract_data = {
        "deal_id": request.deal_id,
        "contract_text": contract_content,
        "pdf_url": pdf_url,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        supabase.table("contracts").insert(contract_data).execute()
    except Exception as e:
        print(f"Failed to store contract: {str(e)}")
    
    return {
        "contract_text": contract_content,
        "pdf_url": pdf_url,
        "deal_id": request.deal_id
    }

@app.get("/api/contracts/download/{deal_id}.pdf")
async def download_contract(deal_id: str):
    """Serve actual contract PDF file"""
    file_path = f"static/contracts/{deal_id}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(
        path=file_path,
        filename=f"Contract_{deal_id}.pdf",
        media_type="application/pdf"
    )

# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "CreatorFlow AI Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)