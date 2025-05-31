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

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = FastAPI(title="CreatorFlow AI Backend", version="1.0.0")


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
    sender: str  # 'brand' or 'creator'

class DealRequest(BaseModel):
    campaign_id: str
    creator_id: str
    final_rate: str
    deliverables: str
    platform: str
    timeline: str

class ContractRequest(BaseModel):
    deal_id: str

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

    Please rewrite it as an engaging influencer brief.
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
    platform: Optional[str] = None,
    min_followers: Optional[int] = None
):
    """Get list of creators with optional filters"""
    creators_data = await get_creators_from_db(category, platform)
    
    # If no creators in database, return mock data
    if not creators_data:
        mock_creators = [
            {
                "id": str(uuid.uuid4()),
                "name": "Emma Rodriguez",
                "handle": "@emmasstyle",
                "platform": "Instagram",
                "followers": "125K",
                "engagement": "4.2%",
                "category": "Fashion",
                "location": "Los Angeles, CA",
                "description": "Fashion & lifestyle creator with authentic voice and high engagement"
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Marcus Chen",
                "handle": "@techbymarcus",
                "platform": "YouTube",
                "followers": "89K",
                "engagement": "6.1%",
                "category": "Technology",
                "location": "San Francisco, CA",
                "description": "Tech reviews and tutorials with engaged community"
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Sofia Martinez",
                "handle": "@sofiafitness",
                "platform": "TikTok",
                "followers": "245K",
                "engagement": "8.7%",
                "category": "Fitness",
                "location": "Miami, FL",
                "description": "Fitness motivation and workout routines"
            }
        ]
        
        # Insert mock creators into database
        for creator_data in mock_creators:
            await create_creator_in_db(creator_data)
        
        creators_data = mock_creators
    
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
async def ai_search_creators(query: str, campaign_id: str):
    """AI-powered semantic search for creators"""
    # Simulate embedding search
    await asyncio.sleep(1)  # Simulate processing time
    
    # Return filtered results based on query
    creators = await get_creators()
    return {
        "results": creators,
        "query_processed": query,
        "semantic_matches": ["fashion", "lifestyle", "authentic"]
    }

# 3. AI OUTREACH ROUTES
@app.post("/api/outreach", response_model=OutreachResponse)
async def generate_outreach(request: OutreachRequest):
    """Generate AI-powered outreach email and voice message"""
    campaign_data = await get_campaign_from_db(request.campaign_id)
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Handle multiple platforms in outreach
    platform_text = ", ".join(campaign_data["platforms"]) if len(campaign_data["platforms"]) > 1 else campaign_data["platforms"][0]
    
    # Simulate GPT-4 outreach generation
    email_content = f"""Subject: Exciting Collaboration Opportunity - {campaign_data['title']}

Hi there!

I hope this message finds you well. I've been following your content and I'm impressed by your authentic voice and engagement with your {campaign_data['audience']} audience.

We're launching an exciting campaign for {campaign_data['title']} across {platform_text}, and I believe your creative style would be a perfect fit for our brand.

Campaign Brief: {campaign_data['brief']}

Budget: {campaign_data['budget']} INR

Would you be interested in discussing a collaboration? We're offering competitive rates and flexible creative freedom.

Looking forward to hearing from you!

Best regards,
CreatorFlow AI Team"""
    
    # Simulate ElevenLabs voice generation
    audio_url = f"/api/audio/outreach_{request.campaign_id}_{request.creator_id}.mp3"
    
    outreach_data = {
        "campaign_id": request.campaign_id,
        "creator_id": request.creator_id,
        "email_content": email_content,
        "audio_url": audio_url,
        "created_at": datetime.now().isoformat()
    }
    
    await create_outreach_in_db(outreach_data)
    
    return OutreachResponse(
        email_content=email_content,
        audio_url=audio_url
    )

@app.get("/api/outreach/{campaign_id}/{creator_id}")
async def get_outreach(campaign_id: str, creator_id: str):
    """Get outreach details"""
    outreach_data = await get_outreach_from_db(campaign_id, creator_id)
    if not outreach_data:
        raise HTTPException(status_code=404, detail="Outreach not found")
    
    return outreach_data

# 4. VOICE NEGOTIATION ROUTES
@app.post("/api/negotiations/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe audio using Whisper API"""
    # Simulate Whisper transcription
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
        "final_rate": deal.final_rate,
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
Rate: {deal_data['final_rate']}
Deliverables: {deal_data['deliverables']}
Timeline: {deal_data['timeline']}

Terms and Conditions:
1. Content creation and posting requirements
2. Usage rights and licensing
3. Payment terms and conditions
4. Performance metrics and reporting
5. Cancellation and modification clauses

Generated by CreatorFlow AI
"""
    
    # Simulate PDF generation
    pdf_url = f"/api/contracts/download/{request.deal_id}.pdf"
    
    # Store contract in database
    contract_data = {
        "deal_id": request.deal_id,
        "contract_content": contract_content,
        "pdf_url": pdf_url,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        supabase.table("contracts").insert(contract_data).execute()
    except Exception as e:
        print(f"Failed to store contract: {str(e)}")
    
    return {
        "contract_content": contract_content,
        "pdf_url": pdf_url,
        "deal_id": request.deal_id
    }

@app.get("/api/contracts/download/{deal_id}.pdf")
async def download_contract(deal_id: str):
    """Download contract PDF"""
    # In real implementation, return actual PDF file
    return {"message": f"Contract PDF for deal {deal_id} would be downloaded here"}

# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "CreatorFlow AI Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)