"""
MedSpatial AI — Chat / Q&A API
Conversational endpoint for medical Q&A tied to specific scans.
"""

import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Analysis, ChatSession, Scan, Volume
from app.models.database import get_db
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["Chat"])
chat_svc = ChatService()


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Send a question about a scan and receive an AI-generated answer."""
    # Validate scan exists
    result = await db.execute(select(Scan).where(Scan.id == req.scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    # Get or create chat session
    session = None
    if req.session_id:
        result = await db.execute(select(ChatSession).where(ChatSession.id == req.session_id))
        session = result.scalar_one_or_none()
    if not session:
        session = ChatSession(
            id=str(uuid.uuid4()),
            scan_id=req.scan_id,
            messages=[],
        )
        db.add(session)
        await db.flush()

    # Gather context: volume data, analysis results
    volume_ctx = None
    result = await db.execute(select(Volume).where(Volume.scan_id == req.scan_id))
    volume = result.scalar_one_or_none()
    if volume:
        volume_ctx = {
            "dimensions": volume.dimensions,
            "voxel_spacing": volume.voxel_spacing,
            "hu_min": volume.hu_min,
            "hu_max": volume.hu_max,
            "volume_path": volume.volume_path,
        }

    analysis_ctx = []
    result = await db.execute(
        select(Analysis).where(Analysis.scan_id == req.scan_id, Analysis.status == "completed")
    )
    analyses = result.scalars().all()
    for a in analyses:
        analysis_ctx.append({
            "type": a.analysis_type,
            "findings": a.findings,
            "summary": a.summary,
            "confidence": a.confidence,
        })

    scan_ctx = {
        "modality": scan.modality,
        "body_part": scan.body_part,
        "study_description": scan.study_description,
        "num_slices": scan.num_slices,
        "metadata": scan.metadata_json,
    }

    # Add user message to history
    messages = session.messages or []
    messages.append({
        "role": "user",
        "content": req.message,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })

    # Generate AI response
    ai_response = chat_svc.answer_question(
        question=req.message,
        chat_history=messages,
        scan_context=scan_ctx,
        volume_context=volume_ctx,
        analysis_context=analysis_ctx,
    )

    # Append assistant response to history
    messages.append({
        "role": "assistant",
        "content": ai_response["answer"],
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })

    session.messages = messages
    session.context_summary = ai_response.get("context_summary", "")
    await db.flush()

    return ChatResponse(
        session_id=session.id,
        response=ai_response["answer"],
        referenced_slices=ai_response.get("referenced_slices"),
        referenced_regions=ai_response.get("referenced_regions"),
        findings_mentioned=ai_response.get("findings_mentioned"),
    )


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get full conversation history for a chat session."""
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    return {
        "session_id": session.id,
        "scan_id": session.scan_id,
        "messages": session.messages,
        "created_at": session.created_at,
    }


@router.get("/sessions/{scan_id}")
async def list_chat_sessions(scan_id: str, db: AsyncSession = Depends(get_db)):
    """List all chat sessions for a scan."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.scan_id == scan_id).order_by(ChatSession.created_at.desc())
    )
    sessions = result.scalars().all()
    return {
        "scan_id": scan_id,
        "sessions": [
            {
                "session_id": s.id,
                "message_count": len(s.messages) if s.messages else 0,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ],
    }
