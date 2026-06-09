"""AWM-JGEM API contracts layered over the existing scan workflow."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.graph.graph_builder import build_scene_graph
from app.awm.schema import AnatomicalWorldModel
from app.awm.store import get_awm_store
from app.models import Scan
from app.models.database import get_db
from app.policy.audit_logger import AuditLogger
from app.policy.executor_gateway import ExecutorGateway
from app.policy.intent_schema import Intent
from app.reasoning.grounded_answer import GroundedAnswer, answer_question
from app.reasoning.intent_parser import parse_question

router = APIRouter(prefix="/api", tags=["AWM-JGEM"])


class AskRequest(BaseModel):
    question: str
    actor_id: str = "local-user"
    actor_role: str = "clinician"
    selected_finding_id: str | None = None


@router.get("/studies/{study_id}/awm", response_model=AnatomicalWorldModel)
async def get_awm(study_id: str):
    awm = await get_awm_store().get(study_id)
    if awm is None:
        raise HTTPException(status_code=404, detail="Anatomical world model not found")
    return awm


@router.get("/studies/{study_id}/scene-graph")
async def get_scene_graph(study_id: str):
    awm = await get_awm_store().get(study_id)
    if awm is None:
        raise HTTPException(status_code=404, detail="Anatomical world model not found")
    graph = build_scene_graph(awm)
    if graph != awm.scene_graph:
        awm.scene_graph = graph
        await get_awm_store().save(awm)
    return graph


@router.post("/studies/{study_id}/ask", response_model=GroundedAnswer)
async def ask(study_id: str, request: AskRequest):
    awm = await get_awm_store().get(study_id)
    if awm is None:
        raise HTTPException(status_code=404, detail="Anatomical world model not found")
    intent = parse_question(
        question=request.question,
        actor_id=request.actor_id,
        actor_role=request.actor_role,
        study_id=study_id,
        deidentified_study_id=awm.study.deidentified_study_id,
        modality=awm.study.modality.value,
    )
    decision = await ExecutorGateway().evaluate(intent)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.model_dump())
    return answer_question(awm, request.question)


@router.post("/intents/evaluate")
async def evaluate_intent(intent: Intent):
    return await ExecutorGateway().evaluate(intent)


@router.get("/audit/{study_id}")
async def get_audit(study_id: str):
    return {"study_id": study_id, "events": await AuditLogger().read(study_id)}


@router.get("/models/registry")
async def get_model_registry():
    return {
        "architecture": "AWM-JGEM",
        "models": [
            {"name": "jepa-light", "version": "1.0.0", "provider": "local", "calibrated": False},
            {"name": "anatomical-gnn", "version": "1.0.0", "provider": "local", "calibrated": False},
            {"name": "energy-scorer", "version": "1.0.0", "provider": "local", "calibrated": False},
        ],
        "warning": "Bundled fallback models are architectural baselines and are not clinically validated.",
    }


@router.get("/studies")
async def list_studies(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Scan).order_by(Scan.created_at.desc()))
    return {"studies": [
        {
            "study_id": scan.id,
            "modality": scan.modality,
            "body_region": scan.body_region or scan.body_part,
            "status": scan.status.value,
        }
        for scan in result.scalars().all()
    ]}
