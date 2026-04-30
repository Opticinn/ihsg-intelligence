"""
app/api/retrain.py
==================
API endpoints for monitoring and controlling adaptive retraining.

Endpoints:
  GET  /retrain/status   → current retrain state
  GET  /retrain/check    → evaluate retrain conditions now
  POST /retrain/trigger  → emergency retrain via GitHub Actions API
"""

import logging
import os

import httpx
from fastapi import APIRouter, HTTPException
from starlette.requests import Request

from app.core.limiter import limiter
from app.services.retrain_trigger import get_retrain_status, should_retrain

router = APIRouter(prefix="/retrain", tags=["Adaptive Retraining"])
logger = logging.getLogger(__name__)

GH_TOKEN = os.getenv("GITHUB_TOKEN", "")
GH_REPO  = os.getenv("GITHUB_REPO", "")  # format: "username/repo"


@router.get("/status", summary="Current retraining status")
async def get_status():
    """
    Return when the model was last retrained, accuracy at that point,
    and the configuration of all retrain triggers.
    """
    return get_retrain_status()


@router.get("/check", summary="Evaluate retrain conditions right now")
@limiter.limit("10/hour")
async def check_retrain(request: Request):
    """
    Run evaluation of all retrain conditions:
    - Schedule (Monday/Thursday)
    - IHSG move > 2.5%
    - Volatility > 90th percentile
    - Accuracy drift > 10%

    Does NOT trigger retraining — evaluation only.
    """
    result = should_retrain()
    return result


@router.post("/trigger", summary="Trigger emergency retrain via GitHub Actions")
@limiter.limit("5/hour")
async def trigger_emergency_retrain(request: Request, reason: str = "manual_api"):
    """
    Trigger an emergency retrain by sending a repository_dispatch
    event to GitHub Actions.

    Required env vars:
      GITHUB_TOKEN — personal access token with repo scope
      GITHUB_REPO  — format: "username/repo-name"
    """
    if not GH_TOKEN or not GH_REPO:
        raise HTTPException(
            status_code=503,
            detail=(
                "GITHUB_TOKEN or GITHUB_REPO not configured in .env. "
                "Set both environment variables to enable remote trigger."
            ),
        )

    url     = f"https://api.github.com/repos/{GH_REPO}/dispatches"
    headers = {
        "Authorization":        f"Bearer {GH_TOKEN}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "event_type":     "emergency_retrain",
        "client_payload": {
            "trigger": "manual_api",
            "reason":  reason,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, headers=headers, json=payload)

        if resp.status_code == 204:
            logger.info(f"[retrain] Emergency retrain triggered via GitHub API (reason: {reason})")
            return {
                "status":  "triggered",
                "message": "Emergency retrain successfully triggered. Check GitHub Actions tab for progress.",
                "github_actions_url": f"https://github.com/{GH_REPO}/actions",
                "reason":  reason,
            }
        else:
            raise HTTPException(
                status_code=502,
                detail=f"GitHub API error: {resp.status_code} — {resp.text}",
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout connecting to GitHub API.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))