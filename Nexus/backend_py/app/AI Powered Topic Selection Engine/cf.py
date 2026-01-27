from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def cf_scores(user_id: Optional[str], candidate_topic_ids: List[str]) -> Tuple[Optional[Dict[str, float]], bool]:
    """CF is disabled when using only the Datasets/ folder.

    User requested not to use JSON files under data/. The previous CF PoC relied on
    data/user_topic_interactions.json, so we always return cold-start.
    """

    _ = (user_id, candidate_topic_ids)
    return None, True
