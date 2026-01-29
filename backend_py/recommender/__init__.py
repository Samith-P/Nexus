"""
Recommender package for AI-powered journal recommendation system.

Phase 3 implementation:
- Converts Gemini output to semantic queries
- Performs embedding and similarity search
- Applies metric-aware scoring
- Generates deterministic explanations
"""

from .journal_recommender import JournalRecommender

__all__ = ["JournalRecommender"]
