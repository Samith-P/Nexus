"""
FastAPI routes for topic selection engine.
Note: Routes are now defined directly in app.py for simplicity.
This file is kept for reference/backward compatibility.
"""

try:
	from .engine import generate_topics
except ImportError:
	from engine import generate_topics

# Example usage in FastAPI:
# 
# from fastapi import FastAPI
# app = FastAPI()
# 
# @app.post("/topic-recommendation")
# def topic_recommendation(payload: dict):
#     return generate_topics(payload)

