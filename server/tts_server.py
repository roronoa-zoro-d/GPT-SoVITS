from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi import APIRouter
import uvicorn
import argparse
import logging


import os
import sys


from gpt_sovits_server import router as gpt_sovits_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


# 注册子路由
app.include_router(gpt_sovits_router, prefix="/gpt_sovits", tags=["tts1"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)