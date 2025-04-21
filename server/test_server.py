from fastapi import FastAPI

app = FastAPI()

@app.post("/greet/")
async def greet(data: dict):
    name = data['name']
    return {"message": f"Hello {name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)