# Start_openclip_server.py
import uvicorn

if __name__ == '__main__':
    uvicorn.run("test_fastapi:app", host="0.0.0.0", port=8001)