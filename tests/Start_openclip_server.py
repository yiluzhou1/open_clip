# Start_openclip_server.py
import uvicorn

if __name__ == '__main__':
    uvicorn.run("test_fastapi:app", host="0.0.0.0", port=8001)

"""
Compiling to EXE:
pyinstaller --noconfirm Start_openclip_server.spec

a = Analysis(
    ['Start_openclip_server.py'],
    pathex=[],
    binaries=[],
    datas=[('test_fastapi.py', '.'),('static', 'static')],
    hiddenimports=[
        'test_fastapi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)



"""