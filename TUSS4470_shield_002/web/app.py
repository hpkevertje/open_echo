from contextlib import asynccontextmanager
from depth_output import OutputManager
from recorder import Recorder
from settings import Settings
from echo import EchoReader, SerialReader
import logging
from fastapi import FastAPI, WebSocket, Request, Form, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("uvicorn")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"WebSocket connected: {websocket.client}")

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data):
        for connection in self.active_connections:
            await connection.send_json(data)


connection_manager = ConnectionManager()
output_manager = OutputManager()
recorder = Recorder()


async def on_data(data):
    data["recording"] = recorder.running
    recorder.record(data)
    await connection_manager.broadcast_json(data)


echo_reader = EchoReader(
    data_callback=on_data,
    depth_callback=output_manager.update,
    sediment_callback=output_manager.update_sediment,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await update_settings(Settings.load())
    except Exception as e:
        log.error(f"Failed to load settings: {e}")

    with output_manager, echo_reader:
        try:
            yield
        finally:
            await recorder.stop()


app = FastAPI(lifespan=lifespan)
app.state.settings = Settings()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

async def update_settings(new_settings: Settings):
    previous = app.state.settings
    settings = Settings.model_validate(
        {
            **previous.model_dump(exclude_none=True, exclude_unset=True, exclude_defaults=True),
            **new_settings.model_dump(exclude_none=True, exclude_unset=True, exclude_defaults=True),
        }
    )

    # Checkbox fields and the recorder form fields are defaults when absent;
    # apply them explicitly so unchecking "enabled" actually disables them.
    for field in (
        "recording_enabled",
        "recording_hz",
        "recording_dir",
        "recording_max_mb",
        "recording_duration_hours",
        "recording_segment_minutes",
    ):
        setattr(settings, field, getattr(new_settings, field))

    echo_reader.update_settings(settings)
    app.state.settings = settings

    try:
        await recorder.update_settings(settings, previous)
    except Exception as e:
        log.error(f"Failed to update recorder: {e}")

    settings.save()

    try:
        await output_manager.update_settings(settings)
    except Exception as e:
        # Keep the saved configuration even if an output cannot connect.
        log.error(f"Failed to update outputs: {e}")



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Just here to keep the connection alive
    except Exception as e:
        log.error(f"WebSocket closed: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@app.get("/")
async def home(request: Request):
    if app.state.settings.serial_port == "init":
        return RedirectResponse("/config", status_code=303)

    return templates.TemplateResponse(
        "frontend.html", {"request": request, "settings": app.state.settings}
    )


@app.get("/config")
async def config(request: Request):
    return templates.TemplateResponse(
        "config.html",
        {
            "request": request,
            "settings": app.state.settings,
            "ports": SerialReader.get_serial_ports(),
            "recorder_status": recorder.status(),
        },
    )


@app.get("/recordings")
async def recordings(request: Request):
    return templates.TemplateResponse(
        "recordings.html",
        {
            "request": request,
            "files": recorder.list_files(),
            "status": recorder.status(),
        },
    )


@app.get("/recordings/{name}")
async def download_recording(name: str):
    path = recorder.path_for(name)
    if path is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    media_type = "application/gzip" if name.endswith(".gz") else "application/x-ndjson"
    return FileResponse(
        path,
        media_type=media_type,
        filename=name,
    )


@app.post("/config")
async def config_post(request: Request, new_settings: Settings = Form(...)):
    await update_settings(new_settings)
    return RedirectResponse("/", status_code=303)
