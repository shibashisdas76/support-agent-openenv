import uvicorn
from openenv.core.env_server import create_fastapi_app
from .models import SupportAction, SupportObservation
from .env import SupportEnv

# Create the FastAPI app instance
app = create_fastapi_app(
    SupportEnv, 
    SupportAction, 
    SupportObservation
)

# Required by OpenEnv validator
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()