from openenv.core.env_server import create_fastapi_app
from .models import SupportAction, SupportObservation
from .env import SupportEnv

# We pass the class SupportEnv. 
# The server will call SupportEnv() internally during /reset.
app = create_fastapi_app(
    SupportEnv, 
    SupportAction, 
    SupportObservation
)