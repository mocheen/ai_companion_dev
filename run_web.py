"""
Web服务启动脚本
启动FastAPI Web服务
"""

import uvicorn
from api import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5163,
        log_level="info"
    )
