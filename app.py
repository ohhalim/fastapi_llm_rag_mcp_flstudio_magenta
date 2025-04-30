from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import subprocess
import os
import logging
import uvicorn
from typing import Optional, List, Dict
import uuid
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="BitNet API", description="다른 환경의 BitNet 모델을 호출하는 API 서비스")

# BitNet 설정 (실제 경로로 수정 필요)
BITNET_ROOT = "/Users/ohhalim/git_box/BitNet"  # BitNet 저장소 루트 디렉토리
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"  # 모델 파일 경로 (BitNet 저장소 내 상대 경로)
CONDA_ENV = "bitnet-cpp"  # BitNet이 설치된 conda 환경 이름

# 캐시 - 요청 처리 결과 저장
result_cache: Dict[str, Dict] = {}

# 요청 모델 정의
class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    threads: int = 4
    context_size: int = 2048
    conversation: bool = True
    system_prompt: Optional[str] = None

# 응답 모델 정의
class QueryResponse(BaseModel):
    id: str
    status: str
    response: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 BitNet 환경 확인"""
    # 경로 확인
    full_model_path = os.path.join(BITNET_ROOT, MODEL_PATH)
    if not os.path.exists(full_model_path):
        logger.warning(f"모델 파일을 찾을 수 없습니다: {full_model_path}")
        logger.info("MODEL_PATH 변수를 실제 모델 파일 경로로 수정하세요.")
    else:
        logger.info(f"모델 파일 확인 완료: {full_model_path}")
    
    # run_inference.py 스크립트 확인
    inference_script = os.path.join(BITNET_ROOT, "run_inference.py")
    if not os.path.exists(inference_script):
        logger.warning(f"추론 스크립트를 찾을 수 없습니다: {inference_script}")
    else:
        logger.info(f"추론 스크립트 확인 완료: {inference_script}")

def run_bitnet_inference(task_id: str, request: QueryRequest):
    """BitNet 모델을 사용하여 추론 실행 (백그라운드 작업)"""
    try:
        full_model_path = os.path.join(BITNET_ROOT, MODEL_PATH)
        
        # conda 환경에서 추론 스크립트 실행을 위한 명령어 구성
        if os.name == 'nt':  # Windows
            conda_cmd = f"conda activate {CONDA_ENV} && "
        else:  # macOS/Linux
            conda_cmd = f"source activate {CONDA_ENV} && "
        
        # 추론 명령어 구성
        inference_cmd = f"python {BITNET_ROOT}/run_inference.py -m {full_model_path} -n {request.max_tokens} -t {request.threads} -c {request.context_size} -temp {request.temperature}"
        
        # 대화 모드 활성화 여부에 따라 추가 옵션 설정
        if request.conversation:
            inference_cmd += " -cnv"
            # 사용자가 별도의 시스템 프롬프트를 제공했는지 확인
            prompt = request.system_prompt if request.system_prompt else "You are a helpful assistant."
            inference_cmd += f" -p \"{prompt}\""
            
            # 사용자 메시지 준비
            user_input = request.prompt
        else:
            # 대화 모드가 아닌 경우 프롬프트를 그대로 사용
            inference_cmd += f" -p \"{request.prompt}\""
            user_input = None
        
        # 전체 명령어 구성 (conda 활성화 + 추론 실행)
        full_cmd = conda_cmd + inference_cmd
        
        logger.info(f"실행 명령어: {full_cmd}")
        start_time = time.time()
        
        # 셸을 통해 명령어 실행
        process = subprocess.Popen(
            full_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True
        )
        
        # 대화 모드인 경우 사용자 입력 전달
        stdout, stderr = "", ""
        if request.conversation and user_input:
            stdout, stderr = process.communicate(input=user_input)
        else:
            stdout, stderr = process.communicate()
        
        end_time = time.time()
        
        # 프로세스 오류 확인
        if process.returncode != 0:
            logger.error(f"BitNet 실행 오류: {stderr}")
            result_cache[task_id] = {
                "status": "error",
                "error": f"BitNet 실행 오류: {stderr}",
                "created_at": start_time,
                "completed_at": end_time
            }
            return
        
        # 응답 처리 및 캐싱
        response_text = stdout.strip()
        result_cache[task_id] = {
            "status": "completed",
            "response": response_text,
            "created_at": start_time,
            "completed_at": end_time
        }
        
        logger.info(f"BitNet 실행 완료: Task ID {task_id}")
        
    except Exception as e:
        logger.error(f"추론 과정에서 오류 발생: {str(e)}")
        result_cache[task_id] = {
            "status": "error",
            "error": str(e),
            "created_at": time.time(),
            "completed_at": time.time()
        }

@app.post("/generate", response_model=QueryResponse)
async def generate_response(request: QueryRequest, background_tasks: BackgroundTasks):
    """비동기적으로 응답을 생성합니다."""
    task_id = str(uuid.uuid4())
    created_at = time.time()
    
    # 초기 상태 캐싱
    result_cache[task_id] = {
        "status": "processing",
        "created_at": created_at,
        "completed_at": None
    }
    
    # 백그라운드에서 추론 실행
    background_tasks.add_task(run_bitnet_inference, task_id, request)
    
    return QueryResponse(
        id=task_id,
        status="processing",
        created_at=created_at,
        response=None
    )

@app.get("/generate/{task_id}", response_model=QueryResponse)
async def get_generation_result(task_id: str):
    """작업 ID로 결과를 조회합니다."""
    if task_id not in result_cache:
        raise HTTPException(status_code=404, detail=f"작업 ID를 찾을 수 없습니다: {task_id}")
    
    result = result_cache[task_id]
    
    return QueryResponse(
        id=task_id,
        status=result.get("status", "unknown"),
        response=result.get("response"),
        created_at=result.get("created_at", 0),
        completed_at=result.get("completed_at"),
        error=result.get("error")
    )

@app.post("/generate_sync", response_model=QueryResponse)
async def generate_response_sync(request: QueryRequest):
    """동기적으로 응답을 생성합니다 (응답이 반환될 때까지 대기)."""
    task_id = str(uuid.uuid4())
    created_at = time.time()
    
    # 동기적으로 추론 실행
    run_bitnet_inference(task_id, request)
    
    # 결과 확인
    if task_id not in result_cache:
        raise HTTPException(status_code=500, detail="추론 중 오류가 발생했습니다.")
    
    result = result_cache[task_id]
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "알 수 없는 오류가 발생했습니다."))
    
    return QueryResponse(
        id=task_id,
        status=result.get("status", "completed"),
        response=result.get("response"),
        created_at=created_at,
        completed_at=result.get("completed_at"),
        error=None
    )

@app.get("/health")
async def health_check():
    """서비스 상태를 확인합니다."""
    full_model_path = os.path.join(BITNET_ROOT, MODEL_PATH)
    inference_script = os.path.join(BITNET_ROOT, "run_inference.py")
    
    return {
        "status": "online",
        "conda_env": CONDA_ENV,
        "bitnet_root": {
            "path": BITNET_ROOT,
            "exists": os.path.exists(BITNET_ROOT)
        },
        "model": {
            "path": full_model_path,
            "exists": os.path.exists(full_model_path)
        },
        "inference_script": {
            "path": inference_script,
            "exists": os.path.exists(inference_script)
        }
    }

# 직접 실행 시 uvicorn 서버 시작
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)