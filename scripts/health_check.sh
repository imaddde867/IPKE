#!/bin/bash
set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
success(){ echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
fail(){ echo -e "${RED}[FAIL]${NC} $1"; }

info "Explainium Health Check"

# 1. Python env
if python -c 'import sys; assert sys.version_info>=(3,12)' 2>/dev/null; then success "Python version >=3.12"; else fail "Python 3.12+ required"; exit 1; fi

# 2. Core imports
python - <<'PY'
mods = [
    'fastapi','streamlit','pandas','numpy','spacy','sentence_transformers','llama_cpp'
]
import importlib,sys
missing=[]
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: missing.append(f"{m}: {e}")
if missing:
    print("MISSING:")
    for m in missing: print(m)
    sys.exit(1)
print("ALL CORE IMPORTS OK")
PY
[ $? -eq 0 ] && success "Core Python deps import" || { fail "Core imports failed"; exit 1; }

# 3. spaCy model
python - <<'PY'
import spacy, sys
try:
    spacy.load('en_core_web_sm')
    print('SPACY MODEL OK')
except Exception as e:
    print('SPACY MODEL MISSING:', e); sys.exit(1)
PY
[ $? -eq 0 ] && success "spaCy model present" || warn "spaCy model missing (run: python -m spacy download en_core_web_sm)"

# 4. LLM minimal inference (skip if llama-cpp not installed)
python - <<'PY'
try:
    from llama_cpp import Llama
    import os, pathlib
    model_dir = pathlib.Path('models/llm/Mistral-7B-Instruct-v0.2-GGUF')
    ggufs = list(model_dir.glob('*.gguf')) if model_dir.exists() else []
    if ggufs:
        llm = Llama(model_path=str(ggufs[0]), n_ctx=256, n_batch=16, n_threads=4, n_gpu_layers=0, verbose=False)
        out = llm("Extract: Test system init ok", max_tokens=8)
        print('LLM OK:', out['choices'][0]['text'].strip())
    else:
        print('LLM SKIP: no gguf file present')
except ImportError:
    print('LLM SKIP: llama_cpp not installed')
except Exception as e:
    print('LLM ERROR:', e)
PY

# 5. Processor smoke test
python - <<'PY'
import asyncio, sys
try:
    from src.ai.llm_processing_engine import LLMProcessingEngine
    async def main():
        eng = LLMProcessingEngine()
        await eng.initialize()
        res = await eng.process_document("Simple procedure: tighten valve to 50 PSI", "text", {})
        print('Entities:', len(res.entities))
    asyncio.run(main())
    print('PROCESSOR OK')
except Exception as e:
    print('PROCESSOR ERROR:', e); sys.exit(1)
PY
[ $? -eq 0 ] && success "Processor smoke test" || warn "Processor smoke test failed"

success "Health check complete"
