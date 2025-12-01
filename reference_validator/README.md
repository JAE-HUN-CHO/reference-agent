# 논문 레퍼런스 검증 Multi-Agent 시스템

DeepAgents 아키텍처를 기반으로 설계된 논문 레퍼런스 검증 시스템입니다.

## 주요 기능

- **PDF 파싱**: 논문에서 레퍼런스 및 인용 문맥 자동 추출
- **존재 검증**: 웹 검색을 통한 레퍼런스 실제 존재 여부 확인
- **인용 적절성 분석**: 인용이 문맥에 적절하게 사용되었는지 평가
- **종합 리포트**: 상세한 검증 결과 및 권장 사항 제공

## 지원 LLM

### 로컬 (Ollama)
- **glm-4.6:cloud** (기본 모델)
- llama3, llama3.1, llama3.2
- qwen2.5
- gemma2
- mistral

### 클라우드
- OpenAI: gpt-4, gpt-4o, gpt-4o-mini
- Anthropic: claude-3-sonnet, claude-3-haiku
- Google: gemini-2.5-pro, gemini-2.5-flash

## 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# Ollama 설치 (로컬 LLM 사용 시)
# https://ollama.ai 에서 설치 후:
ollama pull llama3.2
```

## 사용법

### 커맨드라인

```bash
# 기본 실행 (Ollama glm-4.6:cloud)
python main.py paper.pdf

# Ollama 다른 모델
python main.py paper.pdf --provider ollama --model llama3.2
python main.py paper.pdf --provider ollama --model qwen2.5

# OpenAI
export OPENAI_API_KEY=your_key
python main.py paper.pdf --provider openai --model gpt-4o

# Google Gemini
export GOOGLE_API_KEY=your_key
python main.py paper.pdf --provider google --model gemini-2.5-pro

# Tavily 검색 활성화
export TAVILY_API_KEY=your_key
python main.py paper.pdf

# 최대 레퍼런스 수 제한
python main.py paper.pdf --max-refs 10

# 출력 파일 지정
python main.py paper.pdf --output results.json
```

### Python 코드

```python
from reference_validator import validate_paper, configure

# LLM 설정 (기본: glm-4.6:cloud)
configure(provider="ollama", model_name="glm-4.6:cloud")

# 검증 실행
report = validate_paper("paper.pdf")

# 결과 확인
print(f"유효한 레퍼런스: {report.valid_count}/{report.total_references}")
print(f"인용 적절성 평균: {report.average_appropriateness_score:.1f}점")
print(f"\n요약:\n{report.summary}")
```

### 에이전트별 다른 모델 사용

```python
from reference_validator import configure_multi_model, ModelPresets

# Parser는 빠른 모델, Validation은 강력한 모델 사용
configure_multi_model(
    parser_config=ModelPresets.OLLAMA_GLM4_CLOUD,
    validation_config=ModelPresets.OPENAI_GPT4O,
    context_config=ModelPresets.GOOGLE_GEMINI_PRO,
)
```

## 아키텍처

```
                     ┌─────────────────────────────────────────┐
                     │           SUPERVISOR AGENT              │
                     │     (LangGraph 워크플로우 조율)          │
                     └─────────────────────────────────────────┘
                                         │
         ┌───────────────┬───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  PARSER   │   │    WEB    │   │VALIDATION │   │  CONTEXT  │   │ CITATION  │
   │  AGENT    │   │   AGENT   │   │  AGENT    │   │   AGENT   │   │   AGENT   │
   └───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘
   PDF 파싱 및     웹 검색         존재 여부       인용 문맥       인용 적절성
   레퍼런스 추출   (Tavily,        검증           분석            최종 판단
                  CrossRef 등)
```

## 워크플로우

1. **Parser Agent**: PDF에서 레퍼런스 및 인용 문맥 추출
2. **Web Agent**: 각 레퍼런스에 대해 웹 검색 수행
3. **Validation Agent**: 검색 결과와 레퍼런스 정보 비교 검증
4. **Context Agent**: 인용 문맥 분석 (관련성, 일치성)
5. **Citation Agent**: 인용 적절성 최종 판단
6. **Report Generator**: 종합 리포트 생성

## 출력 예시

```
================================================================================
검증 결과 요약
================================================================================

📄 논문: Attention Is All You Need
📊 총 레퍼런스: 42개

🔍 존재 검증 결과:
   ✅ 유효: 38개
   ❌ 유효하지 않음: 2개
   ❓ 검증 불가: 2개

📝 인용 적절성:
   ✅ 적절: 36개
   ❌ 부적절: 6개
   📈 평균 점수: 82.5/100

⚠️ 문제가 있는 레퍼런스:
   - [15]: INVALID - 제목 불일치
   - [23]: INVALID - 존재하지 않음

💡 권장 사항:
   • 레퍼런스 [15], [23]의 존재 여부를 재확인하세요
   • 6개의 인용에서 부적절한 사용이 발견되었습니다
================================================================================
```

## 환경 변수

| 변수명 | 설명 | 필수 |
|--------|------|------|
| `TAVILY_API_KEY` | Tavily 웹 검색 API 키 | 권장 |
| `OPENAI_API_KEY` | OpenAI API 키 | OpenAI 사용 시 |
| `ANTHROPIC_API_KEY` | Anthropic API 키 | Claude 사용 시 |
| `GOOGLE_API_KEY` | Google AI API 키 | Gemini 사용 시 |

## 프로젝트 구조

```
reference_validator/
├── __init__.py          # 패키지 초기화
├── main.py              # 메인 실행 파일
├── config.py            # LLM 및 전역 설정
├── schema.py            # 데이터 스키마
├── supervisor.py        # LangGraph 워크플로우
├── requirements.txt     # 의존성
├── README.md            # 문서
│
├── agents/              # 에이전트 모듈
│   ├── __init__.py
│   ├── parser_agent.py    # PDF 파싱
│   ├── web_agent.py       # 웹 검색
│   ├── validation_agent.py # 존재 검증
│   ├── context_agent.py   # 문맥 분석
│   └── citation_agent.py  # 인용 적절성
│
├── tools/               # 도구 모듈
│   ├── __init__.py
│   └── search_tools.py    # 검색 도구 (Tavily, CrossRef 등)
│
└── workspace/           # 작업 디렉토리
    ├── input/           # 입력 PDF
    ├── parsed/          # 파싱 결과
    ├── verified_papers/ # 검증된 논문 정보
    ├── context/         # 문맥 분석 결과
    ├── validation_results/
    └── output/          # 최종 결과
```

## 라이선스

MIT License
