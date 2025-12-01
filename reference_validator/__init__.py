"""
논문 레퍼런스 검증 Multi-Agent 시스템

DeepAgents 아키텍처를 기반으로 설계된 논문 레퍼런스 검증 시스템입니다.

특징:
- Ollama 및 Cloud LLM 모델 지원 (llama3, qwen, gpt-4, claude, gemini 등)
- Tavily 및 Crawl 기반 웹 검색
- Semantic Scholar, CrossRef API 지원
- LangGraph 기반 워크플로우

사용법:
    from reference_validator import validate_paper, configure
    
    # Ollama로 설정
    configure(provider="ollama", model_name="llama3.2")
    
    # 또는 OpenAI로 설정
    configure(provider="openai", model_name="gpt-4o", api_key="...")
    
    # 검증 실행
    report = validate_paper("paper.pdf")
    print(report.summary)
"""

from .config import (
    configure,
    configure_multi_model,
    settings,
    LLMConfig,
    LLMProvider,
    LLMFactory,
    ModelPresets,
    Settings,
)

from .schema import (
    Reference,
    CitationContext,
    SearchResult,
    ExistenceValidationResult,
    ContextAnalysisResult,
    CitationValidationResult,
    FullValidationResult,
    ValidationReport,
    ValidationStatus,
    ConsistencyStatus,
    CitationAppropriateness,
    ReferenceValidationState,
    create_initial_state,
)

from .supervisor import (
    build_graph,
    run_validation,
)


def validate_paper(
    pdf_path: str,
    max_references: int = None,
    verbose: bool = True
) -> ValidationReport:
    """
    PDF 논문의 레퍼런스를 검증합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        max_references: 최대 검증할 레퍼런스 수 (None이면 전체)
        verbose: 상세 로그 출력 여부
        
    Returns:
        ValidationReport: 검증 리포트
        
    예시:
        >>> from reference_validator import validate_paper, configure
        >>> configure(provider="ollama", model_name="llama3.2")
        >>> report = validate_paper("my_paper.pdf")
        >>> print(f"유효한 레퍼런스: {report.valid_count}/{report.total_references}")
    """
    return run_validation(pdf_path, max_references, verbose)


__version__ = "1.0.0"
__all__ = [
    # 설정
    "configure",
    "configure_multi_model",
    "settings",
    "LLMConfig",
    "LLMProvider",
    "LLMFactory",
    "ModelPresets",
    "Settings",
    
    # 스키마
    "Reference",
    "CitationContext",
    "SearchResult",
    "ExistenceValidationResult",
    "ContextAnalysisResult",
    "CitationValidationResult",
    "FullValidationResult",
    "ValidationReport",
    "ValidationStatus",
    "ConsistencyStatus",
    "CitationAppropriateness",
    "ReferenceValidationState",
    "create_initial_state",
    
    # 실행
    "build_graph",
    "run_validation",
    "validate_paper",
]
