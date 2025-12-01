"""
데이터 스키마 정의 모듈

DeepAgents 아키텍처의 상태 스키마 및 데이터 모델을 정의합니다.
"""

from typing import List, TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# 열거형 정의
# ============================================================================

class ValidationStatus(str, Enum):
    """레퍼런스 검증 상태"""
    VALID = "VALID"           # 유효함 - 실제 존재하고 정보가 일치
    INVALID = "INVALID"       # 유효하지 않음 - 존재하지 않거나 정보 불일치
    UNVERIFIABLE = "UNVERIFIABLE"  # 검증 불가 - 정보 부족 또는 검색 실패


class ConsistencyStatus(str, Enum):
    """인용 내용 일치 상태"""
    CONSISTENT = "CONSISTENT"       # 일치 - 원문과 인용 내용이 일치
    INCONSISTENT = "INCONSISTENT"   # 불일치 - 원문과 인용 내용이 다름
    UNCLEAR = "UNCLEAR"             # 불명확 - 판단하기 어려움


class CitationAppropriateness(str, Enum):
    """인용 적절성 등급"""
    VERY_APPROPRIATE = "VERY_APPROPRIATE"        # 90-100점: 매우 적절
    APPROPRIATE = "APPROPRIATE"                   # 70-89점: 적절
    PARTIALLY_APPROPRIATE = "PARTIALLY_APPROPRIATE"  # 50-69점: 부분적 적절
    INAPPROPRIATE = "INAPPROPRIATE"               # 30-49점: 부적절
    VERY_INAPPROPRIATE = "VERY_INAPPROPRIATE"     # 0-29점: 매우 부적절/오류


# ============================================================================
# 기본 데이터 모델
# ============================================================================

class Reference(BaseModel):
    """논문에서 추출된 단일 레퍼런스 정보"""
    ref_id: str = Field(..., description="논문 내에서 레퍼런스를 식별하는 고유 ID (예: [1], [2])")
    raw_text: str = Field(..., description="레퍼런스 섹션에서 추출된 원본 텍스트")
    title: Optional[str] = Field(None, description="논문/책 제목")
    authors: Optional[List[str]] = Field(None, description="저자 목록")
    year: Optional[int] = Field(None, description="출판 연도")
    venue: Optional[str] = Field(None, description="저널/학회명")
    doi: Optional[str] = Field(None, description="DOI (Digital Object Identifier)")
    url: Optional[str] = Field(None, description="주요 URL")


class CitationContext(BaseModel):
    """논문 본문에서 레퍼런스가 인용된 문맥 정보"""
    context_id: str = Field(..., description="문맥을 식별하는 고유 ID")
    ref_id: str = Field(..., description="인용된 레퍼런스의 ID (Reference.ref_id와 일치)")
    text_snippet: str = Field(..., description="인용이 포함된 주변 문장")
    paragraph_before: Optional[str] = Field(None, description="인용 위치의 이전 문단")
    paragraph_current: str = Field("", description="인용이 포함된 현재 문단")
    paragraph_after: Optional[str] = Field(None, description="인용 위치의 다음 문단")
    page_number: Optional[int] = Field(None, description="인용이 발생한 페이지 번호")
    section: Optional[str] = Field(None, description="인용이 발생한 섹션명")


class SearchResult(BaseModel):
    """웹 검색 결과"""
    source: str = Field(..., description="검색 소스 (예: 'tavily', 'crawl', 'google_scholar')")
    title: str = Field("", description="검색 결과 제목")
    url: str = Field("", description="검색 결과 URL")
    snippet: str = Field("", description="검색 결과 스니펫/요약")
    content: Optional[str] = Field(None, description="크롤링된 전체 내용")
    score: float = Field(0.0, description="검색 관련도 점수")


# ============================================================================
# 검증 결과 모델
# ============================================================================

class ExistenceValidationResult(BaseModel):
    """레퍼런스 존재 검증 결과"""
    ref_id: str = Field(..., description="검증된 레퍼런스의 ID")
    status: ValidationStatus = Field(..., description="검증 상태")
    confidence_score: float = Field(..., description="검증 결과에 대한 신뢰도 (0.0 ~ 1.0)")
    
    # 검색 정보
    search_queries: List[str] = Field(default_factory=list, description="검증에 사용된 검색어 목록")
    search_results: List[SearchResult] = Field(default_factory=list, description="검색 결과 목록")
    retry_count: int = Field(0, description="검증 시도 횟수")
    
    # 비교 결과
    title_match_score: float = Field(0.0, description="제목 유사도 점수")
    author_match_score: float = Field(0.0, description="저자 일치 점수")
    year_match: bool = Field(False, description="연도 일치 여부")
    
    # 추론 및 제안
    reasoning: str = Field("", description="검증 상태를 결정한 상세 추론 과정")
    suggested_fix: Optional[str] = Field(None, description="오류가 발견된 경우 제안하는 수정 사항")


class ContextAnalysisResult(BaseModel):
    """단일 인용 문맥에 대한 분석 결과"""
    context_id: str = Field(..., description="분석된 인용 문맥의 ID")
    ref_id: str = Field(..., description="해당 레퍼런스 ID")
    
    # 분석 결과
    relevance_score: float = Field(..., description="인용된 레퍼런스와 문맥의 관련성 점수 (0.0 ~ 1.0)")
    consistency_check: ConsistencyStatus = Field(..., description="인용된 내용이 레퍼런스의 핵심 주장과 일치하는지 여부")
    
    # 인용 목적
    citation_purpose: str = Field("", description="인용 목적 (증거, 배경, 비교, 방법론 등)")
    claim_in_paper: str = Field("", description="논문에서 인용이 뒷받침하는 주장")
    
    # 추론
    analysis_reasoning: str = Field("", description="관련성 점수 및 일치 여부를 결정한 상세 추론 과정")


class CitationValidationResult(BaseModel):
    """인용 적절성 최종 검증 결과"""
    ref_id: str = Field(..., description="검증된 레퍼런스의 ID")
    
    # 종합 점수
    appropriateness_score: float = Field(..., description="인용 적절성 종합 점수 (0-100)")
    appropriateness_grade: CitationAppropriateness = Field(..., description="적절성 등급")
    
    # 세부 평가
    topic_relevance_score: float = Field(0.0, description="주제 관련성 점수 (0-100)")
    claim_accuracy_score: float = Field(0.0, description="주장 정확성 점수 (0-100)")
    citation_style_appropriate: bool = Field(True, description="인용 방식이 적절한지")
    misrepresentation_detected: bool = Field(False, description="왜곡/과장 탐지 여부")
    
    # 문맥별 분석 결과
    context_analyses: List[ContextAnalysisResult] = Field(default_factory=list, description="각 문맥에 대한 분석 결과")
    
    # 문제점
    issues: List[str] = Field(default_factory=list, description="발견된 문제점 목록")
    reasoning: str = Field("", description="최종 판단에 대한 상세 추론")


class FullValidationResult(BaseModel):
    """레퍼런스에 대한 전체 검증 결과 (존재 + 인용 적절성)"""
    ref_id: str = Field(..., description="검증된 레퍼런스의 ID")
    reference: Reference = Field(..., description="원본 레퍼런스 정보")
    
    # 존재 검증 결과
    existence_result: Optional[ExistenceValidationResult] = Field(None, description="존재 검증 결과")
    
    # 인용 적절성 결과
    citation_result: Optional[CitationValidationResult] = Field(None, description="인용 적절성 결과")
    
    # 최종 상태
    is_valid: bool = Field(False, description="최종 유효성")
    overall_score: float = Field(0.0, description="종합 점수")
    summary: str = Field("", description="검증 결과 요약")


# ============================================================================
# 최종 리포트 모델
# ============================================================================

class ValidationReport(BaseModel):
    """전체 논문에 대한 최종 검증 리포트"""
    # 논문 정보
    paper_title: str = Field("", description="검증된 논문의 제목")
    paper_path: str = Field("", description="논문 파일 경로")
    
    # 통계
    total_references: int = Field(0, description="총 레퍼런스 개수")
    valid_count: int = Field(0, description="유효한 레퍼런스 개수")
    invalid_count: int = Field(0, description="유효하지 않은 레퍼런스 개수")
    unverifiable_count: int = Field(0, description="검증 불가능한 레퍼런스 개수")
    
    # 인용 적절성 통계
    appropriate_citations: int = Field(0, description="적절한 인용 개수")
    inappropriate_citations: int = Field(0, description="부적절한 인용 개수")
    average_appropriateness_score: float = Field(0.0, description="평균 적절성 점수")
    
    # 상세 결과
    results: List[FullValidationResult] = Field(default_factory=list, description="각 레퍼런스별 상세 검증 결과")
    
    # 문제 레퍼런스 목록
    invalid_references: List[str] = Field(default_factory=list, description="존재하지 않는 레퍼런스 ID 목록")
    inappropriate_citations_list: List[Dict[str, Any]] = Field(default_factory=list, description="부적절한 인용 상세 정보")
    
    # 요약
    summary: str = Field("", description="전체 검증 결과에 대한 요약 및 권장 사항")
    recommendations: List[str] = Field(default_factory=list, description="권장 수정 사항 목록")


# ============================================================================
# LangGraph 상태 스키마
# ============================================================================

class ReferenceValidationState(TypedDict):
    """LangGraph의 상태를 정의하는 TypedDict"""
    
    # ========== 입력 데이터 ==========
    paper_path: str                          # 원본 논문 파일 경로
    paper_content: str                       # 원본 논문 텍스트
    paper_title: str                         # 논문 제목
    
    # ========== 파싱 결과 ==========
    references: List[Reference]              # 추출된 전체 레퍼런스 목록
    citation_contexts: List[CitationContext] # 추출된 인용 문맥 목록
    
    # ========== 현재 처리 상태 ==========
    current_ref_index: int                   # 현재 검증 중인 레퍼런스 인덱스
    current_reference: Optional[Reference]   # 현재 검증 중인 레퍼런스
    retry_count: int                         # 현재 레퍼런스에 대한 재시도 횟수
    
    # ========== 검색 결과 ==========
    search_queries: List[str]                # 생성된 검색어 목록
    search_results: List[SearchResult]       # 웹 검색 결과
    
    # ========== 에이전트 결과 ==========
    existence_result: Optional[ExistenceValidationResult]   # 존재 검증 결과
    context_analyses: List[ContextAnalysisResult]          # 문맥 분석 결과
    citation_result: Optional[CitationValidationResult]    # 인용 적절성 결과
    
    # ========== 누적 결과 ==========
    all_validation_results: List[FullValidationResult]     # 모든 검증 결과 누적
    
    # ========== 최종 결과 ==========
    final_report: Optional[ValidationReport]               # 최종 검증 리포트
    
    # ========== 메타 정보 ==========
    error_log: List[str]                     # 오류 로그
    processing_log: List[str]                # 처리 로그


def create_initial_state(paper_path: str) -> ReferenceValidationState:
    """초기 상태를 생성합니다."""
    return ReferenceValidationState(
        # 입력 데이터
        paper_path=paper_path,
        paper_content="",
        paper_title="",
        
        # 파싱 결과
        references=[],
        citation_contexts=[],
        
        # 현재 처리 상태
        current_ref_index=0,
        current_reference=None,
        retry_count=0,
        
        # 검색 결과
        search_queries=[],
        search_results=[],
        
        # 에이전트 결과
        existence_result=None,
        context_analyses=[],
        citation_result=None,
        
        # 누적 결과
        all_validation_results=[],
        
        # 최종 결과
        final_report=None,
        
        # 메타 정보
        error_log=[],
        processing_log=[],
    )


# ============================================================================
# LLM 출력 스키마 (파싱용)
# ============================================================================

class ReferenceListOutput(BaseModel):
    """LLM 출력용 레퍼런스 목록"""
    references: List[Reference] = Field(..., description="논문에서 추출된 레퍼런스 목록")


class CitationContextListOutput(BaseModel):
    """LLM 출력용 인용 문맥 목록"""
    citation_contexts: List[CitationContext] = Field(..., description="논문 본문에서 추출된 인용 문맥 목록")


class SearchQueryListOutput(BaseModel):
    """LLM 출력용 검색어 목록"""
    queries: List[str] = Field(..., description="레퍼런스 검증을 위한 검색어 목록 (최대 3개)")


class ExistenceValidationOutput(BaseModel):
    """LLM 출력용 존재 검증 결과"""
    status: str = Field(..., description="검증 상태: VALID, INVALID, UNVERIFIABLE")
    confidence_score: float = Field(..., description="신뢰도 (0.0 ~ 1.0)")
    title_match_score: float = Field(0.0, description="제목 유사도")
    author_match_score: float = Field(0.0, description="저자 일치율")
    year_match: bool = Field(False, description="연도 일치 여부")
    reasoning: str = Field(..., description="상세 추론 과정")
    suggested_fix: Optional[str] = Field(None, description="수정 제안")


class ContextAnalysisOutput(BaseModel):
    """LLM 출력용 문맥 분석 결과"""
    context_analyses: List[ContextAnalysisResult] = Field(..., description="각 문맥에 대한 분석 결과")


class CitationValidationOutput(BaseModel):
    """LLM 출력용 인용 적절성 결과"""
    appropriateness_score: float = Field(..., description="종합 점수 (0-100)")
    topic_relevance_score: float = Field(0.0, description="주제 관련성 점수")
    claim_accuracy_score: float = Field(0.0, description="주장 정확성 점수")
    citation_style_appropriate: bool = Field(True, description="인용 방식 적절성")
    misrepresentation_detected: bool = Field(False, description="왜곡 탐지 여부")
    issues: List[str] = Field(default_factory=list, description="문제점 목록")
    reasoning: str = Field(..., description="상세 추론")


if __name__ == "__main__":
    # 테스트
    print("=== Schema 테스트 ===")
    
    # 레퍼런스 생성 테스트
    ref = Reference(
        ref_id="[1]",
        raw_text="Jain, A. K., Murty, M. N., & Flynn, P. J. (1999). Data clustering: a review.",
        title="Data clustering: a review",
        authors=["A. K. Jain", "M. N. Murty", "P. J. Flynn"],
        year=1999,
    )
    print(f"\n레퍼런스: {ref.ref_id} - {ref.title}")
    
    # 초기 상태 생성 테스트
    state = create_initial_state("test.pdf")
    print(f"\n초기 상태 생성 완료: {len(state.keys())} 필드")
