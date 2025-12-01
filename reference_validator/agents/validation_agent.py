"""
Validation Agent 모듈

검색 결과를 분석하여 레퍼런스의 존재 여부와 정보 일치도를 검증합니다.

역할:
- 제목 유사도 비교
- 저자명 일치 확인
- 출판년도 확인
- 저널/학회 확인
- 종합 신뢰도 점수 산출
"""

import os
import re
from typing import List, Optional
from difflib import SequenceMatcher
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import (
    Reference,
    SearchResult,
    ExistenceValidationResult,
    ReferenceValidationState,
    ValidationStatus,
    ExistenceValidationOutput,
)
from config import settings


class ValidationAgent:
    """레퍼런스 존재 검증을 담당하는 에이전트"""
    
    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain 호환 LLM 인스턴스
        """
        self.llm = llm or settings.get_validation_llm()
        self.title_threshold = settings.title_similarity_threshold
        self.author_threshold = settings.author_match_threshold
    
    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        두 제목 간의 유사도를 계산합니다.
        
        Args:
            title1: 첫 번째 제목
            title2: 두 번째 제목
            
        Returns:
            유사도 점수 (0.0 ~ 1.0)
        """
        if not title1 or not title2:
            return 0.0
        
        # 정규화: 소문자 변환, 특수문자 제거
        def normalize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return ' '.join(text.split())
        
        norm1 = normalize(title1)
        norm2 = normalize(title2)
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def calculate_author_match(self, ref_authors: List[str], found_authors: List[str]) -> float:
        """
        저자 목록 일치율을 계산합니다.
        
        Args:
            ref_authors: 레퍼런스의 저자 목록
            found_authors: 검색된 저자 목록
            
        Returns:
            일치율 (0.0 ~ 1.0)
        """
        if not ref_authors or not found_authors:
            return 0.0
        
        # 저자명 정규화 함수
        def normalize_author(name):
            # 소문자, 공백 제거, 이니셜 처리
            name = name.lower().strip()
            # "A. B. Smith" -> "smith"
            parts = name.replace('.', ' ').split()
            if parts:
                # 가장 긴 부분(성)을 반환
                return max(parts, key=len)
            return name
        
        ref_normalized = {normalize_author(a) for a in ref_authors}
        found_normalized = {normalize_author(a) for a in found_authors}
        
        # 교집합 / 합집합
        if not ref_normalized:
            return 0.0
        
        intersection = len(ref_normalized & found_normalized)
        return intersection / len(ref_normalized)
    
    def check_year_match(self, ref_year: Optional[int], found_year: Optional[int]) -> bool:
        """
        연도 일치 여부를 확인합니다 (±1년 허용).
        
        Args:
            ref_year: 레퍼런스 연도
            found_year: 검색된 연도
            
        Returns:
            일치 여부
        """
        if ref_year is None or found_year is None:
            return False
        
        return abs(ref_year - found_year) <= 1
    
    def validate_with_llm(
        self,
        reference: Reference,
        search_results: List[SearchResult]
    ) -> ExistenceValidationOutput:
        """
        LLM을 사용하여 종합적인 검증을 수행합니다.
        
        Args:
            reference: 검증할 레퍼런스
            search_results: 검색 결과 리스트
            
        Returns:
            검증 결과
        """
        parser = PydanticOutputParser(pydantic_object=ExistenceValidationOutput)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 학술 논문 레퍼런스의 유효성을 검증하는 전문가입니다.
제공된 원본 레퍼런스 정보와 웹 검색 결과를 비교하여 레퍼런스의 상태를 판단해야 합니다.

검증 기준:
1. 제목 유사도: 85% 이상이면 일치로 간주
2. 저자 일치: 첫 번째 저자 필수, 전체의 50% 이상 일치
3. 연도 일치: 정확히 일치하거나 ±1년 허용
4. DOI 일치: DOI가 있으면 완전 일치 필수

검증 상태:
- VALID: 논문이 존재하고 주요 정보가 일치
- INVALID: 논문이 존재하지 않거나 정보가 심각하게 불일치
- UNVERIFIABLE: 검색 결과가 부족하여 판단 불가

신뢰도 점수(confidence_score)는 0.0~1.0 사이로 부여하세요.

{format_instructions}"""),
            ("human", """다음 레퍼런스를 검증해 주세요:

--- 원본 레퍼런스 ---
ID: {ref_id}
제목: {title}
저자: {authors}
연도: {year}
DOI: {doi}
원본 텍스트: {raw_text}

--- 검색 결과 ---
{search_results_text}

위 정보를 바탕으로 레퍼런스 검증 결과를 JSON 형식으로 출력하세요.""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # 검색 결과 텍스트 생성
        search_results_text = self._format_search_results(search_results)
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "ref_id": reference.ref_id,
                "title": reference.title or "정보 없음",
                "authors": ", ".join(reference.authors) if reference.authors else "정보 없음",
                "year": reference.year or "정보 없음",
                "doi": reference.doi or "없음",
                "raw_text": reference.raw_text[:500],
                "search_results_text": search_results_text,
            })
            return result
            
        except Exception as e:
            print(f"LLM 검증 오류: {e}")
            return self._validate_rule_based(reference, search_results)
    
    def _format_search_results(self, results: List[SearchResult]) -> str:
        """검색 결과를 텍스트로 포맷합니다."""
        if not results:
            return "검색 결과 없음"
        
        formatted = []
        for i, r in enumerate(results[:5], 1):  # 최대 5개만
            formatted.append(f"""
[결과 {i}] 소스: {r.source}
제목: {r.title}
URL: {r.url}
스니펫: {r.snippet[:300] if r.snippet else '없음'}
""")
        return "\n".join(formatted)
    
    def _validate_rule_based(
        self,
        reference: Reference,
        search_results: List[SearchResult]
    ) -> ExistenceValidationOutput:
        """
        규칙 기반 검증 (LLM 실패 시 폴백).
        """
        if not search_results:
            return ExistenceValidationOutput(
                status="UNVERIFIABLE",
                confidence_score=0.0,
                reasoning="검색 결과가 없어 검증할 수 없습니다.",
            )
        
        best_match = None
        best_score = 0.0
        
        for result in search_results:
            # 제목 유사도 계산
            title_score = self.calculate_title_similarity(
                reference.title or reference.raw_text[:100],
                result.title
            )
            
            if title_score > best_score:
                best_score = title_score
                best_match = result
        
        # 상태 결정
        if best_score >= self.title_threshold:
            status = "VALID"
            confidence = best_score
        elif best_score >= 0.5:
            status = "UNVERIFIABLE"
            confidence = best_score * 0.7
        else:
            status = "INVALID"
            confidence = 1.0 - best_score
        
        return ExistenceValidationOutput(
            status=status,
            confidence_score=round(confidence, 2),
            title_match_score=round(best_score, 2),
            reasoning=f"최고 제목 유사도: {best_score:.2%}, 매칭 결과: {best_match.title if best_match else '없음'}",
        )
    
    def validate_reference(
        self,
        reference: Reference,
        search_results: List[SearchResult],
        search_queries: List[str]
    ) -> ExistenceValidationResult:
        """
        레퍼런스를 종합적으로 검증합니다.
        
        Args:
            reference: 검증할 레퍼런스
            search_results: 검색 결과 리스트
            search_queries: 사용된 검색어 리스트
            
        Returns:
            ExistenceValidationResult
        """
        # LLM 기반 검증
        llm_result = self.validate_with_llm(reference, search_results)
        
        # 결과 생성
        return ExistenceValidationResult(
            ref_id=reference.ref_id,
            status=ValidationStatus(llm_result.status),
            confidence_score=llm_result.confidence_score,
            search_queries=search_queries,
            search_results=search_results,
            title_match_score=llm_result.title_match_score,
            author_match_score=llm_result.author_match_score,
            year_match=llm_result.year_match,
            reasoning=llm_result.reasoning,
            suggested_fix=llm_result.suggested_fix,
        )


def validation_agent_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """
    LangGraph 노드 역할을 하는 Validation Agent 함수
    
    Args:
        state: 현재 상태
        
    Returns:
        업데이트된 상태
    """
    print("--- Validation Agent 실행: 레퍼런스 유효성 검증 시작 ---")
    
    agent = ValidationAgent()
    
    current_reference = state["current_reference"]
    search_results = state["search_results"]
    search_queries = state["search_queries"]
    
    if not current_reference:
        print("  ERROR: 검증할 레퍼런스가 없습니다")
        return state
    
    print(f"  레퍼런스: {current_reference.ref_id}")
    print(f"  검색 결과: {len(search_results)}개")
    
    # 처리 로그 추가
    state["processing_log"].append(
        f"검증 시작: {current_reference.ref_id}"
    )
    
    try:
        # 검증 수행
        result = agent.validate_reference(
            current_reference,
            search_results,
            search_queries
        )
        
        # 상태 업데이트
        state["existence_result"] = result
        
        print(f"  결과: {result.status.value}")
        print(f"  신뢰도: {result.confidence_score:.2f}")
        print(f"  추론: {result.reasoning[:100]}...")
        
        state["processing_log"].append(
            f"검증 완료: {current_reference.ref_id} -> {result.status.value} ({result.confidence_score:.2f})"
        )
        
    except Exception as e:
        error_msg = f"Validation Agent 오류: {e}"
        print(f"  ERROR: {error_msg}")
        state["error_log"].append(error_msg)
        
        # 실패 결과 생성
        state["existence_result"] = ExistenceValidationResult(
            ref_id=current_reference.ref_id,
            status=ValidationStatus.UNVERIFIABLE,
            confidence_score=0.0,
            search_queries=search_queries,
            search_results=search_results,
            reasoning=f"검증 오류: {e}",
        )
    
    return state


if __name__ == "__main__":
    # 테스트
    print("=== Validation Agent 테스트 ===")
    
    # 테스트용 레퍼런스
    test_reference = Reference(
        ref_id="[1]",
        raw_text='Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.',
        title="Attention is all you need",
        authors=["A. Vaswani", "N. Shazeer", "N. Parmar"],
        year=2017,
    )
    
    # 테스트용 검색 결과
    test_results = [
        SearchResult(
            source="semantic_scholar",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            snippet="[2017] Ashish Vaswani, Noam Shazeer, Niki Parmar. The dominant sequence transduction models are based on complex recurrent...",
            score=0.95,
        ),
        SearchResult(
            source="crossref",
            title="Attention is All you Need",
            url="https://doi.org/10.5555/3295222.3295349",
            snippet="[2017] Vaswani, Shazeer, Parmar. NIPS 2017",
            score=0.9,
        ),
    ]
    
    agent = ValidationAgent()
    
    # 제목 유사도 테스트
    print("\n1. 제목 유사도 테스트:")
    similarity = agent.calculate_title_similarity(
        "Attention is all you need",
        "Attention Is All You Need"
    )
    print(f"   유사도: {similarity:.2%}")
    
    # 전체 검증 테스트
    print("\n2. 전체 검증 테스트:")
    result = agent.validate_reference(test_reference, test_results, ["attention is all you need"])
    print(f"   상태: {result.status.value}")
    print(f"   신뢰도: {result.confidence_score:.2f}")
    print(f"   추론: {result.reasoning}")
