"""
Web Agent 모듈

레퍼런스 검증을 위한 웹 검색을 수행합니다.

역할:
- 검색어 생성 (LLM 기반)
- 다중 소스 검색 (Tavily, Semantic Scholar, CrossRef)
- 검색 결과 수집 및 정리
- 재시도 로직 구현
"""

import os
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import (
    Reference,
    SearchResult,
    ReferenceValidationState,
    SearchQueryListOutput,
)
from config import settings
from tools.search_tools import create_search_tool, UnifiedSearchTool


class WebAgent:
    """웹 검색을 담당하는 에이전트"""
    
    def __init__(self, llm=None, search_tool: Optional[UnifiedSearchTool] = None):
        """
        Args:
            llm: LangChain 호환 LLM 인스턴스
            search_tool: 통합 검색 도구
        """
        self.llm = llm or settings.get_web_llm()
        self.search_tool = search_tool or create_search_tool(
            tavily_api_key=settings.tavily_api_key
        )
    
    def generate_search_queries(self, reference: Reference, retry_count: int = 0) -> List[str]:
        """
        LLM을 사용하여 레퍼런스 검증을 위한 검색어를 생성합니다.
        
        Args:
            reference: 검증할 레퍼런스
            retry_count: 재시도 횟수 (검색 전략 변경에 사용)
            
        Returns:
            검색어 리스트
        """
        parser = PydanticOutputParser(pydantic_object=SearchQueryListOutput)
        
        # 재시도 시 다른 전략 사용
        strategy_hints = [
            "정확한 제목과 저자로 검색어를 생성하세요.",
            "제목의 핵심 키워드와 연도를 조합하여 검색어를 생성하세요.",
            "저자명과 출판 연도를 중심으로 검색어를 생성하세요.",
        ]
        strategy = strategy_hints[min(retry_count, len(strategy_hints) - 1)]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 학술 논문 검색 전문가입니다.
제공된 레퍼런스 정보를 바탕으로, 해당 논문이 실제로 존재하는지 확인할 수 있는 검색어 3개를 생성해야 합니다.

검색 전략: {strategy}

검색어 생성 가이드라인:
1. 첫 번째 검색어: 논문 제목을 정확하게 포함
2. 두 번째 검색어: 제목 + 첫 번째 저자명
3. 세 번째 검색어: 핵심 키워드 + 연도 + 저널/학회명

검색어는 Google Scholar, Semantic Scholar에서 효과적으로 검색될 수 있어야 합니다.

{format_instructions}"""),
            ("human", """다음 레퍼런스에 대한 검색어 3개를 생성해 주세요:

레퍼런스 ID: {ref_id}
제목: {title}
저자: {authors}
연도: {year}
원본 텍스트: {raw_text}

검색어를 JSON 형식으로 출력하세요.""")
        ]).partial(
            format_instructions=parser.get_format_instructions(),
            strategy=strategy
        )
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "ref_id": reference.ref_id,
                "title": reference.title or "정보 없음",
                "authors": ", ".join(reference.authors) if reference.authors else "정보 없음",
                "year": reference.year or "정보 없음",
                "raw_text": reference.raw_text[:500],  # 최대 500자
            })
            return result.queries
            
        except Exception as e:
            print(f"LLM 검색어 생성 오류: {e}")
            # 폴백: 기본 검색어 반환
            return self._generate_fallback_queries(reference)
    
    def _generate_fallback_queries(self, reference: Reference) -> List[str]:
        """LLM 실패 시 기본 검색어를 생성합니다."""
        queries = []
        
        # 제목 기반 검색어
        if reference.title:
            queries.append(reference.title)
        
        # 제목 + 저자 기반 검색어
        if reference.title and reference.authors:
            queries.append(f"{reference.title} {reference.authors[0]}")
        
        # 원본 텍스트 기반 검색어 (앞부분만)
        if reference.raw_text:
            queries.append(reference.raw_text[:100])
        
        # 최소 1개 검색어 보장
        if not queries:
            queries.append(reference.raw_text[:200] if reference.raw_text else reference.ref_id)
        
        return queries[:3]  # 최대 3개
    
    def search_reference(
        self,
        reference: Reference,
        queries: List[str],
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        여러 검색어와 소스를 사용하여 레퍼런스를 검색합니다.
        
        Args:
            reference: 검증할 레퍼런스
            queries: 검색어 리스트
            max_results: 최대 결과 수
            
        Returns:
            SearchResult 리스트
        """
        all_results = []
        
        # DOI가 있으면 먼저 DOI로 검색
        if reference.doi:
            print(f"    DOI로 검색 중: {reference.doi}")
            doi_result = self.search_tool.search_by_doi(reference.doi)
            if doi_result:
                all_results.append(doi_result)
                print(f"      DOI 검색 성공: {doi_result.title[:50]}...")
        
        # URL이 있으면 크롤링
        if reference.url:
            print(f"    URL 크롤링 중: {reference.url}")
            url_result = self.search_tool.crawl_url(reference.url)
            if url_result:
                all_results.append(url_result)
                print(f"      크롤링 성공")
        
        # 검색어로 검색
        if len(all_results) < max_results:
            remaining = max_results - len(all_results)
            search_results = self.search_tool.search_with_strategies(
                queries,
                max_total_results=remaining
            )
            all_results.extend(search_results)
        
        return all_results
    
    def search_with_retry(
        self,
        reference: Reference,
        max_retries: int = 3
    ) -> tuple[List[str], List[SearchResult], int]:
        """
        재시도 로직을 포함한 검색을 수행합니다.
        
        Args:
            reference: 검증할 레퍼런스
            max_retries: 최대 재시도 횟수
            
        Returns:
            (검색어 리스트, 검색 결과 리스트, 재시도 횟수) 튜플
        """
        for retry in range(max_retries):
            print(f"\n  검색 시도 {retry + 1}/{max_retries}")
            
            # 검색어 생성
            queries = self.generate_search_queries(reference, retry_count=retry)
            print(f"    생성된 검색어: {queries}")
            
            # 검색 수행
            results = self.search_reference(reference, queries)
            
            if results:
                print(f"    검색 결과: {len(results)}개 발견")
                return queries, results, retry + 1
            
            print(f"    검색 결과 없음, 재시도...")
        
        return queries, [], max_retries


def web_agent_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """
    LangGraph 노드 역할을 하는 Web Agent 함수
    
    Args:
        state: 현재 상태
        
    Returns:
        업데이트된 상태
    """
    print("--- Web Agent 실행: 레퍼런스 검색 시작 ---")
    
    agent = WebAgent()
    
    # 현재 검증할 레퍼런스 가져오기
    current_ref_index = state["current_ref_index"]
    
    if current_ref_index >= len(state["references"]):
        print("  모든 레퍼런스 처리 완료")
        return state
    
    current_reference = state["references"][current_ref_index]
    state["current_reference"] = current_reference
    
    print(f"  현재 레퍼런스: {current_reference.ref_id}")
    print(f"    제목: {current_reference.title or current_reference.raw_text[:50]}...")
    
    # 처리 로그 추가
    state["processing_log"].append(
        f"Web 검색 시작: {current_reference.ref_id}"
    )
    
    try:
        # 검색 수행 (재시도 포함)
        queries, results, retry_count = agent.search_with_retry(
            current_reference,
            max_retries=settings.max_retry_attempts
        )
        
        # 상태 업데이트
        state["search_queries"] = queries
        state["search_results"] = results
        state["retry_count"] = retry_count
        
        state["processing_log"].append(
            f"Web 검색 완료: {current_reference.ref_id} - {len(results)}개 결과 ({retry_count}회 시도)"
        )
        
    except Exception as e:
        error_msg = f"Web Agent 오류: {e}"
        print(f"  ERROR: {error_msg}")
        state["error_log"].append(error_msg)
        state["search_results"] = []
    
    return state


if __name__ == "__main__":
    # 테스트
    print("=== Web Agent 테스트 ===")
    
    # 테스트용 레퍼런스
    test_reference = Reference(
        ref_id="[1]",
        raw_text='Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.',
        title="Attention is all you need",
        authors=["A. Vaswani", "N. Shazeer", "N. Parmar"],
        year=2017,
    )
    
    agent = WebAgent()
    
    # 검색어 생성 테스트
    print("\n1. 검색어 생성:")
    queries = agent.generate_search_queries(test_reference)
    for q in queries:
        print(f"   - {q}")
    
    # 검색 테스트
    print("\n2. 검색 수행:")
    results = agent.search_reference(test_reference, queries, max_results=5)
    for r in results:
        print(f"   [{r.source}] {r.title[:50]}...")
