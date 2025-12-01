"""
Context Agent 모듈

인용 주변 맥락을 분석하여 인용이 적절하게 사용되었는지 평가합니다.

역할:
- 주변 문단 추출 및 분석
- 인용이 뒷받침하는 주장 식별
- 인용 목적 분석 (증거, 배경, 비교, 방법론 등)
- 문맥 관련성 점수 산출
"""

import os
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import (
    Reference,
    CitationContext,
    SearchResult,
    ContextAnalysisResult,
    ReferenceValidationState,
    ConsistencyStatus,
    ContextAnalysisOutput,
)
from config import settings


class ContextAgent:
    """인용 문맥 분석을 담당하는 에이전트"""
    
    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain 호환 LLM 인스턴스
        """
        self.llm = llm or settings.get_context_llm()
    
    def analyze_contexts(
        self,
        reference: Reference,
        contexts: List[CitationContext],
        search_results: List[SearchResult]
    ) -> List[ContextAnalysisResult]:
        """
        해당 레퍼런스의 모든 인용 문맥을 분석합니다.
        
        Args:
            reference: 분석할 레퍼런스
            contexts: 해당 레퍼런스의 인용 문맥 리스트
            search_results: 레퍼런스 관련 검색 결과
            
        Returns:
            ContextAnalysisResult 리스트
        """
        if not contexts:
            return []
        
        parser = PydanticOutputParser(pydantic_object=ContextAnalysisOutput)
        
        # 레퍼런스의 핵심 내용 요약
        reference_summary = self._summarize_reference(reference, search_results)
        
        # 문맥 정보 포맷
        contexts_text = self._format_contexts(contexts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 학술 논문의 인용 문맥을 분석하는 전문가입니다.
레퍼런스의 핵심 내용과 본문에서 해당 레퍼런스가 인용된 문맥을 비교하여 각 인용이 적절한지 분석해야 합니다.

분석 기준:
1. 관련성 점수(relevance_score): 0.0~1.0
   - 1.0: 인용이 문맥과 완벽하게 관련
   - 0.7-0.9: 관련성 높음
   - 0.4-0.6: 부분적 관련성
   - 0.0-0.3: 관련성 낮음

2. 일치 여부(consistency_check):
   - CONSISTENT: 인용된 내용이 원문의 주장과 일치
   - INCONSISTENT: 인용된 내용이 원문과 다르거나 왜곡됨
   - UNCLEAR: 판단하기 어려움

3. 인용 목적(citation_purpose):
   - evidence: 주장을 뒷받침하는 증거로 사용
   - background: 배경 정보 제공
   - comparison: 비교/대조를 위해 사용
   - methodology: 방법론 참조
   - definition: 정의나 개념 설명
   - other: 기타

4. claim_in_paper: 본문에서 이 인용이 뒷받침하는 주장을 요약

{format_instructions}"""),
            ("human", """다음 레퍼런스와 인용 문맥을 분석해 주세요:

--- 레퍼런스 핵심 내용 ---
{reference_summary}

--- 인용 문맥 목록 ---
{contexts_text}

각 인용 문맥에 대해 분석 결과를 JSON 형식으로 출력하세요.""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "reference_summary": reference_summary,
                "contexts_text": contexts_text,
            })
            
            # ref_id 추가
            for analysis in result.context_analyses:
                analysis.ref_id = reference.ref_id
            
            return result.context_analyses
            
        except Exception as e:
            print(f"LLM 문맥 분석 오류: {e}")
            return self._analyze_rule_based(reference, contexts)
    
    def _summarize_reference(
        self,
        reference: Reference,
        search_results: List[SearchResult]
    ) -> str:
        """레퍼런스의 핵심 내용을 요약합니다."""
        summary_parts = [
            f"레퍼런스 ID: {reference.ref_id}",
            f"제목: {reference.title or '알 수 없음'}",
            f"저자: {', '.join(reference.authors) if reference.authors else '알 수 없음'}",
            f"연도: {reference.year or '알 수 없음'}",
            f"원본 텍스트: {reference.raw_text}",
        ]
        
        # 검색 결과에서 추가 정보 추출
        if search_results:
            abstracts = []
            for r in search_results[:2]:
                if r.content:
                    abstracts.append(f"[{r.source}] {r.content[:300]}...")
                elif r.snippet:
                    abstracts.append(f"[{r.source}] {r.snippet}")
            
            if abstracts:
                summary_parts.append("\n검색된 초록/내용:")
                summary_parts.extend(abstracts)
        
        return "\n".join(summary_parts)
    
    def _format_contexts(self, contexts: List[CitationContext]) -> str:
        """인용 문맥 목록을 텍스트로 포맷합니다."""
        formatted = []
        for ctx in contexts:
            ctx_text = f"""
[문맥 ID: {ctx.context_id}]
섹션: {ctx.section or '알 수 없음'}
인용 문장: {ctx.text_snippet}
현재 문단: {ctx.paragraph_current}
"""
            if ctx.paragraph_before:
                ctx_text += f"이전 문단: {ctx.paragraph_before[:200]}...\n"
            if ctx.paragraph_after:
                ctx_text += f"다음 문단: {ctx.paragraph_after[:200]}...\n"
            
            formatted.append(ctx_text)
        
        return "\n---\n".join(formatted)
    
    def _analyze_rule_based(
        self,
        reference: Reference,
        contexts: List[CitationContext]
    ) -> List[ContextAnalysisResult]:
        """
        규칙 기반 문맥 분석 (LLM 실패 시 폴백).
        """
        results = []
        
        for ctx in contexts:
            # 기본적인 키워드 매칭으로 관련성 추정
            ref_keywords = set()
            if reference.title:
                ref_keywords.update(reference.title.lower().split())
            
            context_words = set(ctx.text_snippet.lower().split())
            
            # 키워드 겹침 계산
            overlap = len(ref_keywords & context_words)
            relevance = min(overlap / max(len(ref_keywords), 1), 1.0)
            
            results.append(ContextAnalysisResult(
                context_id=ctx.context_id,
                ref_id=reference.ref_id,
                relevance_score=round(relevance, 2),
                consistency_check=ConsistencyStatus.UNCLEAR,
                citation_purpose="unknown",
                claim_in_paper="규칙 기반 분석으로 인해 판단 불가",
                analysis_reasoning="LLM 분석 실패로 규칙 기반 분석 수행",
            ))
        
        return results


def context_agent_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """
    LangGraph 노드 역할을 하는 Context Agent 함수
    
    Args:
        state: 현재 상태
        
    Returns:
        업데이트된 상태
    """
    print("--- Context Agent 실행: 인용 문맥 분석 시작 ---")
    
    agent = ContextAgent()
    
    current_reference = state["current_reference"]
    search_results = state["search_results"]
    all_contexts = state["citation_contexts"]
    
    if not current_reference:
        print("  ERROR: 분석할 레퍼런스가 없습니다")
        return state
    
    # 현재 레퍼런스에 해당하는 인용 문맥만 필터링
    contexts_for_ref = [
        ctx for ctx in all_contexts 
        if ctx.ref_id == current_reference.ref_id
    ]
    
    print(f"  레퍼런스: {current_reference.ref_id}")
    print(f"  인용 문맥: {len(contexts_for_ref)}개")
    
    # 처리 로그 추가
    state["processing_log"].append(
        f"문맥 분석 시작: {current_reference.ref_id} ({len(contexts_for_ref)}개 문맥)"
    )
    
    if not contexts_for_ref:
        print("  인용 문맥이 없어 분석을 건너뜁니다")
        state["context_analyses"] = []
        return state
    
    try:
        # 문맥 분석 수행
        analyses = agent.analyze_contexts(
            current_reference,
            contexts_for_ref,
            search_results
        )
        
        # 상태 업데이트
        state["context_analyses"] = analyses
        
        print(f"  분석 완료: {len(analyses)}개 문맥")
        for analysis in analyses[:3]:
            print(f"    - {analysis.context_id}: 관련성 {analysis.relevance_score:.2f}, {analysis.consistency_check.value}")
        
        state["processing_log"].append(
            f"문맥 분석 완료: {current_reference.ref_id} -> {len(analyses)}개 분석"
        )
        
    except Exception as e:
        error_msg = f"Context Agent 오류: {e}"
        print(f"  ERROR: {error_msg}")
        state["error_log"].append(error_msg)
        state["context_analyses"] = []
    
    return state


if __name__ == "__main__":
    # 테스트
    print("=== Context Agent 테스트 ===")
    
    # 테스트용 레퍼런스
    test_reference = Reference(
        ref_id="[1]",
        raw_text='Vaswani, A., et al. (2017). Attention is all you need.',
        title="Attention is all you need",
        authors=["A. Vaswani"],
        year=2017,
    )
    
    # 테스트용 인용 문맥
    test_contexts = [
        CitationContext(
            context_id="ctx_1",
            ref_id="[1]",
            text_snippet="The Transformer architecture [1] has revolutionized natural language processing.",
            paragraph_current="Recent advances in deep learning have been largely driven by attention mechanisms. The Transformer architecture [1] has revolutionized natural language processing by replacing recurrent structures with self-attention.",
        ),
        CitationContext(
            context_id="ctx_2",
            ref_id="[1]",
            text_snippet="Following [1], we apply multi-head attention to capture different aspects of the input.",
            paragraph_current="Our method builds upon the self-attention mechanism. Following [1], we apply multi-head attention to capture different aspects of the input sequence.",
        ),
    ]
    
    # 테스트용 검색 결과
    test_results = [
        SearchResult(
            source="semantic_scholar",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            snippet="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            content="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            score=0.95,
        ),
    ]
    
    agent = ContextAgent()
    
    # 문맥 분석 테스트
    print("\n문맥 분석 실행:")
    analyses = agent.analyze_contexts(test_reference, test_contexts, test_results)
    
    for analysis in analyses:
        print(f"\n  {analysis.context_id}:")
        print(f"    관련성: {analysis.relevance_score:.2f}")
        print(f"    일치: {analysis.consistency_check.value}")
        print(f"    목적: {analysis.citation_purpose}")
        print(f"    주장: {analysis.claim_in_paper[:100]}...")
