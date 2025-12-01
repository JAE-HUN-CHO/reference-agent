"""
Citation Agent 모듈

인용 적절성을 최종 판단합니다.

역할:
- 주제 관련성 확인
- 주장 정확성 확인
- 인용 방식 평가
- 왜곡/과장 탐지
- 종합 적절성 점수 생성
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
    CitationValidationResult,
    ReferenceValidationState,
    CitationAppropriateness,
    CitationValidationOutput,
)
from config import settings


class CitationAgent:
    """인용 적절성 최종 판단을 담당하는 에이전트"""
    
    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain 호환 LLM 인스턴스
        """
        self.llm = llm or settings.get_citation_llm()
    
    def _get_appropriateness_grade(self, score: float) -> CitationAppropriateness:
        """점수에 따른 적절성 등급을 반환합니다."""
        if score >= 90:
            return CitationAppropriateness.VERY_APPROPRIATE
        elif score >= 70:
            return CitationAppropriateness.APPROPRIATE
        elif score >= 50:
            return CitationAppropriateness.PARTIALLY_APPROPRIATE
        elif score >= 30:
            return CitationAppropriateness.INAPPROPRIATE
        else:
            return CitationAppropriateness.VERY_INAPPROPRIATE
    
    def validate_citation(
        self,
        reference: Reference,
        contexts: List[CitationContext],
        context_analyses: List[ContextAnalysisResult],
        search_results: List[SearchResult]
    ) -> CitationValidationResult:
        """
        인용의 적절성을 종합적으로 판단합니다.
        
        Args:
            reference: 분석할 레퍼런스
            contexts: 해당 레퍼런스의 인용 문맥 리스트
            context_analyses: Context Agent의 분석 결과
            search_results: 검색 결과
            
        Returns:
            CitationValidationResult
        """
        parser = PydanticOutputParser(pydantic_object=CitationValidationOutput)
        
        # 입력 데이터 포맷
        reference_info = self._format_reference(reference, search_results)
        contexts_info = self._format_contexts_with_analyses(contexts, context_analyses)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 학술 논문의 인용 적절성을 판단하는 전문가입니다.
레퍼런스 정보와 인용 문맥 분석 결과를 바탕으로 인용의 전반적인 적절성을 평가해야 합니다.

평가 기준:
1. appropriateness_score (0-100): 종합 적절성 점수
   - 90-100: 매우 적절 - 인용이 정확하고 문맥에 완벽히 부합
   - 70-89: 적절 - 인용이 대체로 정확하고 문맥에 부합
   - 50-69: 부분적 적절 - 일부 문제가 있으나 큰 오류 없음
   - 30-49: 부적절 - 인용이 문맥과 맞지 않거나 불정확
   - 0-29: 매우 부적절 - 심각한 오류나 왜곡이 있음

2. topic_relevance_score (0-100): 주제 관련성
   - 인용된 논문의 주제가 본문 맥락과 얼마나 관련있는지

3. claim_accuracy_score (0-100): 주장 정확성
   - 원문에서 해당 내용을 실제로 주장하는지

4. citation_style_appropriate: 인용 방식이 적절한지 (true/false)
   - 직접 인용 vs 간접 인용이 적절히 사용되었는지

5. misrepresentation_detected: 왜곡/과장이 탐지되었는지 (true/false)
   - 원문의 의도가 왜곡되거나 과장되었는지

6. issues: 발견된 문제점 목록

{format_instructions}"""),
            ("human", """다음 레퍼런스와 인용 문맥을 분석하여 인용 적절성을 평가해 주세요:

--- 레퍼런스 정보 ---
{reference_info}

--- 인용 문맥 및 분석 결과 ---
{contexts_info}

위 정보를 바탕으로 인용 적절성 평가 결과를 JSON 형식으로 출력하세요.""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "reference_info": reference_info,
                "contexts_info": contexts_info,
            })
            
            # 결과 변환
            return CitationValidationResult(
                ref_id=reference.ref_id,
                appropriateness_score=result.appropriateness_score,
                appropriateness_grade=self._get_appropriateness_grade(result.appropriateness_score),
                topic_relevance_score=result.topic_relevance_score,
                claim_accuracy_score=result.claim_accuracy_score,
                citation_style_appropriate=result.citation_style_appropriate,
                misrepresentation_detected=result.misrepresentation_detected,
                context_analyses=context_analyses,
                issues=result.issues,
                reasoning=result.reasoning,
            )
            
        except Exception as e:
            print(f"LLM 인용 적절성 평가 오류: {e}")
            return self._evaluate_rule_based(reference, context_analyses)
    
    def _format_reference(self, reference: Reference, search_results: List[SearchResult]) -> str:
        """레퍼런스 정보를 텍스트로 포맷합니다."""
        lines = [
            f"ID: {reference.ref_id}",
            f"제목: {reference.title or '알 수 없음'}",
            f"저자: {', '.join(reference.authors) if reference.authors else '알 수 없음'}",
            f"연도: {reference.year or '알 수 없음'}",
            f"원본 텍스트: {reference.raw_text}",
        ]
        
        if search_results:
            lines.append("\n검색된 원문 내용:")
            for r in search_results[:2]:
                content = r.content or r.snippet
                if content:
                    lines.append(f"  [{r.source}] {content[:500]}...")
        
        return "\n".join(lines)
    
    def _format_contexts_with_analyses(
        self,
        contexts: List[CitationContext],
        analyses: List[ContextAnalysisResult]
    ) -> str:
        """인용 문맥과 분석 결과를 함께 포맷합니다."""
        # 분석 결과를 context_id로 인덱싱
        analysis_map = {a.context_id: a for a in analyses}
        
        formatted = []
        for ctx in contexts:
            analysis = analysis_map.get(ctx.context_id)
            
            ctx_text = f"""
[문맥: {ctx.context_id}]
인용 문장: {ctx.text_snippet}
현재 문단: {ctx.paragraph_current}
"""
            if analysis:
                ctx_text += f"""
[분석 결과]
- 관련성 점수: {analysis.relevance_score}
- 일치 여부: {analysis.consistency_check.value}
- 인용 목적: {analysis.citation_purpose}
- 뒷받침하는 주장: {analysis.claim_in_paper}
- 분석 추론: {analysis.analysis_reasoning}
"""
            formatted.append(ctx_text)
        
        return "\n---\n".join(formatted)
    
    def _evaluate_rule_based(
        self,
        reference: Reference,
        context_analyses: List[ContextAnalysisResult]
    ) -> CitationValidationResult:
        """
        규칙 기반 인용 적절성 평가 (LLM 실패 시 폴백).
        """
        if not context_analyses:
            return CitationValidationResult(
                ref_id=reference.ref_id,
                appropriateness_score=50.0,
                appropriateness_grade=CitationAppropriateness.PARTIALLY_APPROPRIATE,
                context_analyses=[],
                reasoning="분석 결과가 없어 중간 점수 부여",
            )
        
        # 평균 관련성 점수 계산
        avg_relevance = sum(a.relevance_score for a in context_analyses) / len(context_analyses)
        
        # 일치하는 문맥 비율 계산
        from schema import ConsistencyStatus
        consistent_count = sum(
            1 for a in context_analyses 
            if a.consistency_check == ConsistencyStatus.CONSISTENT
        )
        consistency_ratio = consistent_count / len(context_analyses)
        
        # 종합 점수 계산
        score = (avg_relevance * 50 + consistency_ratio * 50)
        
        return CitationValidationResult(
            ref_id=reference.ref_id,
            appropriateness_score=round(score, 1),
            appropriateness_grade=self._get_appropriateness_grade(score),
            topic_relevance_score=round(avg_relevance * 100, 1),
            claim_accuracy_score=round(consistency_ratio * 100, 1),
            context_analyses=context_analyses,
            reasoning=f"규칙 기반 평가: 평균 관련성 {avg_relevance:.2f}, 일치율 {consistency_ratio:.2%}",
        )


def citation_agent_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """
    LangGraph 노드 역할을 하는 Citation Agent 함수
    
    Args:
        state: 현재 상태
        
    Returns:
        업데이트된 상태
    """
    print("--- Citation Agent 실행: 인용 적절성 최종 판단 시작 ---")
    
    agent = CitationAgent()
    
    current_reference = state["current_reference"]
    search_results = state["search_results"]
    context_analyses = state["context_analyses"]
    all_contexts = state["citation_contexts"]
    
    if not current_reference:
        print("  ERROR: 평가할 레퍼런스가 없습니다")
        return state
    
    # 현재 레퍼런스에 해당하는 인용 문맥 필터링
    contexts_for_ref = [
        ctx for ctx in all_contexts 
        if ctx.ref_id == current_reference.ref_id
    ]
    
    print(f"  레퍼런스: {current_reference.ref_id}")
    print(f"  문맥 분석 결과: {len(context_analyses)}개")
    
    # 처리 로그 추가
    state["processing_log"].append(
        f"인용 적절성 평가 시작: {current_reference.ref_id}"
    )
    
    try:
        # 인용 적절성 평가
        result = agent.validate_citation(
            current_reference,
            contexts_for_ref,
            context_analyses,
            search_results
        )
        
        # 상태 업데이트
        state["citation_result"] = result
        
        print(f"  적절성 점수: {result.appropriateness_score}/100")
        print(f"  등급: {result.appropriateness_grade.value}")
        if result.issues:
            print(f"  문제점: {result.issues[:3]}")
        
        state["processing_log"].append(
            f"인용 적절성 평가 완료: {current_reference.ref_id} -> {result.appropriateness_score}/100 ({result.appropriateness_grade.value})"
        )
        
    except Exception as e:
        error_msg = f"Citation Agent 오류: {e}"
        print(f"  ERROR: {error_msg}")
        state["error_log"].append(error_msg)
        
        # 실패 결과 생성
        state["citation_result"] = CitationValidationResult(
            ref_id=current_reference.ref_id,
            appropriateness_score=0.0,
            appropriateness_grade=CitationAppropriateness.VERY_INAPPROPRIATE,
            reasoning=f"평가 오류: {e}",
        )
    
    return state


if __name__ == "__main__":
    # 테스트
    print("=== Citation Agent 테스트 ===")
    
    from schema import ConsistencyStatus
    
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
            text_snippet="The Transformer architecture [1] has revolutionized NLP.",
            paragraph_current="Recent advances in deep learning have been driven by attention mechanisms. The Transformer architecture [1] has revolutionized NLP.",
        ),
    ]
    
    # 테스트용 분석 결과
    test_analyses = [
        ContextAnalysisResult(
            context_id="ctx_1",
            ref_id="[1]",
            relevance_score=0.95,
            consistency_check=ConsistencyStatus.CONSISTENT,
            citation_purpose="background",
            claim_in_paper="Transformer가 NLP를 혁신했다",
            analysis_reasoning="Transformer 논문을 배경 정보로 적절히 인용함",
        ),
    ]
    
    # 테스트용 검색 결과
    test_results = [
        SearchResult(
            source="semantic_scholar",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            content="We propose a new simple network architecture, the Transformer...",
            score=0.95,
        ),
    ]
    
    agent = CitationAgent()
    
    # 인용 적절성 평가 테스트
    print("\n인용 적절성 평가:")
    result = agent.validate_citation(
        test_reference,
        test_contexts,
        test_analyses,
        test_results
    )
    
    print(f"\n  종합 점수: {result.appropriateness_score}/100")
    print(f"  등급: {result.appropriateness_grade.value}")
    print(f"  주제 관련성: {result.topic_relevance_score}/100")
    print(f"  주장 정확성: {result.claim_accuracy_score}/100")
    print(f"  인용 방식 적절: {result.citation_style_appropriate}")
    print(f"  왜곡 탐지: {result.misrepresentation_detected}")
    print(f"  추론: {result.reasoning[:200]}...")
