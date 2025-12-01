"""
Supervisor Agent 및 LangGraph 워크플로우 모듈

전체 레퍼런스 검증 워크플로우를 조율하고 최종 리포트를 생성합니다.

역할:
- 전체 워크플로우 조율 및 상태 관리
- 각 서브에이전트 호출 결정
- 최종 검증 리포트 생성
- LangGraph 기반 스트리밍 지원
"""

import os
from typing import Dict, Any, List, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schema import (
    ReferenceValidationState,
    FullValidationResult,
    ValidationReport,
    ValidationStatus,
    CitationAppropriateness,
    create_initial_state,
)
from config import settings
from agents.parser_agent import parser_agent_node
from agents.web_agent import web_agent_node
from agents.validation_agent import validation_agent_node
from agents.context_agent import context_agent_node
from agents.citation_agent import citation_agent_node


class ReportGenerator:
    """최종 리포트 생성 클래스"""
    
    def __init__(self, llm=None):
        self.llm = llm or settings.get_report_llm()
    
    def generate_summary(self, state: ReferenceValidationState) -> str:
        """LLM을 사용하여 검증 결과 요약을 생성합니다."""
        all_results = state["all_validation_results"]
        
        if not all_results:
            return "검증된 레퍼런스가 없습니다."
        
        # 통계 계산
        total = len(all_results)
        valid = sum(1 for r in all_results if r.existence_result and r.existence_result.status == ValidationStatus.VALID)
        invalid = sum(1 for r in all_results if r.existence_result and r.existence_result.status == ValidationStatus.INVALID)
        unverifiable = sum(1 for r in all_results if r.existence_result and r.existence_result.status == ValidationStatus.UNVERIFIABLE)
        
        # 인용 적절성 통계
        appropriate = sum(
            1 for r in all_results 
            if r.citation_result and r.citation_result.appropriateness_score >= 70
        )
        
        # 문제가 있는 레퍼런스 요약
        issues_summary = []
        for r in all_results:
            if r.existence_result and r.existence_result.status != ValidationStatus.VALID:
                issues_summary.append(f"- {r.ref_id}: {r.existence_result.status.value} - {r.existence_result.reasoning[:100]}")
            if r.citation_result and r.citation_result.appropriateness_score < 70:
                issues_summary.append(f"- {r.ref_id}: 인용 부적절 ({r.citation_result.appropriateness_score:.0f}점) - {r.citation_result.issues[:2] if r.citation_result.issues else '문제 발견'}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 학술 논문 레퍼런스 검증 결과를 요약하는 전문가입니다.
검증 결과를 간결하고 명확하게 요약하고, 실행 가능한 권장 사항을 제시해야 합니다."""),
            ("human", """다음 검증 결과를 요약해 주세요:

논문 제목: {paper_title}

--- 통계 ---
총 레퍼런스: {total}개
- 유효: {valid}개
- 유효하지 않음: {invalid}개
- 검증 불가: {unverifiable}개

적절한 인용: {appropriate}개 (전체의 {appropriate_pct:.1f}%)

--- 문제 있는 레퍼런스 ---
{issues}

위 결과를 바탕으로:
1. 전반적인 검증 결과 요약 (2-3문장)
2. 주요 문제점
3. 권장 수정 사항

을 작성해 주세요.""")
        ])
        
        chain = prompt | self.llm
        
        try:
            result = chain.invoke({
                "paper_title": state.get("paper_title", "알 수 없음"),
                "total": total,
                "valid": valid,
                "invalid": invalid,
                "unverifiable": unverifiable,
                "appropriate": appropriate,
                "appropriate_pct": (appropriate / total * 100) if total > 0 else 0,
                "issues": "\n".join(issues_summary[:10]) if issues_summary else "문제 없음",
            })
            return result.content
            
        except Exception as e:
            print(f"요약 생성 오류: {e}")
            return f"""검증 완료: 총 {total}개 레퍼런스 중 {valid}개 유효, {invalid}개 유효하지 않음, {unverifiable}개 검증 불가.
인용 적절성: {appropriate}개가 적절함 ({appropriate / total * 100:.1f}%)."""


def aggregate_results_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """
    현재 레퍼런스의 검증 결과를 집계하고 다음 레퍼런스로 이동합니다.
    
    Args:
        state: 현재 상태
        
    Returns:
        업데이트된 상태
    """
    print("--- Results Aggregator 실행: 결과 집계 ---")
    
    current_reference = state["current_reference"]
    existence_result = state["existence_result"]
    citation_result = state["citation_result"]
    
    if current_reference and existence_result:
        # 전체 검증 결과 생성
        full_result = FullValidationResult(
            ref_id=current_reference.ref_id,
            reference=current_reference,
            existence_result=existence_result,
            citation_result=citation_result,
            is_valid=existence_result.status == ValidationStatus.VALID,
            overall_score=_calculate_overall_score(existence_result, citation_result),
            summary=_generate_result_summary(existence_result, citation_result),
        )
        
        # 결과 누적
        state["all_validation_results"].append(full_result)
        print(f"  결과 저장: {current_reference.ref_id} -> 종합 {full_result.overall_score:.1f}점")
    
    # 다음 레퍼런스로 이동
    state["current_ref_index"] += 1
    state["current_reference"] = None
    state["existence_result"] = None
    state["citation_result"] = None
    state["context_analyses"] = []
    state["search_results"] = []
    state["retry_count"] = 0
    
    print(f"  다음 레퍼런스 인덱스: {state['current_ref_index']}")
    
    return state


def _calculate_overall_score(existence_result, citation_result) -> float:
    """종합 점수를 계산합니다."""
    score = 0.0
    
    # 존재 검증 점수 (40%)
    if existence_result:
        if existence_result.status == ValidationStatus.VALID:
            score += 40 * existence_result.confidence_score
        elif existence_result.status == ValidationStatus.UNVERIFIABLE:
            score += 20 * existence_result.confidence_score
    
    # 인용 적절성 점수 (60%)
    if citation_result:
        score += 0.6 * citation_result.appropriateness_score
    else:
        # 인용 적절성 검사를 하지 않은 경우
        if existence_result and existence_result.status == ValidationStatus.VALID:
            score += 30  # 기본 점수
    
    return round(score, 1)


def _generate_result_summary(existence_result, citation_result) -> str:
    """개별 결과 요약을 생성합니다."""
    parts = []
    
    if existence_result:
        parts.append(f"존재 검증: {existence_result.status.value} (신뢰도: {existence_result.confidence_score:.2f})")
    
    if citation_result:
        parts.append(f"인용 적절성: {citation_result.appropriateness_score:.0f}점 ({citation_result.appropriateness_grade.value})")
    
    return " | ".join(parts) if parts else "검증 정보 없음"


def generate_final_report_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """
    최종 검증 리포트를 생성합니다.
    
    Args:
        state: 현재 상태
        
    Returns:
        업데이트된 상태
    """
    print("--- Final Report Generator 실행: 최종 리포트 생성 ---")
    
    all_results = state["all_validation_results"]
    
    # 통계 계산
    total = len(all_results)
    valid_count = sum(1 for r in all_results if r.existence_result and r.existence_result.status == ValidationStatus.VALID)
    invalid_count = sum(1 for r in all_results if r.existence_result and r.existence_result.status == ValidationStatus.INVALID)
    unverifiable_count = sum(1 for r in all_results if r.existence_result and r.existence_result.status == ValidationStatus.UNVERIFIABLE)
    
    # 인용 적절성 통계
    appropriate_citations = sum(
        1 for r in all_results 
        if r.citation_result and r.citation_result.appropriateness_score >= 70
    )
    inappropriate_citations = sum(
        1 for r in all_results 
        if r.citation_result and r.citation_result.appropriateness_score < 70
    )
    
    # 평균 적절성 점수
    citation_scores = [
        r.citation_result.appropriateness_score 
        for r in all_results 
        if r.citation_result
    ]
    avg_score = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
    
    # 문제 레퍼런스 목록
    invalid_refs = [
        r.ref_id for r in all_results 
        if r.existence_result and r.existence_result.status == ValidationStatus.INVALID
    ]
    
    inappropriate_list = [
        {
            "ref_id": r.ref_id,
            "score": r.citation_result.appropriateness_score,
            "issues": r.citation_result.issues,
            "reasoning": r.citation_result.reasoning,
        }
        for r in all_results 
        if r.citation_result and r.citation_result.appropriateness_score < 70
    ]
    
    # 권장 사항 생성
    recommendations = []
    if invalid_refs:
        recommendations.append(f"다음 레퍼런스의 존재 여부를 재확인하세요: {', '.join(invalid_refs)}")
    if inappropriate_citations > 0:
        recommendations.append(f"{inappropriate_citations}개의 인용에서 부적절한 사용이 발견되었습니다. 원문을 다시 확인하세요.")
    if avg_score < 70:
        recommendations.append("전반적인 인용 품질 개선이 필요합니다.")
    
    # 요약 생성
    report_generator = ReportGenerator()
    summary = report_generator.generate_summary(state)
    
    # 최종 리포트 생성
    report = ValidationReport(
        paper_title=state.get("paper_title", "알 수 없음"),
        paper_path=state.get("paper_path", ""),
        total_references=total,
        valid_count=valid_count,
        invalid_count=invalid_count,
        unverifiable_count=unverifiable_count,
        appropriate_citations=appropriate_citations,
        inappropriate_citations=inappropriate_citations,
        average_appropriateness_score=round(avg_score, 1),
        results=all_results,
        invalid_references=invalid_refs,
        inappropriate_citations_list=inappropriate_list,
        summary=summary,
        recommendations=recommendations,
    )
    
    state["final_report"] = report
    
    print(f"\n=== 최종 리포트 ===")
    print(f"논문: {report.paper_title}")
    print(f"총 레퍼런스: {report.total_references}개")
    print(f"  - 유효: {report.valid_count}개")
    print(f"  - 유효하지 않음: {report.invalid_count}개")
    print(f"  - 검증 불가: {report.unverifiable_count}개")
    print(f"인용 적절성: 평균 {report.average_appropriateness_score:.1f}점")
    print(f"권장 사항: {len(report.recommendations)}개")
    
    return state


def should_continue_validation(state: ReferenceValidationState) -> Literal["continue", "finish"]:
    """
    다음 레퍼런스로 계속할지 종료할지 결정합니다.
    
    Args:
        state: 현재 상태
        
    Returns:
        "continue" 또는 "finish"
    """
    current_index = state["current_ref_index"]
    total_refs = len(state["references"])
    
    if current_index >= total_refs:
        print(f"  모든 레퍼런스 처리 완료 ({total_refs}개)")
        return "finish"
    else:
        print(f"  다음 레퍼런스로 계속 ({current_index + 1}/{total_refs})")
        return "continue"


def build_graph() -> StateGraph:
    """
    LangGraph 워크플로우를 구축합니다.
    
    Returns:
        컴파일된 StateGraph
    """
    # 그래프 생성
    workflow = StateGraph(ReferenceValidationState)
    
    # 노드 추가
    workflow.add_node("parser", parser_agent_node)
    workflow.add_node("web_search", web_agent_node)
    workflow.add_node("validation", validation_agent_node)
    workflow.add_node("context_analysis", context_agent_node)
    workflow.add_node("citation_validation", citation_agent_node)
    workflow.add_node("aggregate_results", aggregate_results_node)
    workflow.add_node("generate_report", generate_final_report_node)
    
    # 엣지 정의
    # 시작점: Parser Agent
    workflow.set_entry_point("parser")
    
    # Parser -> Web Search (첫 번째 레퍼런스)
    workflow.add_edge("parser", "web_search")
    
    # Web Search -> Validation
    workflow.add_edge("web_search", "validation")
    
    # Validation -> Context Analysis
    workflow.add_edge("validation", "context_analysis")
    
    # Context Analysis -> Citation Validation
    workflow.add_edge("context_analysis", "citation_validation")
    
    # Citation Validation -> Aggregate Results
    workflow.add_edge("citation_validation", "aggregate_results")
    
    # Aggregate Results -> 조건부 분기
    workflow.add_conditional_edges(
        "aggregate_results",
        should_continue_validation,
        {
            "continue": "web_search",      # 다음 레퍼런스 검증
            "finish": "generate_report"    # 최종 리포트 생성
        }
    )
    
    # Generate Report -> END
    workflow.add_edge("generate_report", END)
    
    # 컴파일
    return workflow.compile()


def run_validation(
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
        ValidationReport
    """
    print(f"\n{'='*60}")
    print(f"논문 레퍼런스 검증 시작")
    print(f"파일: {pdf_path}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 초기 상태 생성
    initial_state = create_initial_state(pdf_path)
    
    # 그래프 구축 및 실행
    graph = build_graph()
    
    # 실행
    final_state = graph.invoke(initial_state)
    
    # 최대 레퍼런스 수 제한이 있으면 적용
    if max_references and len(final_state["references"]) > max_references:
        final_state["references"] = final_state["references"][:max_references]
    
    print(f"\n{'='*60}")
    print(f"검증 완료")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    return final_state["final_report"]


if __name__ == "__main__":
    # 테스트
    print("=== Supervisor 테스트 ===")
    
    # 그래프 구조 출력
    graph = build_graph()
    print("\n그래프 노드:", graph.nodes.keys())
    
    # 테스트 실행 (실제 PDF 필요)
    test_pdf = "workspace/input/test_paper.pdf"
    
    if os.path.exists(test_pdf):
        report = run_validation(test_pdf, max_references=3)
        print(f"\n최종 리포트 요약:")
        print(report.summary)
    else:
        print(f"\n테스트 PDF가 없습니다: {test_pdf}")
        print("실제 테스트를 위해 PDF 파일을 workspace/input/ 에 넣어주세요.")
