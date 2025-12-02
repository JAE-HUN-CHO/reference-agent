#!/usr/bin/env python3
"""
논문 레퍼런스 검증 시스템 - 메인 실행 파일

Deep Agent 아키텍처 기반 (teddynote-lab/deep-agents-from-scratch)

사용법:
    # 기본 실행 (Upstage solar-pro2)
    python main.py paper.pdf

    # Upstage 다른 모델
    python main.py paper.pdf --provider upstage --model solar-mini

    # Ollama 모델
    python main.py paper.pdf --provider ollama --model llama3.2

    # OpenAI
    python main.py paper.pdf --provider openai --model gpt-4o

    # Google Gemini
    python main.py paper.pdf --provider google --model gemini-2.5-pro

    # Deep Agent 모드 (기본)
    python main.py paper.pdf --mode deep-agent

    # 기존 LangGraph 워크플로우 모드
    python main.py paper.pdf --mode langgraph

    # 최대 레퍼런스 수 제한
    python main.py paper.pdf --max-refs 10

    # 출력 파일 지정
    python main.py paper.pdf --output results.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 

# 현재 디렉토리를 파이썬 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import configure, ModelPresets, settings
from schema import ValidationReport


def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="논문 레퍼런스 검증 Multi-Agent 시스템 (Deep Agent Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # Deep Agent 모드 (기본, 권장)
    python main.py paper.pdf
    
    # OpenAI 사용
    python main.py paper.pdf --provider openai --model gpt-4o
    
    # 기존 LangGraph 워크플로우 모드
    python main.py paper.pdf --mode langgraph
    
    # Tavily 검색 활성화
    export TAVILY_API_KEY=your_key
    python main.py paper.pdf

Deep Agent 아키텍처 특징:
    - Context Offloading: 가상 파일 시스템으로 컨텍스트 관리
    - Sub-agent Delegation: 병렬 검색을 위한 서브 에이전트 위임
    - Strategic Thinking: 전략적 사고 도구로 검증 품질 향상
    - TODO Management: 복잡한 워크플로우 진행 상황 추적
        """
    )
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="검증할 PDF 논문 파일 경로"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="deep-agent",
        choices=["deep-agent", "langgraph"],
        help="실행 모드 (기본: deep-agent)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="upstage",
        choices=["upstage", "ollama", "openai", "anthropic", "google"],
        help="LLM 제공자 (기본: upstage)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="solar-pro2",
        help="사용할 모델 이름 (기본: solar-pro2)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM 생성 온도 (기본: 0.1)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="커스텀 API URL (Ollama 등)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API 키 (환경 변수로도 설정 가능)"
    )
    
    parser.add_argument(
        "--tavily-key",
        type=str,
        default=None,
        help="Tavily API 키 (환경 변수 TAVILY_API_KEY로도 설정 가능)"
    )
    
    parser.add_argument(
        "--max-refs",
        type=int,
        default=None,
        help="최대 검증할 레퍼런스 수"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과를 저장할 JSON 파일 경로"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="상세 로그 출력 비활성화"
    )
    
    parser.add_argument(
        "--max-agents",
        type=int,
        default=3,
        help="Deep Agent 모드에서 최대 병렬 서브 에이전트 수 (기본: 3)"
    )
    
    return parser.parse_args()


def save_report(report: ValidationReport, output_path: str):
    """검증 리포트를 JSON 파일로 저장합니다."""
    # Pydantic 모델을 딕셔너리로 변환
    report_dict = {
        "paper_title": report.paper_title,
        "paper_path": report.paper_path,
        "generated_at": datetime.now().isoformat(),
        "statistics": {
            "total_references": report.total_references,
            "valid_count": report.valid_count,
            "invalid_count": report.invalid_count,
            "unverifiable_count": report.unverifiable_count,
            "appropriate_citations": report.appropriate_citations,
            "inappropriate_citations": report.inappropriate_citations,
            "average_appropriateness_score": report.average_appropriateness_score,
        },
        "invalid_references": report.invalid_references,
        "inappropriate_citations": report.inappropriate_citations_list,
        "summary": report.summary,
        "recommendations": report.recommendations,
        "detailed_results": [
            {
                "ref_id": r.ref_id,
                "reference": {
                    "title": r.reference.title,
                    "authors": r.reference.authors,
                    "year": r.reference.year,
                    "raw_text": r.reference.raw_text,
                },
                "existence": {
                    "status": r.existence_result.status.value if r.existence_result else None,
                    "confidence": r.existence_result.confidence_score if r.existence_result else None,
                    "reasoning": r.existence_result.reasoning if r.existence_result else None,
                } if r.existence_result else None,
                "citation": {
                    "score": r.citation_result.appropriateness_score if r.citation_result else None,
                    "grade": r.citation_result.appropriateness_grade.value if r.citation_result else None,
                    "issues": r.citation_result.issues if r.citation_result else [],
                } if r.citation_result else None,
                "is_valid": r.is_valid,
                "overall_score": r.overall_score,
            }
            for r in report.results
        ],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 저장되었습니다: {output_path}")


def save_deep_agent_result(result: dict, output_path: str):
    """Deep Agent 결과를 JSON 파일로 저장합니다."""
    result_dict = {
        "paper_path": result.get("paper_path", ""),
        "generated_at": datetime.now().isoformat(),
        "mode": "deep-agent",
        "success": result.get("success", False),
        "output": result.get("output", ""),
        "error": result.get("error", None),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 저장되었습니다: {output_path}")


def print_report_summary(report: ValidationReport):
    """검증 리포트 요약을 출력합니다."""
    print("\n" + "=" * 70)
    print("검증 결과 요약")
    print("=" * 70)
    
    print(f"\n📄 논문: {report.paper_title}")
    print(f"📊 총 레퍼런스: {report.total_references}개")
    print()
    
    # 존재 검증 결과
    print("🔍 존재 검증 결과:")
    print(f"   ✅ 유효: {report.valid_count}개")
    print(f"   ❌ 유효하지 않음: {report.invalid_count}개")
    print(f"   ❓ 검증 불가: {report.unverifiable_count}개")
    print()
    
    # 인용 적절성 결과
    print("📝 인용 적절성:")
    print(f"   ✅ 적절: {report.appropriate_citations}개")
    print(f"   ❌ 부적절: {report.inappropriate_citations}개")
    print(f"   📈 평균 점수: {report.average_appropriateness_score:.1f}/100")
    print()
    
    # 문제 레퍼런스
    if report.invalid_references:
        print("⚠️ 문제가 있는 레퍼런스:")
        for ref_id in report.invalid_references[:5]:
            print(f"   - {ref_id}")
        if len(report.invalid_references) > 5:
            print(f"   ... 외 {len(report.invalid_references) - 5}개")
        print()
    
    # 권장 사항
    if report.recommendations:
        print("💡 권장 사항:")
        for rec in report.recommendations[:3]:
            print(f"   • {rec}")
        print()
    
    # 요약
    print("-" * 70)
    print("📋 상세 요약:")
    print(report.summary)
    print("=" * 70)


def print_deep_agent_result(result: dict):
    """Deep Agent 결과를 출력합니다."""
    print("\n" + "=" * 70)
    print("Deep Agent 검증 결과")
    print("=" * 70)
    
    print(f"\n📄 논문: {result.get('paper_path', 'Unknown')}")
    print(f"✅ 성공: {result.get('success', False)}")
    
    if result.get("error"):
        print(f"⚠️ 오류: {result.get('error')}")
    
    print("\n" + "-" * 70)
    print("📋 결과:")
    print(result.get("output", "No output"))
    print("=" * 70)


def run_deep_agent_mode(args):
    """Deep Agent 모드로 검증을 실행합니다."""
    from deep_agent import create_deep_agent_supervisor
    
    print(f"\n🤖 Deep Agent 모드로 실행")
    print(f"   병렬 에이전트 수: {args.max_agents}")
    
    # LLM 생성
    model = settings.get_parser_llm()
    
    # Deep Agent Supervisor 생성
    supervisor = create_deep_agent_supervisor(
        model=model,
        tavily_api_key=settings.tavily_api_key or args.tavily_key,
        verbose=not args.quiet,
    )
    
    # 검증 실행
    result = supervisor.validate_paper(
        args.pdf_path,
        max_references=args.max_refs,
    )
    
    return result


def run_langgraph_mode(args):
    """기존 LangGraph 워크플로우 모드로 검증을 실행합니다."""
    from supervisor import run_validation
    
    print(f"\n📊 LangGraph 워크플로우 모드로 실행")
    
    report = run_validation(
        args.pdf_path,
        max_references=args.max_refs,
        verbose=not args.quiet
    )
    
    return report


def main():
    """메인 함수"""
    args = parse_args()
    
    # PDF 파일 확인
    if not os.path.exists(args.pdf_path):
        print(f"오류: 파일을 찾을 수 없습니다: {args.pdf_path}")
        sys.exit(1)
    
    # LLM 설정
    print(f"\n🔧 설정:")
    print(f"   모드: {args.mode}")
    print(f"   LLM: {args.provider}/{args.model}")
    print(f"   Temperature: {args.temperature}")
    
    configure(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        api_key=args.api_key,
        tavily_api_key=args.tavily_key,
    )
    
    # Tavily 설정 확인
    if settings.tavily_api_key:
        print("   Tavily: 활성화됨")
    else:
        print("   Tavily: 비활성화 (TAVILY_API_KEY 환경 변수로 설정 가능)")
    
    # 검증 실행
    try:
        if args.mode == "deep-agent":
            # Deep Agent 모드
            result = run_deep_agent_mode(args)
            
            # 결과 출력
            print_deep_agent_result(result)
            
            # 결과 저장
            if args.output:
                save_deep_agent_result(result, args.output)
            else:
                pdf_name = Path(args.pdf_path).stem
                default_output = f"deep_agent_result_{pdf_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_deep_agent_result(result, default_output)
                
        else:
            # LangGraph 모드
            report = run_langgraph_mode(args)
            
            # 결과 출력
            print_report_summary(report)
            
            # 결과 저장
            if args.output:
                save_report(report, args.output)
            else:
                pdf_name = Path(args.pdf_path).stem
                default_output = f"validation_result_{pdf_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_report(report, default_output)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
