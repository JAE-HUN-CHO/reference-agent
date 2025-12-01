"""
Parser Agent 모듈

논문 PDF에서 텍스트를 추출하고, 레퍼런스 목록 및 인용 문맥을 파싱합니다.

역할:
- PDF 텍스트 추출
- 레퍼런스 섹션 식별 및 추출
- 개별 레퍼런스 파싱 (저자, 제목, 연도, DOI 등)
- 본문 내 인용 위치 매핑
"""

import re
import os
from typing import List, Optional, Tuple
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import (
    Reference,
    CitationContext,
    ReferenceValidationState,
    ReferenceListOutput,
    CitationContextListOutput,
)
from config import settings


class ParserAgent:
    """논문 파싱을 담당하는 에이전트"""
    
    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain 호환 LLM 인스턴스 (None이면 설정에서 가져옴)
        """
        self.llm = llm or settings.get_parser_llm()
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, str]:
        """
        PDF 파일에서 전체 텍스트를 추출합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            (전체 텍스트, 논문 제목) 튜플
        """
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts)
        
        # 제목 추출 (첫 페이지에서)
        title = self._extract_title(text_parts[0] if text_parts else "")
        
        return full_text, title
    
    def _extract_title(self, first_page_text: str) -> str:
        """첫 페이지에서 논문 제목을 추출합니다."""
        lines = first_page_text.split('\n')
        
        # 첫 몇 줄에서 제목 찾기 (보통 가장 긴 줄이 제목)
        candidate_lines = []
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                candidate_lines.append(line)
        
        if candidate_lines:
            # 가장 긴 줄을 제목으로 가정
            title = max(candidate_lines, key=len)
            return title[:200]  # 최대 200자
        
        return "Unknown Title"
    
    def _find_references_section(self, paper_content: str) -> Tuple[str, str]:
        """
        레퍼런스 섹션과 본문을 분리합니다.
        
        Returns:
            (본문 텍스트, 레퍼런스 섹션 텍스트) 튜플
        """
        # 레퍼런스 섹션 시작 패턴
        patterns = [
            r'\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n',
            r'\n\s*\d+\.\s*(References|REFERENCES)\s*\n',
            r'\n\s*References\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, paper_content, re.MULTILINE)
            if match:
                body_text = paper_content[:match.start()]
                references_text = paper_content[match.end():]
                return body_text, references_text
        
        # 패턴을 찾지 못한 경우, 마지막 20%를 레퍼런스로 가정
        split_point = int(len(paper_content) * 0.8)
        return paper_content[:split_point], paper_content[split_point:]
    
    def parse_references_with_llm(self, references_text: str) -> List[Reference]:
        """
        LLM을 사용하여 레퍼런스 텍스트에서 개별 레퍼런스를 추출합니다.
        
        Args:
            references_text: 레퍼런스 섹션 텍스트
            
        Returns:
            Reference 객체 리스트
        """
        parser = PydanticOutputParser(pydantic_object=ReferenceListOutput)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 학술 논문의 레퍼런스 섹션을 분석하는 전문가입니다.
제공된 레퍼런스 텍스트를 분석하여 각 레퍼런스의 정보를 구조화된 JSON 형식으로 추출해야 합니다.

추출해야 하는 정보:
- ref_id: 레퍼런스 번호 (예: "[1]", "[2]" 또는 숫자만)
- raw_text: 원본 레퍼런스 텍스트 전체
- title: 논문/책 제목 (인용부호 안의 텍스트 또는 이탤릭체 부분)
- authors: 저자 목록 (배열 형태)
- year: 출판 연도 (4자리 숫자)
- venue: 저널/학회명 (있는 경우)
- doi: DOI 식별자 (있는 경우)
- url: URL (있는 경우)

{format_instructions}"""),
            ("human", """다음 레퍼런스 섹션 텍스트를 분석하여 각 레퍼런스를 추출해 주세요:

---
{references_text}
---

위 텍스트에서 각 레퍼런스를 JSON 형식으로 추출하세요.""")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        chain = prompt | self.llm | parser
        
        try:
            # 텍스트가 너무 길면 잘라서 처리
            max_length = 15000
            if len(references_text) > max_length:
                references_text = references_text[:max_length]
            
            result = chain.invoke({"references_text": references_text})
            return result.references
            
        except Exception as e:
            print(f"LLM 레퍼런스 파싱 오류: {e}")
            # 폴백: 규칙 기반 파싱 시도
            return self._parse_references_rule_based(references_text)
    
    def _parse_references_rule_based(self, references_text: str) -> List[Reference]:
        """
        규칙 기반으로 레퍼런스를 파싱합니다 (LLM 실패 시 폴백).
        """
        references = []
        
        # [1], [2] 등의 패턴으로 분리
        pattern = r'\[(\d+)\]'
        matches = list(re.finditer(pattern, references_text))
        
        for i, match in enumerate(matches):
            ref_id = f"[{match.group(1)}]"
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(references_text)
            
            raw_text = references_text[start:end].strip()
            
            # 연도 추출
            year_match = re.search(r'\b(19|20)\d{2}\b', raw_text)
            year = int(year_match.group()) if year_match else None
            
            # DOI 추출
            doi_match = re.search(r'10\.\d{4,}/[^\s]+', raw_text)
            doi = doi_match.group() if doi_match else None
            
            references.append(Reference(
                ref_id=ref_id,
                raw_text=raw_text,
                year=year,
                doi=doi,
            ))
        
        return references
    
    def parse_citation_contexts_with_llm(
        self,
        body_text: str,
        references: List[Reference]
    ) -> List[CitationContext]:
        """
        LLM을 사용하여 본문에서 인용 문맥을 추출합니다.
        
        Args:
            body_text: 논문 본문 텍스트
            references: 추출된 레퍼런스 목록
            
        Returns:
            CitationContext 객체 리스트
        """
        parser = PydanticOutputParser(pydantic_object=CitationContextListOutput)
        
        # 레퍼런스 ID 목록
        ref_ids = [ref.ref_id for ref in references]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 학술 논문의 인용 문맥을 분석하는 전문가입니다.
논문 본문에서 레퍼런스가 인용된 위치를 찾아 해당 문맥을 추출해야 합니다.

추출해야 하는 정보:
- context_id: 고유 ID (예: "ctx_1", "ctx_2")
- ref_id: 인용된 레퍼런스 ID (제공된 목록과 일치해야 함: {ref_ids})
- text_snippet: 인용이 포함된 문장 (인용 전후 맥락 포함)
- paragraph_current: 인용이 포함된 현재 문단 전체
- section: 섹션명 (알 수 있는 경우)

주의사항:
- [1], [2] 또는 [1, 2, 3] 형식의 인용 마커를 찾으세요
- 하나의 레퍼런스가 여러 번 인용될 수 있습니다
- 인용된 문맥의 의미가 명확하도록 충분한 텍스트를 포함하세요

{format_instructions}"""),
            ("human", """다음 논문 본문에서 인용 문맥을 추출해 주세요:

레퍼런스 ID 목록: {ref_ids}

---
{body_text}
---

위 텍스트에서 각 인용 문맥을 JSON 형식으로 추출하세요.""")
        ]).partial(
            format_instructions=parser.get_format_instructions(),
            ref_ids=str(ref_ids)
        )
        
        chain = prompt | self.llm | parser
        
        try:
            # 텍스트가 너무 길면 잘라서 처리
            max_length = 20000
            if len(body_text) > max_length:
                body_text = body_text[:max_length]
            
            result = chain.invoke({
                "body_text": body_text,
                "ref_ids": str(ref_ids)
            })
            return result.citation_contexts
            
        except Exception as e:
            print(f"LLM 인용 문맥 파싱 오류: {e}")
            # 폴백: 규칙 기반 파싱 시도
            return self._parse_citation_contexts_rule_based(body_text, references)
    
    def _parse_citation_contexts_rule_based(
        self,
        body_text: str,
        references: List[Reference]
    ) -> List[CitationContext]:
        """
        규칙 기반으로 인용 문맥을 파싱합니다 (LLM 실패 시 폴백).
        """
        contexts = []
        context_id = 0
        
        # 문장 단위로 분리
        sentences = re.split(r'(?<=[.!?])\s+', body_text)
        
        for sentence in sentences:
            # [숫자] 패턴 찾기
            citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
            matches = re.findall(citation_pattern, sentence)
            
            for match in matches:
                # 쉼표로 구분된 여러 레퍼런스 처리
                ref_nums = [n.strip() for n in match.split(',')]
                
                for ref_num in ref_nums:
                    ref_id = f"[{ref_num}]"
                    context_id += 1
                    
                    contexts.append(CitationContext(
                        context_id=f"ctx_{context_id}",
                        ref_id=ref_id,
                        text_snippet=sentence.strip(),
                        paragraph_current=sentence.strip(),
                    ))
        
        return contexts


def parser_agent_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """
    LangGraph 노드 역할을 하는 Parser Agent 함수
    
    Args:
        state: 현재 상태
        
    Returns:
        업데이트된 상태
    """
    print("--- Parser Agent 실행: PDF 텍스트 추출 및 레퍼런스 파싱 시작 ---")
    
    agent = ParserAgent()
    pdf_path = state["paper_path"]
    
    # 처리 로그 추가
    state["processing_log"].append(f"PDF 파싱 시작: {pdf_path}")
    
    try:
        # 1. PDF 텍스트 추출
        print("  [1/4] PDF 텍스트 추출 중...")
        paper_content, paper_title = agent.extract_text_from_pdf(pdf_path)
        state["paper_content"] = paper_content
        state["paper_title"] = paper_title
        print(f"       완료: {len(paper_content):,} 문자, 제목: {paper_title[:50]}...")
        
        # 2. 레퍼런스 섹션 분리
        print("  [2/4] 레퍼런스 섹션 분리 중...")
        body_text, references_text = agent._find_references_section(paper_content)
        print(f"       본문: {len(body_text):,} 문자, 레퍼런스: {len(references_text):,} 문자")
        
        # 3. 레퍼런스 목록 추출
        print("  [3/4] 레퍼런스 목록 추출 중...")
        references = agent.parse_references_with_llm(references_text)
        state["references"] = references
        print(f"       완료: {len(references)}개 레퍼런스 추출됨")
        
        # 4. 인용 문맥 추출
        print("  [4/4] 인용 문맥 추출 중...")
        citation_contexts = agent.parse_citation_contexts_with_llm(body_text, references)
        state["citation_contexts"] = citation_contexts
        print(f"       완료: {len(citation_contexts)}개 인용 문맥 추출됨")
        
        # 초기 상태 설정
        state["current_ref_index"] = 0
        state["all_validation_results"] = []
        
        state["processing_log"].append(
            f"파싱 완료: {len(references)}개 레퍼런스, {len(citation_contexts)}개 인용 문맥"
        )
        
    except Exception as e:
        error_msg = f"Parser Agent 오류: {e}"
        print(f"  ERROR: {error_msg}")
        state["error_log"].append(error_msg)
    
    return state


if __name__ == "__main__":
    # 테스트
    print("=== Parser Agent 테스트 ===")
    
    # 테스트용 상태 생성
    from schema import create_initial_state
    
    # 테스트 PDF 경로 (실제 파일이 필요)
    test_pdf = "workspace/input/test_paper.pdf"
    
    if os.path.exists(test_pdf):
        state = create_initial_state(test_pdf)
        result_state = parser_agent_node(state)
        
        print(f"\n추출된 레퍼런스: {len(result_state['references'])}개")
        for ref in result_state['references'][:3]:
            print(f"  {ref.ref_id}: {ref.title or ref.raw_text[:50]}...")
        
        print(f"\n추출된 인용 문맥: {len(result_state['citation_contexts'])}개")
        for ctx in result_state['citation_contexts'][:3]:
            print(f"  {ctx.context_id} -> {ctx.ref_id}: {ctx.text_snippet[:50]}...")
    else:
        print(f"테스트 PDF가 없습니다: {test_pdf}")
